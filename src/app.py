"""
AI Data Analyst - NiceGUI frontend.

    pip install nicegui
    python app.py

Architecture notes (read this before changing anything):

MULTI-USER SAFETY
    Every mutable piece of an investigation - InvestigationState,
    InvestigationBudget, each turn's step-event queue - is created
    INSIDE the @ui.page('/') function, as a local variable (or a local
    variable of a nested function closing over it). NiceGUI calls
    main_page() once per browser connection, so each user gets their
    own closure over their own objects. There is exactly one
    intentional module-level object: `session_store` (imported from
    src.session_store) - a thread-safe store keyed by investigation_id.

    Do not add a module-level `state = InvestigationState(...)` or
    similar. That is the classic way a demo like this breaks the moment
    a second person opens the page.

CHAT-STYLE PERSISTENCE
    Each question creates a new "turn" - a permanent block appended to
    `conversation_column` containing: the user's question, that turn's
    OWN step log, and that turn's OWN answer card. Turns are never
    cleared or overwritten; asking a new question appends another turn
    below the previous ones, like a chat thread. Only the input row at
    the top is reused across turns.

BLOCKING WORK / CONCURRENCY
    run_investigation() makes real, blocking LLM/DB calls. NiceGUI is
    async - a blocking call on the main thread would freeze the UI for
    EVERY connected user. Every call into the controller goes through
    `nicegui.run.io_bound(...)`, which runs it in a worker thread.

LIVE STEP UPDATES ACROSS THE THREAD BOUNDARY
    on_event fires from that WORKER THREAD, where touching NiceGUI UI
    elements directly is not safe. Each turn gets its own
    queue.Queue(); on_event just does queue.put() (thread-safe). A
    per-turn ui.timer polls that queue back on the UI/event-loop thread
    and is the only thing that touches that turn's ui.log. The timer is
    stopped once the turn reaches a terminal state, so idle timers
    don't accumulate over a long session.

HORIZONTAL SCALING (multiple processes/machines)
    Swapping InMemorySessionStore for a Redis/Postgres-backed one (see
    src/session_store.py) is what lets investigation state survive past
    a single process. NiceGUI's own websocket connection still needs
    sticky sessions at the load balancer - true of any websocket app.
"""

import queue

from nicegui import run, ui

from src.planner_models import InvestigationState
from src.budget import InvestigationBudget
from src.worker_interface import SqlAgWorker
from src.controller import run_investigation, resume_with_clarification
from src.session_store import session_store
from src.followup_resolver import resolve_followup


def _format_event(event: dict) -> "str | None":
    """
    Turn a controller event into a step-log line. Returns None for
    events not worth showing (kept out of the visible log entirely,
    rather than shown as noise).

    The guiding rule: every visible line should carry an actual fact
    (the real subquestion, the real finding, the real question) - never
    a bare generic status word.
    """
    t = event["type"]

    if t == "planning":
        return None  # internal step; "investigate_start"/"clarify" already say what happened next

    if t == "investigation_started":
        return "🔎 Starting investigation"
    if t == "investigate_start":
        return f"📊 Querying: {event['subquestion']}"
    if t == "duplicate_skipped":
        return f"↩️ Already answered, skipping: {event['subquestion']}"
    if t == "duplicate_clarify_skipped":
        return f"↩️ Already know '{event['question']}' -> {event.get('reused_answer', '')} - not asking again"
    if t == "evidence":
        # Show the actual finding, not just a counter. Capped so a
        # worker answer that lists many raw IDs never floods the live
        # log - this is a display-layer backstop; the FULL claim is
        # still recorded untouched in state.evidence for the evidence
        # panel/audit trail, only what's shown here is shortened.
        claim = event.get("claim", "").strip()
        if not claim:
            return "✅ Evidence recorded"
        if len(claim) > 220:
            claim = claim[:220].rsplit(" ", 1)[0] + "\u2026 (see Evidence below)"
        return f"✅ {claim}"
    if t == "clarify":
        return f"❓ {event['question']}"
    if t == "synthesizing":
        return "✍️ Writing final answer from recorded evidence"
    if t == "complete":
        return "🏁 Done"
    if t == "stopped":
        return f"⛔ Stopped: {event.get('reason', '')}"
    if t == "forced_termination":
        return f"⚠️ {event.get('reason', '')}"
    return t


def _scroll_to_bottom() -> None:
    ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")


@ui.page("/")
def main_page():
    # --- everything below is local to THIS browser connection ---

    worker = SqlAgWorker()
    busy = {"value": False}  # one investigation in flight at a time per user

    # Local to this connection - each browser tab/user gets its OWN
    # history, so follow-up resolution ("that year", "those sellers")
    # never mixes context between different users. This is the actual
    # fix for "the agent doesn't remember the previous turn": the CHAT
    # UI was already persistent (turns never got cleared), but each
    # turn started a brand-new InvestigationState with zero memory of
    # earlier turns, so references like "that year" had nothing to
    # resolve against and silently defaulted to whatever the planner
    # picked on its own.
    conversation_history: list = []

    # Same idea, for clarify answers specifically. The controller
    # already refuses to re-ask a question that's in
    # state.resolved_ambiguities - but that guard only sees ONE turn's
    # state. Without this, "which metric = worst" gets answered once,
    # then asked again from scratch on the very next unrelated-looking
    # question, because a fresh InvestigationState always starts with
    # resolved_ambiguities = {}. Seeding every new turn's state with
    # everything resolved so far makes the guard effective ACROSS
    # turns, not just within one.
    sticky_resolved_ambiguities: dict = {}

    ui.label("AI Data Analyst").classes("text-2xl font-bold")
    ui.label(
        "Ask an analytical question. Ambiguous questions may pause for "
        "a quick clarification."
    ).classes("text-sm text-gray-500")

    with ui.row().classes("w-full items-center gap-2"):
        question_input = ui.input(
            placeholder="e.g. Why did sales drop at the end of 2018?"
        ).classes("flex-grow")
        run_button = ui.button("Investigate")
        spinner = ui.spinner(size="md")
        spinner.visible = False

    ui.separator()

    # Every turn is appended here, permanently. Nothing in this column
    # is ever cleared - that's the "chat window" persistence.
    conversation_column = ui.column().classes("w-full gap-4 mt-2")

    def new_turn(question: str) -> dict:
        """
        Build the persistent UI for one turn (one question) and return
        the pieces later code needs to update it. Everything here is
        local to this call - two turns never share elements.
        """
        with conversation_column:
            with ui.card().classes("w-full bg-blue-50"):
                ui.label(question).classes("font-medium")
                interpreted_label = ui.label("").classes(
                    "text-xs text-gray-500 italic"
                )
                interpreted_label.visible = False

            with ui.card().classes("w-full") as turn_card:
                turn_status = ui.label("Investigating\u2026").classes(
                    "text-sm text-gray-500"
                )
                turn_log = ui.log(max_lines=200).classes(
                    "w-full h-32 bg-gray-50 text-xs"
                )
                turn_answer = ui.markdown("")
                turn_evidence = ui.expansion("Evidence")
                turn_evidence.visible = False

        return {
            "card": turn_card,
            "status": turn_status,
            "log": turn_log,
            "answer": turn_answer,
            "interpreted": interpreted_label,
            "evidence": turn_evidence,
        }

    def render_evidence(evidence_el, state) -> None:
        evidence_el.clear()
        evidence_el.visible = bool(state.evidence)
        with evidence_el:
            for i, ev in enumerate(state.evidence, 1):
                ui.label(f"{i}. {ev['subquestion']}").classes("font-medium")
                if ev.get("sql"):
                    ui.code(ev["sql"], language="sql").classes("w-full text-xs")

    async def show_clarify(question: str, options: list, rationale: str) -> str:
        # One dialog, built fresh per call so it always reflects the
        # CURRENT question/options/rationale (no stale content from a
        # previous turn).
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label(question).classes("font-medium")
            if rationale:
                ui.label(rationale).classes("text-sm text-gray-500 italic")
            options_row = ui.row()
            with options_row:
                for opt in options:
                    ui.button(
                        opt, on_click=lambda o=opt: dialog.submit(o)
                    ).props("outline")
            free_text = ui.input(placeholder="or type your own answer")
            ui.button("Submit", on_click=lambda: dialog.submit(free_text.value))

        result = await dialog
        dialog.delete()
        return result or ""

    async def run_turn(question: str) -> None:
        turn = new_turn(question)
        step_queue: "queue.Queue[dict]" = queue.Queue()

        def on_event(event: dict) -> None:
            # WORKER THREAD - must stay cheap/thread-safe.
            step_queue.put(event)

        def drain() -> None:
            # UI-THREAD (ui.timer) - safe to touch turn_log here.
            while True:
                try:
                    event = step_queue.get_nowait()
                except queue.Empty:
                    return
                line = _format_event(event)
                if line is not None:
                    turn["log"].push(line)

        timer = ui.timer(0.25, drain)

        # Resolve follow-up references ("that year", "those sellers")
        # against this connection's OWN conversation history, before
        # the planner ever sees the question. resolve_followup() makes
        # a blocking LLM call, so it goes through run.io_bound like
        # everything else that talks to a model.
        resolved_question = await run.io_bound(
            resolve_followup, question, list(conversation_history),
        )
        if resolved_question != question:
            turn["interpreted"].set_text(f"Interpreted as: {resolved_question}")
            turn["interpreted"].visible = True

        state = InvestigationState(
            question=resolved_question,
            resolved_ambiguities=dict(sticky_resolved_ambiguities),
        )
        budget = InvestigationBudget(max_queries=6, max_clarifications=2)
        investigation_id = session_store.create(question)

        result = await run.io_bound(
            run_investigation, state, worker, budget, on_event=on_event,
        )

        while result.status == "needs_clarification":
            answer = await show_clarify(
                result.clarification_question,
                result.clarification_options,
                result.clarification_rationale,
            )
            result = await run.io_bound(
                resume_with_clarification,
                result.state, answer, worker, budget,
                dimension=result.clarification_question,
                on_event=on_event,
            )

        drain()          # flush anything queued right before completion
        timer.active = False  # stop polling - this turn is done

        # Carry forward anything resolved this turn (including whatever
        # was seeded in) so the NEXT turn's fresh InvestigationState
        # starts already knowing it, instead of re-asking.
        if result.state is not None:
            sticky_resolved_ambiguities.update(
                getattr(result.state, "resolved_ambiguities", {}) or {}
            )

        session_store.update(
            investigation_id,
            status=result.status,
            answer=result.answer or result.reason,
            evidence_count=len(result.state.evidence) if result.state else 0,
        )

        if result.status == "complete":
            turn["status"].set_text("Complete")
            turn["answer"].set_content(result.answer)
            render_evidence(turn["evidence"], result.state)
        else:
            turn["status"].set_text("Stopped")
            turn["answer"].set_content(f"*Stopped: {result.reason}*")

        # Record the ORIGINAL question (natural conversation transcript)
        # so future turns' follow-up resolution has real history to
        # work from - this is what makes the chat's memory actually
        # persistent, not just its on-screen layout.
        conversation_history.append({
            "question": question,
            "answer": result.answer or result.reason,
        })

        _scroll_to_bottom()

    async def start_investigation() -> None:
        question = question_input.value.strip()
        if not question:
            ui.notify("Enter a question first", type="warning")
            return
        if busy["value"]:
            ui.notify("An investigation is already running", type="warning")
            return

        busy["value"] = True
        run_button.disable()
        spinner.visible = True
        question_input.value = ""

        try:
            await run_turn(question)
        finally:
            busy["value"] = False
            run_button.enable()
            spinner.visible = False

    run_button.on_click(start_investigation)
    question_input.on("keydown.enter", start_investigation)


ui.run(title="AI Data Analyst")