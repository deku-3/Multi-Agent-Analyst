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
    OWN step trail, and that turn's OWN answer. Turns are never cleared
    or overwritten; asking a new question appends another turn below
    the previous ones. Only the input row at the top is reused.

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
    and is the only thing that touches that turn's step trail. The
    timer is stopped once the turn reaches a terminal state.

VISUAL DESIGN
    Color encodes STATE, not decoration - the left "rail" on a turn is
    the only saturated color on the page, and it always means the same
    thing: teal while investigating/complete (machine-verified), amber
    while paused for a human answer, brick if stopped. Two typefaces
    split BY ROLE: Space Grotesk for the system talking (labels, steps,
    status), Source Serif 4 for the analysis talking (questions,
    answers) - a content-driven distinction, not decoration. Quasar
    renders button labels in forced uppercase by default; that is
    overridden globally below, since shouting button labels are not a
    deliberate choice this app is making.

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


# ---------------------------------------------------------------
# Design tokens (see module docstring: color encodes state, type
# splits by role). Injected once per connection via ui.add_css - plain
# CSS, not Tailwind's JIT engine, so this never depends on NiceGUI's
# exact Tailwind configuration.
# ---------------------------------------------------------------

DESIGN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');

body {
    background-color: #F1F2EC;
}
body, .q-field__native, .q-btn {
    font-family: 'Space Grotesk', sans-serif;
}
/* Quasar buttons force uppercase + tracked letter-spacing by default -
   that is a framework default, not a choice this app is making. */
.q-btn {
    text-transform: none !important;
    letter-spacing: normal !important;
    font-weight: 500;
}

.app-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem;
             font-weight: 600; color: #1E2A32; }
.app-subtitle { font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem;
                color: #5B6570; }

.eyebrow { font-family: 'Space Grotesk', sans-serif; font-size: 0.78rem;
           font-weight: 500; color: #6B7280; }
.question-text { font-family: 'Source Serif 4', Georgia, serif;
                  font-size: 1.08rem; color: #1E2A32; }
.interpreted-note { font-family: 'Space Grotesk', sans-serif; font-size: 0.78rem;
                     color: #6B7280; font-style: italic; }
.status-line { font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem;
                color: #5B6570; }
.answer-text { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.05rem;
                line-height: 1.7; color: #1E2A32; max-width: 72ch; }

.turn-rail { border-left: 4px solid #D8DAD2; padding-left: 1.15rem;
             transition: border-color 0.3s ease; }
.rail-progress { border-left-color: #0F5C56; }
.rail-clarify   { border-left-color: #B8791E; }
.rail-stopped   { border-left-color: #9C3B32; }

.step-trail { border-left: 2px solid #E1E3DC; margin-left: 3px;
              padding-left: 0.9rem; }
.step-item  { font-family: 'Space Grotesk', sans-serif; font-size: 0.82rem;
              color: #4B5563; }

.suggestion-chip { font-family: 'Space Grotesk', sans-serif; font-size: 0.78rem; }
"""

# Icon + color per event type - the icon itself carries meaning (what
# kind of thing happened), not just a uniform bullet.
_STEP_ICON = {
    "investigation_started": ("search", "#5B6570"),
    "investigate_start": ("query_stats", "#0F5C56"),
    "duplicate_skipped": ("redo", "#9AA3AE"),
    "duplicate_clarify_skipped": ("redo", "#9AA3AE"),
    "evidence": ("check_circle", "#0F5C56"),
    "clarify": ("help", "#B8791E"),
    "synthesizing": ("edit_note", "#5B6570"),
    "complete": ("flag", "#0F5C56"),
    "stopped": ("block", "#9C3B32"),
    "forced_termination": ("warning", "#B8791E"),
}
_DEFAULT_ICON = ("fiber_manual_record", "#9AA3AE")

EXAMPLE_QUESTIONS = [
    "Which product categories generated the most sales?",
    "Why did sales drop at the end of 2018?",
    "Which sellers performed worst?",
    "Compare sales last quarter.",
]


def _format_event(event: dict) -> "str | None":
    """
    Turn a controller event into a step-trail line of TEXT (the icon is
    rendered separately by _STEP_ICON - see drain() below). Returns
    None for events not worth showing.

    The guiding rule: every visible line carries an actual fact (the
    real subquestion, the real finding, the real question) - never a
    bare generic status word.
    """
    t = event["type"]

    if t == "planning":
        return None  # internal step; investigate_start/clarify already say what happened next

    if t == "investigation_started":
        return "Starting investigation"
    if t == "investigate_start":
        return f"Querying: {event['subquestion']}"
    if t == "duplicate_skipped":
        return f"Already answered, skipping: {event['subquestion']}"
    if t == "duplicate_clarify_skipped":
        return f"Already know '{event['question']}' \u2192 {event.get('reused_answer', '')}, not asking again"
    if t == "evidence":
        # Capped so a worker answer listing many raw IDs never floods
        # the trail - the FULL claim still lives untouched in
        # state.evidence for the Evidence panel; only this display is
        # shortened.
        claim = event.get("claim", "").strip()
        if not claim:
            return "Evidence recorded"
        if len(claim) > 220:
            claim = claim[:220].rsplit(" ", 1)[0] + "\u2026 (see Evidence below)"
        return claim
    if t == "clarify":
        return event["question"]
    if t == "synthesizing":
        return "Writing final answer from recorded evidence"
    if t == "complete":
        return "Done"
    if t == "stopped":
        return f"Stopped: {event.get('reason', '')}"
    if t == "forced_termination":
        return event.get("reason", "")
    return t


def _scroll_to_bottom() -> None:
    ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")


@ui.page("/")
def main_page():
    # --- everything below is local to THIS browser connection ---

    ui.add_css(DESIGN_CSS)

    worker = SqlAgWorker()
    busy = {"value": False}  # one investigation in flight at a time per user
    turn_counter = {"n": 0}  # numbering turns is justified - they ARE a sequence

    # Local to this connection - each browser tab/user gets its OWN
    # history, so follow-up resolution ("that year", "those sellers")
    # never mixes context between different users. This is what makes
    # the chat's MEMORY persistent, not just its on-screen layout.
    conversation_history: list = []

    # Same idea, for clarify answers specifically. The controller
    # already refuses to re-ask a question in state.resolved_ambiguities
    # - but that guard only sees ONE turn's state. Seeding every new
    # turn with everything resolved so far makes the guard effective
    # ACROSS turns, not just within one.
    sticky_resolved_ambiguities: dict = {}

    ui.label("AI Data Analyst").classes("app-title")
    ui.label(
        "Ask a question about the Olist dataset. If something is "
        "ambiguous, I'll check with you before digging in."
    ).classes("app-subtitle")

    with ui.row().classes("w-full items-center gap-2 mt-2"):
        question_input = ui.input(
            placeholder="e.g. Why did sales drop at the end of 2018?"
        ).classes("flex-grow")
        run_button = ui.button("Investigate")
        spinner = ui.spinner(size="md", color="#0F5C56")
        spinner.visible = False

    suggestions_row = ui.row().classes("gap-2 flex-wrap mt-2")

    ui.separator().classes("mt-3")

    # Every turn is appended here, permanently. Nothing in this column
    # is ever cleared - that's the chat persistence.
    conversation_column = ui.column().classes("w-full gap-5 mt-3")

    def new_turn(question: str) -> dict:
        """
        Build the persistent UI for one turn and return the pieces
        later code needs to update. Everything here is local to this
        call - two turns never share elements.
        """
        turn_counter["n"] += 1
        n = turn_counter["n"]

        with conversation_column:
            with ui.column().classes("w-full turn-rail rail-progress gap-1") as rail:
                ui.label(f"Q{n}").classes("eyebrow")
                ui.label(question).classes("question-text")
                interpreted_label = ui.label("").classes("interpreted-note")
                interpreted_label.visible = False

                turn_status = ui.label("Investigating\u2026").classes(
                    "status-line mt-1"
                )
                step_trail = ui.column().classes("step-trail mt-1 gap-1")

                turn_answer = ui.markdown("").classes("answer-text mt-2")
                turn_evidence = ui.expansion("Evidence").classes("mt-1")
                turn_evidence.visible = False

        return {
            "rail": rail,
            "status": turn_status,
            "step_trail": step_trail,
            "answer": turn_answer,
            "interpreted": interpreted_label,
            "evidence": turn_evidence,
        }

    def set_rail(turn: dict, state_class: str) -> None:
        turn["rail"].classes(
            remove="rail-progress rail-clarify rail-stopped",
            add=state_class,
        )

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
        # CURRENT question/options/rationale (no stale content).
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label(question).classes("question-text").style("font-size:1rem")
            if rationale:
                ui.label(rationale).classes("interpreted-note")
            options_row = ui.row().classes("gap-2 mt-2")
            with options_row:
                for opt in options:
                    ui.button(
                        opt, on_click=lambda o=opt: dialog.submit(o)
                    ).props("outline").classes("suggestion-chip")
            free_text = ui.input(placeholder="or type your own answer").classes("mt-2")
            ui.button(
                "Submit", on_click=lambda: dialog.submit(free_text.value)
            ).classes("mt-2")

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
            # UI-THREAD (ui.timer) - safe to touch the step trail here.
            while True:
                try:
                    event = step_queue.get_nowait()
                except queue.Empty:
                    return
                line = _format_event(event)
                if line is None:
                    continue
                icon, color = _STEP_ICON.get(event["type"], _DEFAULT_ICON)
                with turn["step_trail"]:
                    with ui.row().classes("items-start gap-2 no-wrap"):
                        ui.icon(icon, size="14px").style(f"color:{color}; margin-top:2px;")
                        ui.label(line).classes("step-item")

        timer = ui.timer(0.25, drain)

        # Resolve follow-up references ("that year", "those sellers")
        # against this connection's OWN conversation history, before
        # the planner ever sees the question. Makes a blocking LLM
        # call, so it goes through run.io_bound like everything else
        # that talks to a model.
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
            set_rail(turn, "rail-clarify")
            turn["status"].set_text("Waiting on you\u2026")
            answer = await show_clarify(
                result.clarification_question,
                result.clarification_options,
                result.clarification_rationale,
            )
            set_rail(turn, "rail-progress")
            turn["status"].set_text("Investigating\u2026")
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
            n = len(result.state.evidence)
            noun = "query" if n == 1 else "queries"
            turn["status"].set_text(f"Complete, based on {n} {noun}." if n else "Complete.")
            turn["answer"].set_content(result.answer)
            render_evidence(turn["evidence"], result.state)
        else:
            set_rail(turn, "rail-stopped")
            turn["status"].set_text(f"Stopped because {result.reason}.")
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
        # NOTE: deliberately zero parameters. This function is bound
        # directly (no wrapping lambda) to both run_button.on_click and
        # question_input's keydown.enter - NiceGUI's event binding may
        # pass the triggering event object into a handler's first
        # parameter, so giving this function a `question` parameter
        # would risk that event object landing there instead of the
        # intended string. Suggestion chips go through use_suggestion()
        # below instead, which sets the input then calls this.
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

    async def use_suggestion(text: str) -> None:
        # Deliberately bound via a lambda that RETURNS this coroutine
        # (see the binding below), not via asyncio.create_task(). A
        # spawned Task loses NiceGUI's slot-tracking context entirely
        # - ui.timer/ui.* calls inside it raise "the current slot
        # cannot be determined... this may happen if you try to create
        # UI from a background task." NiceGUI's own on_click dispatcher
        # correctly awaits a returned coroutine IN-CONTEXT, which is
        # why this pattern (and the direct async on_click bindings
        # elsewhere in this file) works and create_task() does not.
        question_input.value = text
        await start_investigation()

    with suggestions_row:
        ui.label("Try:").classes("status-line self-center")
        for suggestion in EXAMPLE_QUESTIONS:
            ui.button(
                suggestion,
                on_click=lambda s=suggestion: use_suggestion(s),
            ).props("outline no-caps").classes("suggestion-chip")

    run_button.on_click(start_investigation)
    question_input.on("keydown.enter", start_investigation)


ui.run(title="AI Data Analyst")