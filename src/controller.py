"""
Controller: the deterministic execution layer around the planner LLM.

    plan -> validate/guard -> execute (SQL Worker) -> record evidence
         -> update state -> plan ...

The planner decides WHAT analytical question to investigate next. The
controller decides whether that decision is allowed to run, executes it,
records the evidence, and updates the state. Everything the design doc
called "deterministic enforcement" lives here, not in the prompt:

  - query budget            (refuse investigate when exhausted)
  - duplicate detection     (never run the same investigation twice)
  - clarify budget          (bound how often we interrupt the human)
  - step cap                (hard stop against infinite planning loops)
  - plan retry              (reject malformed PlannerActions, re-plan)

Human-in-the-loop uses a return-and-resume model: on `clarify` the loop
suspends and returns the question; the caller collects an answer and
calls resume_with_clarification(), which folds the answer into state as
USER-SPECIFIED and continues. State carries everything, so this works
across stateless workers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from src.budget import InvestigationBudget
from src.evidence import build_evidence
from src.worker_interface import SQLWorker

if TYPE_CHECKING:  # avoid importing the pydantic model stack at runtime
    from src.planner_models import InvestigationState


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------

@dataclass
class ControllerResult:
    # complete | needs_clarification | stopped | failed
    status: str
    answer: str = ""
    reason: str = ""
    clarification_question: str = ""
    clarification_options: list = field(default_factory=list)
    clarification_rationale: str = ""
    state: Any = None
    steps: int = 0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _normalize(subquestion: str) -> str:
    """Cheap syntactic normalization for duplicate detection.

    NOTE: this catches wording-identical repeats, not semantic paraphrase.
    Semantic dedup would need embeddings; deliberately out of scope.
    """
    s = subquestion.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip("?.! ")


def _plan_with_retry(state, queries_remaining, planner, retries):
    """Re-plan on a rejected (malformed) PlannerAction before giving up."""
    last_exc = None
    for _ in range(retries + 1):
        try:
            return planner(state, queries_remaining)
        except Exception as exc:  # e.g. pydantic ValidationError from the validator
            last_exc = exc
    raise last_exc


def _emit(on_event: Optional[Callable[[dict], None]], event_type: str, **data) -> None:
    """
    Safely fire a progress event.

    on_event is called synchronously, in whatever thread run_investigation
    itself is running in (e.g. a worker thread if the caller offloaded it
    there, such as NiceGUI's run.io_bound). It must therefore be CHEAP and
    THREAD-SAFE - e.g. a queue.Queue.put(), never a direct UI mutation.
    A broken/slow on_event must never take down the investigation, so any
    exception it raises is swallowed here.
    """
    if on_event is None:
        return
    try:
        on_event({"type": event_type, **data})
    except Exception:
        pass


def _terminate(state, synthesizer, reason, steps, on_event=None) -> ControllerResult:
    """Forced exit (budget/dup/step cap): synthesize what we have, else stop."""
    _emit(on_event, "forced_termination", reason=reason)
    if state.evidence:
        state.status = "synthesizing"
        _emit(on_event, "synthesizing")
        answer = synthesizer(state)
        state.status = "complete"
        _emit(on_event, "complete", answer=answer, reason=reason)
        return ControllerResult(
            status="complete", answer=answer, reason=reason,
            state=state, steps=steps,
        )
    state.status = "complete"
    _emit(on_event, "stopped", reason=reason)
    return ControllerResult(status="stopped", reason=reason, state=state, steps=steps)


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def run_investigation(
    state: "InvestigationState",
    worker: SQLWorker,
    budget: Optional[InvestigationBudget] = None,
    *,
    planner: Optional[Callable] = None,
    synthesizer: Optional[Callable] = None,
    max_steps: Optional[int] = None,
    plan_retries: int = 2,
    on_event: Optional[Callable[[dict], None]] = None,
) -> ControllerResult:

    if budget is None:
        budget = InvestigationBudget()
    if planner is None:
        from src.planner import plan_next          # lazy
        planner = plan_next
    if synthesizer is None:
        from src.synthesizer import synthesize      # lazy
        synthesizer = synthesize
    if max_steps is None:
        max_steps = budget.max_queries + budget.max_clarifications + 3

    # Seed dedup from completed work so it survives suspend/resume.
    seen = {
        _normalize(s)
        for s in state.completed_steps
        if not s.startswith("[already")
    }
    dup_strikes = 0
    steps = 0
    _emit(on_event, "investigation_started", question=state.question)

    while True:
        steps += 1
        if steps > max_steps:
            return _terminate(state, synthesizer, "step_cap", steps, on_event)

        _emit(on_event, "planning", step=steps)
        action = _plan_with_retry(
            state, budget.queries_remaining, planner, plan_retries,
        )
        act = action.action

        # ---------------- INVESTIGATE ----------------
        if act == "investigate":
            if budget.queries_remaining <= 0:
                return _terminate(state, synthesizer, "budget_exhausted", steps, on_event)

            key = _normalize(action.subquestion)
            if key in seen:
                dup_strikes += 1
                _emit(on_event, "duplicate_skipped", subquestion=action.subquestion)
                if dup_strikes >= 2:
                    # planner is stuck repeating itself; cut it off.
                    return _terminate(state, synthesizer, "duplicate_loop", steps, on_event)
                # nudge (no budget spent) and re-plan
                state.completed_steps.append(
                    f"[already answered] {action.subquestion}"
                )
                continue

            seen.add(key)
            _emit(
                on_event, "investigate_start",
                subquestion=action.subquestion, objective=action.objective,
            )
            result = worker.run(action.subquestion)
            budget.queries_used += 1

            evidence = build_evidence(
                step=budget.queries_used,
                objective=action.objective,
                subquestion=action.subquestion,
                assumptions=list(action.assumptions),
                result=result,
            )
            state.evidence.append(evidence)
            state.completed_steps.append(action.subquestion)
            state.status = "investigating"
            _emit(
                on_event, "evidence",
                subquestion=action.subquestion, claim=evidence.get("claim", ""),
                queries_used=budget.queries_used, queries_remaining=budget.queries_remaining,
            )
            continue

        # ---------------- CLARIFY ----------------
        if act == "clarify":
            if budget.clarifications_remaining <= 0:
                # not allowed to interrupt again: fall back to evidence.
                return _terminate(state, synthesizer, "clarify_budget", steps, on_event)
            budget.clarifications_used += 1
            state.pending_questions.append(action.clarification_question)
            state.status = "planning"
            options = list(getattr(action, "clarification_options", []) or [])
            _emit(
                on_event, "clarify",
                question=action.clarification_question, options=options,
                rationale=action.rationale,
            )
            return ControllerResult(
                status="needs_clarification",
                clarification_question=action.clarification_question,
                clarification_options=options,
                clarification_rationale=action.rationale,
                state=state,
                steps=steps,
            )

        # ---------------- SYNTHESIZE ----------------
        if act == "synthesize":
            state.status = "synthesizing"
            _emit(on_event, "synthesizing")
            answer = synthesizer(state)
            state.status = "complete"
            _emit(on_event, "complete", answer=answer)
            return ControllerResult(
                status="complete", answer=answer, state=state, steps=steps,
            )

        # ---------------- STOP ----------------
        if act == "stop":
            state.status = "complete"
            _emit(on_event, "stopped", reason=action.rationale)
            return ControllerResult(
                status="stopped", reason=action.rationale,
                state=state, steps=steps,
            )

        return ControllerResult(
            status="failed", reason=f"unknown action: {act}",
            state=state, steps=steps,
        )


# ---------------------------------------------------------------------
# Human-in-the-loop resume
# ---------------------------------------------------------------------

def resume_with_clarification(
    state: "InvestigationState",
    answer: str,
    worker: SQLWorker,
    budget: Optional[InvestigationBudget] = None,
    *,
    dimension: Optional[str] = None,
    **kwargs,
) -> ControllerResult:
    """
    Fold the human's answer into state as USER-SPECIFIED, then continue.
    """
    key = dimension or (
        state.pending_questions[-1] if state.pending_questions else "clarification"
    )
    ra = getattr(state, "resolved_ambiguities", None)
    if ra is None:
        try:
            state.resolved_ambiguities = {key: answer}
        except Exception:
            state.context[key] = answer
    else:
        ra[key] = answer

    state.pending_questions = []
    return run_investigation(state, worker, budget, **kwargs)