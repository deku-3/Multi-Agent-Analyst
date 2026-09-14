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


def _terminate(state, synthesizer, reason, steps) -> ControllerResult:
    """Forced exit (budget/dup/step cap): synthesize what we have, else stop."""
    if state.evidence:
        state.status = "synthesizing"
        answer = synthesizer(state)
        state.status = "complete"
        return ControllerResult(
            status="complete", answer=answer, reason=reason,
            state=state, steps=steps,
        )
    state.status = "complete"
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

    while True:
        steps += 1
        if steps > max_steps:
            return _terminate(state, synthesizer, "step_cap", steps)

        action = _plan_with_retry(
            state, budget.queries_remaining, planner, plan_retries,
        )
        act = action.action

        # ---------------- INVESTIGATE ----------------
        if act == "investigate":
            if budget.queries_remaining <= 0:
                return _terminate(state, synthesizer, "budget_exhausted", steps)

            key = _normalize(action.subquestion)
            if key in seen:
                dup_strikes += 1
                if dup_strikes >= 2:
                    # planner is stuck repeating itself; cut it off.
                    return _terminate(state, synthesizer, "duplicate_loop", steps)
                # nudge (no budget spent) and re-plan
                state.completed_steps.append(
                    f"[already answered] {action.subquestion}"
                )
                continue

            seen.add(key)
            result = worker.run(action.subquestion)
            budget.queries_used += 1

            state.evidence.append(
                build_evidence(
                    step=budget.queries_used,
                    objective=action.objective,
                    subquestion=action.subquestion,
                    assumptions=list(action.assumptions),
                    result=result,
                )
            )
            state.completed_steps.append(action.subquestion)
            state.status = "investigating"
            continue

        # ---------------- CLARIFY ----------------
        if act == "clarify":
            if budget.clarifications_remaining <= 0:
                # not allowed to interrupt again: fall back to evidence.
                return _terminate(state, synthesizer, "clarify_budget", steps)
            budget.clarifications_used += 1
            state.pending_questions.append(action.clarification_question)
            state.status = "planning"
            return ControllerResult(
                status="needs_clarification",
                clarification_question=action.clarification_question,
                clarification_options=list(
                    getattr(action, "clarification_options", []) or []
                ),
                state=state,
                steps=steps,
            )

        # ---------------- SYNTHESIZE ----------------
        if act == "synthesize":
            state.status = "synthesizing"
            answer = synthesizer(state)
            state.status = "complete"
            return ControllerResult(
                status="complete", answer=answer, state=state, steps=steps,
            )

        # ---------------- STOP ----------------
        if act == "stop":
            state.status = "complete"
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