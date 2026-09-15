"""
Offline controller tests.

Run the WHOLE investigation loop with no DB / chromadb / API key, using a
scripted planner, a fake worker, and a fake synthesizer. This verifies the
DETERMINISTIC control logic (the part that must never depend on the LLM):
budget enforcement, duplicate detection, clarify suspend/resume, synthesize,
stop, and the step cap.

    python -m src.test_controller
"""

from dataclasses import dataclass, field

from src.budget import InvestigationBudget
from src.worker_interface import FakeWorker
from src.controller import run_investigation, resume_with_clarification


# --- lightweight stand-ins (avoid importing the pydantic model stack) ---

class FakeState:
    def __init__(self, question):
        self.question = question
        self.task_type = ""
        self.objective = ""
        self.context = {}
        self.observations = []
        self.hypotheses = []
        self.evidence = []
        self.completed_steps = []
        self.pending_questions = []
        self.resolved_ambiguities = {}
        self.status = "planning"


@dataclass
class FakeAction:
    action: str
    complexity: str = "analytical"
    objective: str = ""
    subquestion: str = ""
    rationale: str = ""
    clarification_question: str = ""
    clarification_options: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)


class ScriptedPlanner:
    """Yields a fixed sequence of actions; records queries_remaining seen."""
    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0
        self.budget_seen = []

    def __call__(self, state, queries_remaining):
        self.budget_seen.append(queries_remaining)
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a


def fake_synth(state):
    return f"FINAL[{len(state.evidence)} evidence]"


# --------------------------- tests ---------------------------

results = []

def check(name, cond):
    results.append((name, cond))
    print(("PASS " if cond else "FAIL ") + name)


# 1. Happy path: two investigations then synthesize.
planner = ScriptedPlanner([
    FakeAction("investigate", subquestion="monthly GMV trend", objective="trend"),
    FakeAction("investigate", subquestion="AOV vs orders in decline", objective="decompose"),
    FakeAction("synthesize", objective="answer"),
])
budget = InvestigationBudget(max_queries=10)
worker = FakeWorker()
r = run_investigation(FakeState("why did sales decline?"), worker, budget,
                      planner=planner, synthesizer=fake_synth)
check("happy: status complete", r.status == "complete")
check("happy: 2 evidence recorded", len(r.state.evidence) == 2)
check("happy: budget used == 2", budget.queries_used == 2)
check("happy: worker called twice", len(worker.calls) == 2)
check("happy: answer from synth", r.answer == "FINAL[2 evidence]")
check("happy: evidence has sql (audit)", r.state.evidence[0]["sql"] != "")


# 2. Duplicate detection: same subquestion proposed repeatedly.
planner = ScriptedPlanner([
    FakeAction("investigate", subquestion="monthly GMV trend", objective="t"),
    FakeAction("investigate", subquestion="Monthly  GMV trend?", objective="t"),  # dup (norm)
    FakeAction("investigate", subquestion="monthly gmv trend", objective="t"),    # dup again
])
budget = InvestigationBudget(max_queries=10)
worker = FakeWorker()
r = run_investigation(FakeState("q"), worker, budget,
                      planner=planner, synthesizer=fake_synth)
check("dup: worker called ONCE (dup blocked)", len(worker.calls) == 1)
check("dup: budget spent ONCE", budget.queries_used == 1)
check("dup: terminated on duplicate_loop", r.reason == "duplicate_loop")
check("dup: still synthesized what we had", r.status == "complete")


# 3. Budget enforcement: planner wants 3 investigations, budget allows 1.
planner = ScriptedPlanner([
    FakeAction("investigate", subquestion="q1", objective="o"),
    FakeAction("investigate", subquestion="q2", objective="o"),
    FakeAction("investigate", subquestion="q3", objective="o"),
])
budget = InvestigationBudget(max_queries=1)
worker = FakeWorker()
r = run_investigation(FakeState("q"), worker, budget,
                      planner=planner, synthesizer=fake_synth)
check("budget: worker called ONCE", len(worker.calls) == 1)
check("budget: exhausted then terminated", r.reason == "budget_exhausted")
check("budget: planner saw 0 remaining on 2nd call", planner.budget_seen[1] == 0)


# 4. Clarify suspend + resume.
planner = ScriptedPlanner([
    FakeAction("clarify", clarification_question="Which year?",
               clarification_options=["2017", "2018"], objective="year"),
    FakeAction("investigate", subquestion="Q3 2018 GMV", objective="trend"),
    FakeAction("synthesize", objective="answer"),
])
budget = InvestigationBudget(max_queries=10, max_clarifications=1)
worker = FakeWorker()
state = FakeState("how did sales change in Q3?")
r1 = run_investigation(state, worker, budget,
                       planner=planner, synthesizer=fake_synth)
check("clarify: suspends", r1.status == "needs_clarification")
check("clarify: returns options", r1.clarification_options == ["2017", "2018"])
check("clarify: nothing investigated yet", len(worker.calls) == 0)

r2 = resume_with_clarification(r1.state, "2018", worker, budget,
                               planner=planner, synthesizer=fake_synth)
check("resume: answer stored as user-specified",
      r1.state.resolved_ambiguities.get("Which year?") == "2018")
check("resume: completes after resume", r2.status == "complete")
check("resume: one investigation ran", len(worker.calls) == 1)


# 5. Clarify budget: second clarify not allowed -> fall back.
planner = ScriptedPlanner([FakeAction("clarify", clarification_question="again?")])
budget = InvestigationBudget(max_queries=10, max_clarifications=0)
worker = FakeWorker()
r = run_investigation(FakeState("q"), worker, budget,
                      planner=planner, synthesizer=fake_synth)
check("clarify-budget: not asked (0 allowed)", r.status != "needs_clarification")
check("clarify-budget: reason clarify_budget", r.reason == "clarify_budget")


# 6. Stop.
planner = ScriptedPlanner([FakeAction("stop", rationale="data cannot answer this")])
budget = InvestigationBudget()
worker = FakeWorker()
r = run_investigation(FakeState("q"), worker, budget,
                      planner=planner, synthesizer=fake_synth)
check("stop: status stopped", r.status == "stopped")
check("stop: reason passed through", r.reason == "data cannot answer this")
check("stop: nothing investigated", len(worker.calls) == 0)


# --------------------------- summary ---------------------------
passed = sum(1 for _, c in results if c)
print("\n" + "=" * 50)
print(f"{passed}/{len(results)} checks passed")
if passed != len(results):
    raise SystemExit(1)


# --------------------------- on_event tests ---------------------------
# These prove the event hook that the frontend's live step feed depends
# on: events fire in the right order, carry the right data, and a
# broken/raising on_event NEVER crashes the investigation.

events_log = []

def collect(event):
    events_log.append(event)

planner = ScriptedPlanner([
    FakeAction("investigate", subquestion="monthly GMV trend", objective="trend"),
    FakeAction("synthesize", objective="answer"),
])
budget = InvestigationBudget(max_queries=10)
worker = FakeWorker()
r = run_investigation(FakeState("why did sales decline?"), worker, budget,
                      planner=planner, synthesizer=fake_synth, on_event=collect)

types = [e["type"] for e in events_log]
check("events: investigation_started first", types[0] == "investigation_started")
check("events: investigate_start before evidence", types.index("investigate_start") < types.index("evidence"))
check("events: evidence carries subquestion", events_log[types.index("evidence")]["subquestion"] == "monthly GMV trend")
check("events: synthesizing before complete", types.index("synthesizing") < types.index("complete"))
check("events: complete carries answer", events_log[-1]["answer"] == "FINAL[1 evidence]")

# A raising on_event must never crash the investigation.
def bad_event(event):
    raise RuntimeError("frontend blew up")

planner = ScriptedPlanner([
    FakeAction("investigate", subquestion="q1", objective="o"),
    FakeAction("synthesize", objective="answer"),
])
budget = InvestigationBudget(max_queries=10)
worker = FakeWorker()
r = run_investigation(FakeState("q"), worker, budget,
                      planner=planner, synthesizer=fake_synth, on_event=bad_event)
check("events: raising on_event does not crash investigation", r.status == "complete")

# Clarify event carries the options (what the frontend renders as buttons).
planner = ScriptedPlanner([
    FakeAction("clarify", clarification_question="Which year?",
               clarification_options=["2017", "2018"], objective="year"),
])
budget = InvestigationBudget(max_queries=10, max_clarifications=1)
worker = FakeWorker()
events_log2 = []
r = run_investigation(FakeState("q"), worker, budget,
                      planner=planner, synthesizer=fake_synth, on_event=events_log2.append)
clarify_events = [e for e in events_log2 if e["type"] == "clarify"]
check("events: clarify event fired", len(clarify_events) == 1)
check("events: clarify event carries options", clarify_events[0]["options"] == ["2017", "2018"])


passed = sum(1 for _, c in results if c)
print(f"\n{'='*50}\n{passed}/{len(results)} checks passed (incl. on_event tests)")
if passed != len(results):
    raise SystemExit(1)