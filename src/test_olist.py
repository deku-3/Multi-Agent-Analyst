"""
Live end-to-end test through the full controller.
Runs real planner + real SQL worker + real synthesizer.

Requires: .env with OPENAI key, built olist.sqlite, populated Chroma store.

    python -m src.test_live
"""

from src.planner_models import InvestigationState
from src.budget import InvestigationBudget
from src.worker_interface import SqlAgWorker
from src.controller import run_investigation


CASES = [
    # The guard's proof case — the important one.
    ("Why did sales drop at the end of 2018?",
     "should VERIFY with order counts (6512->16), call it a data artifact"),

    # Root cause on a real (non-artifact) framing.
    ("Why did sales decline?",
     "should establish the trend before guessing any driver"),

    # Relative period must anchor to effective_max_date (Aug 2018).
    ("Compare sales last quarter.",
     "last quarter -> Q2/Q3 2018 (NOT 2026, NOT the empty Oct)"),

    # Quartile threshold discovery (Option B) + %q-free quarter logic.
    ("Which product categories have low review scores but high sales?",
     "should discover quartile cutoffs, not invent < 3"),

    # Repeat rate — should report ~3%, using customer_unique_id.
    ("Are customers making repeat purchases?",
     "should report the RATE (~3%), not a raw count"),
]


def divider(ch="="):
    print(ch * 78)


def run_case(question, expectation):
    divider()
    print(f"Q: {question}")
    print(f"   expect: {expectation}")
    divider("-")

    state = InvestigationState(question=question)
    budget = InvestigationBudget(max_queries=6, max_clarifications=1)

    try:
        r = run_investigation(state, SqlAgWorker(), budget)
    except Exception as exc:
        print(f"   !! FAILED: {type(exc).__name__}: {exc}")
        return

    print(f"status: {r.status}   steps: {r.steps}   "
          f"queries_used: {budget.queries_used}")

    if r.status == "needs_clarification":
        print(f"CLARIFY: {r.clarification_question}")
        print(f"OPTIONS: {r.clarification_options}")
        return
    if r.status == "stopped":
        print(f"STOPPED: {r.reason}")

    print("\n--- EVIDENCE (subquestion -> sql) ---")
    for i, e in enumerate(r.state.evidence, 1):
        sql = " ".join(e.get("sql", "").split())          # collapse whitespace
        print(f"  [{i}] {e['subquestion']}")
        print(f"      SQL: {sql[:140]}")
        if e.get("assumptions"):
            print(f"      assumes: {e['assumptions']}")

    print("\n--- FINAL ANSWER ---")
    print(r.answer)
    print()

    # cheap automated flags
    joined = " ".join(
        (e.get("sql", "") + " " + e.get("subquestion", ""))
        for e in r.state.evidence
    ).lower()
    if "drop at the end of 2018" in question.lower():
        checked_counts = "count(" in joined
        print(f"   [flag] order-count verification present: {checked_counts}")
    if "%q" in joined:
        print("   [flag] WARNING: %q still in generated SQL (quarter bug)")


def main():
    for q, exp in CASES:
        run_case(q, exp)


if __name__ == "__main__":
    main()