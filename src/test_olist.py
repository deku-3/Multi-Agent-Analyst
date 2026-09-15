# src/test_fixes.py — python -m src.test_fixes
from src.planner_models import InvestigationState
from src.budget import InvestigationBudget
from src.worker_interface import SqlAgWorker
from src.controller import run_investigation, resume_with_clarification

CASES = [
    ("Compare sales last quarter.",
     "was silently merging Q2+Q3 into 1 row -> should now GROUP BY and show 2 rows"),

    ("Which product categories have low review scores but high sales?",
     "CTE/quartile query was blocked 3x as 'unsafe' -> should now succeed in 1 attempt"),

    ("Why did sales drop at the end of 2018?",
     "guard should verify with order counts (6512->16), call it a data artifact"),

    ("Which sellers performed worst?",
     "HITL: should CLARIFY -> auto-answers '3' to test resume"),

    ("Which product categories generated the most sales?",
     "clean baseline, should be unaffected by any of the fixes"),
]

def run(question, note):
    print("=" * 78)
    print(f"Q: {question}\n   ({note})")
    print("-" * 78)

    state = InvestigationState(question=question)
    budget = InvestigationBudget(max_queries=6, max_clarifications=1)
    worker = SqlAgWorker()

    r = run_investigation(state, worker, budget)

    if r.status == "needs_clarification":
        print(f"CLARIFY: {r.clarification_question}")
        print(f"OPTIONS: {r.clarification_options}")
        # auto-answer so the script runs unattended
        answer = r.clarification_options[0] if r.clarification_options else "3"
        print(f"-> auto-answering: {answer}")
        r = resume_with_clarification(r.state, answer, worker, budget)

    print(f"\nstatus: {r.status}  steps: {r.steps}  queries_used: {budget.queries_used}")
    for i, e in enumerate(r.state.evidence, 1):
        print(f"  [{i}] {e['subquestion']}")
        print(f"      SQL: {' '.join(e.get('sql','').split())[:150]}")
    print(f"\nANSWER:\n{r.answer or r.reason}\n")

if __name__ == "__main__":
    for q, note in CASES:
        run(q, note)