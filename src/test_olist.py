from src.planner_models import InvestigationState
from src.budget import InvestigationBudget
from src.worker_interface import SqlAgWorker
from src.controller import run_investigation

r = run_investigation(
    InvestigationState(question="Why did sales drop at the end of 2018?"),
    SqlAgWorker(),
    InvestigationBudget(max_queries=6),
)
print(r.status, "|", r.steps, "steps")
print(r.answer)
for e in r.state.evidence:
    print("-", e["subquestion"], "→", e["sql"][:80])