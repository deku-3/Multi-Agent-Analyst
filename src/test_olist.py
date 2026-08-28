from src.planner_models import InvestigationState
from src.planner import plan_next


state = InvestigationState(
    question="Why did sales drop last quarter?",
    task_type="root_cause_analysis",
    objective="Explain the change in sales.",
    context={},
)


action = plan_next(
    state,
    queries_remaining=10,
)


print("\n=== PLANNER ACTION ===")
print(action.model_dump_json(indent=2))