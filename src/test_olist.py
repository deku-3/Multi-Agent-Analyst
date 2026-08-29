from src.planner_models import InvestigationState
from src.planner import plan_next


QUESTIONS = [
    "Which product categories generated the most sales?",
    "Which states have the highest number of orders?",
    "How has the average order value changed over time?",
    "Which sellers have the worst delivery performance?",
    "Did payment method usage change over time?",
    "Are customers making repeat purchases?",
    "Which product categories have low review scores but high sales?",
    "Why did sales decline?",
]


for question in QUESTIONS:

    state = InvestigationState(
        question=question,
        task_type="",
        objective="",
        context={},
    )

    action = plan_next(
        state,
        queries_remaining=10,
    )

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)
    print(action.model_dump_json(indent=2))