from src.planner_models import InvestigationState
from src.planner import plan_next


QUESTIONS = [
    # Clear
    "Which product categories generated the most sales?",
    "Which states have the highest number of orders?",

    # Ambiguous but probably answerable with a sensible default
    "How has the average order value changed over time?",
    "Are customers making repeat purchases?",
    "Did payment method usage change over time?",

    # Should require clarification
    "How did sales change in Q3?",
    "Compare sales last quarter.",
    "Which sellers performed worst?",

    # Complex analytical questions
    "Why did sales decline?",
    "Which product categories have low review scores but high sales?",
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