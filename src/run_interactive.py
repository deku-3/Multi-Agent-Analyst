"""
Human-in-the-loop driver: the outer loop that actually talks to a
person.

The controller (src/controller.py) already suspends on `clarify` and
resumes on `resume_with_clarification` - that machinery is tested in
src/test_controller.py. What was missing is the piece that catches a
`needs_clarification` result, shows it to a real human, collects an
answer, and calls resume. That's what this file is.

    python -m src.run_interactive
"""

from src.planner_models import InvestigationState
from src.budget import InvestigationBudget
from src.worker_interface import SqlAgWorker
from src.controller import run_investigation, resume_with_clarification


def investigate(question: str, max_queries: int = 6, max_clarifications: int = 2):

    state = InvestigationState(question=question)
    budget = InvestigationBudget(
        max_queries=max_queries,
        max_clarifications=max_clarifications,
    )
    worker = SqlAgWorker()

    r = run_investigation(state, worker, budget)

    # Close the loop: keep answering clarifications until the
    # investigation reaches a terminal state (complete/stopped/failed).
    while r.status == "needs_clarification":

        print("\n" + "-" * 60)
        print("CLARIFICATION NEEDED:")
        print(r.clarification_question)

        if r.clarification_options:
            for i, opt in enumerate(r.clarification_options, 1):
                print(f"  {i}. {opt}")
            print("(type a number to pick one, or type your own answer)")

        answer = input("> ").strip()

        if r.clarification_options and answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(r.clarification_options):
                answer = r.clarification_options[idx]

        if not answer:
            print("(no answer given - stopping)")
            return r

        r = resume_with_clarification(
            r.state, answer, worker, budget,
            dimension=r.clarification_question,
        )

    print("\n" + "=" * 60)
    print(f"status: {r.status}   steps: {r.steps}   "
          f"queries_used: {budget.queries_used}")

    if r.status == "complete":
        print("\n--- EVIDENCE ---")
        for i, e in enumerate(r.state.evidence, 1):
            print(f"  [{i}] {e['subquestion']}")
        print("\n--- ANSWER ---")
        print(r.answer)
    else:
        print(f"reason: {r.reason}")

    return r


def main():
    print("AI Data Analyst - ask an analytical question")
    print("(try something ambiguous, e.g. 'Which sellers performed worst?'")
    print(" or 'How did sales change in Q3?', to see clarification fire)\n")

    question = input("> ").strip()
    if question:
        investigate(question)


if __name__ == "__main__":
    main()