from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, Any

from src.planner_models import (
    InvestigationState,
    PlannerAction,
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


PLANNER_PROMPT = """
You are the investigation planner for an AI data analyst.

Your job is NOT to write SQL.

Your job is to decide the single most useful next analytical
step needed to answer the user's question.

The database is the Olist Brazilian e-commerce dataset.

You have an SQL Worker available. The SQL Worker can answer
one concrete analytical subquestion against the database.

CURRENT INVESTIGATION:

User question:
{question}

Task type:
{task_type}

Objective:
{objective}

Context:
{context}

Observations:
{observations}

Hypotheses:
{hypotheses}

Evidence:
{evidence}

Completed steps:
{completed_steps}

Pending questions:
{pending_questions}

Budget:
Queries remaining: {queries_remaining}

RULES:

1. Do not write SQL.
2. Produce exactly ONE next action.
3. The subquestion must be concrete enough for the SQL worker.
4. Use existing observations before requesting new information.
5. Do not repeat completed investigations.
6. Prefer the investigation with the highest expected information gain.
7. Do not claim causality without evidence.
8. If enough evidence exists to answer the user's question,
   choose "synthesize".
9. If no useful investigation remains, choose "stop".
"""


def plan_next(
    state: InvestigationState,
    queries_remaining: int,
) -> PlannerAction:

    prompt = PLANNER_PROMPT.format(
        question=state.question,
        task_type=state.task_type,
        objective=state.objective,
        context=state.context,
        observations=state.observations,
        hypotheses=state.hypotheses,
        evidence=state.evidence,
        completed_steps=state.completed_steps,
        pending_questions=state.pending_questions,
        queries_remaining=queries_remaining,
    )

    result = (
        llm
        .with_structured_output(PlannerAction)
        .invoke(prompt)
    )

    return result