from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.planner_models import (
    InvestigationState,
    PlannerAction,
)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


PLANNER_PROMPT = """
You are the investigation planner for an AI data analyst.

Your job is to decide the SINGLE most useful next action needed
to answer the user's question.

You do NOT write SQL.
An SQL Worker handles concrete database questions.

CURRENT DATE/TIME:
{current_datetime}

DATASET CONTEXT:
{dataset_context}

USER QUESTION:
{question}

CURRENT INVESTIGATION STATE:
Task type: {task_type}
Objective: {objective}
Context: {context}

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

RULES:

1. Use existing evidence before requesting new information.

2. Do not invent facts, metrics, dates, thresholds, or results.

3. Use system-defined metric definitions and dataset rules when available.

4. Ask for clarification ONLY when missing information materially
   changes the answer and cannot be resolved deterministically.

5. Do not ask the user how to investigate a root-cause question.
   You decide the investigation strategy.

6. For root-cause questions:
   establish the change → identify drivers → localize drivers
   → investigate relevant explanatory signals → synthesize.

7. Do not assume a particular cause or dimension before evidence
   supports it.

8. Do not repeat completed investigations.

9. If one direct investigation can answer the question, prefer that
   instead of creating unnecessary steps.

10. If enough evidence exists, choose "synthesize".

11. If the question cannot be answered reliably with the available
    data, choose "stop".

12. If clarification is required, ask exactly ONE concise question.

Return exactly one action.
"""


def build_dataset_context() -> dict:
    return {
        "name": "Olist Brazilian E-Commerce",
        "date_min": "2016-09-04",
        "date_max": "2018-10-17",
        "partial_periods": [
            "2016-09",
            "2018-10",
        ],
        "defaults": {
            "sales": "delivered GMV = SUM(order_items.price)",
            "orders": "COUNT(DISTINCT orders.order_id)",
            "unique_customers": (
                "COUNT(DISTINCT customers.customer_unique_id)"
            ),
            "aov": "delivered GMV / delivered orders",
            "freight": "SUM(order_items.freight_value)",
            "sales_date": "orders.order_purchase_timestamp",
        },
    }


def plan_next(
    state: InvestigationState,
    queries_remaining: int,
) -> PlannerAction:

    current_datetime = (
        datetime.now().astimezone().isoformat()
    )

    dataset_context = build_dataset_context()

    prompt = PLANNER_PROMPT.format(
        current_datetime=current_datetime,
        dataset_context=dataset_context,
        question=state.question,
        task_type=state.task_type,
        objective=state.objective,
        context=state.context,
        observations=state.observations,
        hypotheses=state.hypotheses,
        evidence=state.evidence,
        completed_steps=state.completed_steps,
        pending_questions=state.pending_questions,
    )

    prompt += (
        f"\n\nQUERIES REMAINING: {queries_remaining}"
    )

    return (
        llm
        .with_structured_output(PlannerAction)
        .invoke(prompt)
    )