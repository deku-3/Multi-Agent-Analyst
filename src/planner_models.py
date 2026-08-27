from typing import Any, Literal
from pydantic import BaseModel, Field


class InvestigationState(BaseModel):
    question: str

    # What the planner believes the user is asking
    task_type: str = ""
    objective: str = ""

    # Problem-specific interpretation
    context: dict[str, Any] = Field(default_factory=dict)

    # What we have learned so far
    observations: list[dict[str, Any]] = Field(default_factory=list)

    # Things we think may explain the answer
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)

    # Claims supported by actual query results
    evidence: list[dict[str, Any]] = Field(default_factory=list)

    # Prevent duplicate / pointless investigation
    completed_steps: list[str] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)

    # Lifecycle
    status: Literal[
        "planning",
        "investigating",
        "synthesizing",
        "complete",
    ] = "planning"


class PlannerAction(BaseModel):
    action: Literal[
        "investigate",
        "synthesize",
        "stop",
    ]

    objective: str = Field(
        description="What this next action is trying to establish."
    )

    subquestion: str = Field(
        description="One concrete analytical question for the SQL worker."
    )

    rationale: str = Field(
        description="Why this is the most useful next investigation."
    )