from typing import Any, Literal
from pydantic import BaseModel, Field


class InvestigationState(BaseModel):
    question: str
    task_type: str = ""
    objective: str = ""
    context: dict[str, Any] = Field(default_factory=dict)

    observations: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)

    completed_steps: list[str] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)

    status: Literal[
        "planning",
        "investigating",
        "synthesizing",
        "complete",
    ] = "planning"


class PlannerAction(BaseModel):
    action: Literal[
        "investigate",
        "clarify",
        "synthesize",
        "stop",
    ]

    objective: str = ""
    subquestion: str = ""
    rationale: str = ""
    clarification_question: str = ""
    assumptions: list[str] = Field(default_factory=list)