from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class InvestigationState(BaseModel):
    question: str

    task_type: str = ""
    objective: str = ""

    context: dict[str, Any] = Field(
        default_factory=dict
    )

    observations: list[dict[str, Any]] = Field(
        default_factory=list
    )

    hypotheses: list[dict[str, Any]] = Field(
        default_factory=list
    )

    evidence: list[dict[str, Any]] = Field(
        default_factory=list
    )

    completed_steps: list[str] = Field(
        default_factory=list
    )

    pending_questions: list[str] = Field(
        default_factory=list
    )

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

    complexity: Literal[
        "direct",
        "analytical",
        "investigative",
    ]

    objective: str = ""

    subquestion: str = ""

    rationale: str = ""

    clarification_question: str = ""

    assumptions: list[str] = Field(
        default_factory=list
    )

    # -----------------------------------------------------------------
    # Fix 3: enforce the action/field contract deterministically.
    #
    # The LLM will happily emit a `clarification_question` on an
    # `investigate` action (and vice versa). Rather than hope the prompt
    # holds, we validate the shape here. This is the first
    # "LLM proposes, application validates" boundary in the system.
    # -----------------------------------------------------------------
    @model_validator(mode="after")
    def _enforce_action_contract(self) -> "PlannerAction":

        if self.action == "investigate":
            if not self.subquestion.strip():
                raise ValueError(
                    "investigate action requires a non-empty subquestion"
                )
            self.clarification_question = ""

        elif self.action == "clarify":
            if not self.clarification_question.strip():
                raise ValueError(
                    "clarify action requires a non-empty "
                    "clarification_question"
                )
            self.subquestion = ""

        else:  # synthesize | stop
            self.subquestion = ""
            self.clarification_question = ""

        return self