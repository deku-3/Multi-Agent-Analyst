"""
Typed evidence.

Section 16 of the design: final conclusions should be generated from
recorded evidence, not from the LLM's memory of prior results. Evidence
is a stdlib dataclass (no pydantic dependency) so the controller and its
tests run without the model stack. It is stored on
InvestigationState.evidence as a plain dict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Evidence:
    step: int                       # which investigation produced this (1-based)
    objective: str                  # what we were trying to learn
    subquestion: str                # the concrete question sent to the worker
    claim: str                      # the worker's natural-language answer
    sql: str = ""                   # the SQL that produced it (auditability)
    raw_result: str = ""            # the raw rows (auditability)
    assumptions: list[str] = field(default_factory=list)
    evidence_type: str = "observed"  # observed | contributor | supported | causal


def build_evidence(
    *,
    step: int,
    objective: str,
    subquestion: str,
    assumptions: list[str],
    result: Any,
) -> dict:
    """
    Build an evidence dict from a WorkerResult (or anything exposing
    .answer / .sql / .raw_result). Returned as a dict to fit
    InvestigationState.evidence: list[dict].
    """
    ev = Evidence(
        step=step,
        objective=objective,
        subquestion=subquestion,
        claim=getattr(result, "answer", str(result)),
        sql=getattr(result, "sql", ""),
        raw_result=getattr(result, "raw_result", ""),
        assumptions=list(assumptions or []),
    )
    return asdict(ev)