"""
Synthesizer: turn recorded evidence into a grounded final answer.

Critically, it is fed state.evidence (recorded, auditable) - NOT the
LLM's memory of what happened. The prompt forbids introducing facts that
are not present in the evidence and requires distinguishing an observed
change from a measured contributor from a causal claim.
"""

from __future__ import annotations

import json


SYNTHESIS_PROMPT = """
You are the synthesis step of an AI data analyst.

Write the final answer to the user's question using ONLY the recorded
evidence below. Each evidence item is the answer the SQL Worker returned
for one concrete subquestion, with the SQL that produced it.

USER QUESTION:
{question}

RESOLVED AMBIGUITIES (treat as user-specified):
{resolved}

RECORDED EVIDENCE (the only facts you may use):
{evidence}

RULES:

1. Ground every quantitative claim in a specific evidence item. Do not
   introduce numbers, categories, dates, or causes that are not present
   in the evidence.

2. Be precise about epistemic status. Distinguish:
   - an OBSERVED change (a metric moved),
   - a MEASURED CONTRIBUTOR (a component accounts for part of the change),
   - a CAUSAL claim (X caused Y).
   Only state a cause if the evidence actually supports it. Correlation
   is not causation.

3. If the evidence shows an apparent decline concentrated in a partial /
   truncated data period, say clearly that it is likely a data-boundary
   artifact rather than a business decline.

4. Surface the assumptions recorded in the evidence (e.g. how a threshold
   or relative period was resolved) so the user can see the choices made.

5. If the evidence is insufficient to fully answer, say what is known and
   what remains unknown. Do not pad.

6. If an evidence item's claim contains a long enumerated list (dozens or
   hundreds of raw IDs), do NOT reproduce that list in the final answer.
   State the count and a handful of representative examples instead - the
   full list is already recorded in the evidence and available on demand.
   The final answer is what the user reads first; it must never be a wall
   of raw IDs.

Answer directly and concretely.
"""


def synthesize(state, llm=None) -> str:
    if llm is None:
        from src.planner import llm as default_llm  # lazy
        llm = default_llm

    prompt = SYNTHESIS_PROMPT.format(
        question=state.question,
        resolved=json.dumps(
            getattr(state, "resolved_ambiguities", {}) or {},
            indent=2,
            default=str,
        ),
        evidence=json.dumps(
            state.evidence,
            indent=2,
            default=str,
        ),
    )

    return llm.invoke(prompt).content