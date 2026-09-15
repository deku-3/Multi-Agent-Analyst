"""
Resolve a possibly-ambiguous follow-up question into a standalone one,
using conversation history, BEFORE it ever reaches the planner.

WHY THIS EXISTS
    Each turn in app.py starts a brand-new InvestigationState - the
    planner has no memory of earlier turns. A question like "which
    product category performed best THAT YEAR" has no referent without
    the previous turn's answer. Without this step, "that year" silently
    resolves to whatever the planner's own default happens to be (e.g.
    the latest complete year) - which may not be what the user meant,
    and there is no error, just a confident wrong answer.

    This mirrors the existing follow-up-rewrite pattern already proven
    in src/sql_ag.py's `rewrite()` node - same idea, one level up: that
    one resolves references WITHIN a single SQL Worker call; this one
    resolves references ACROSS a whole investigation/turn in the chat.
"""

from __future__ import annotations

from typing import Optional


REWRITE_PROMPT = """Given the conversation so far and a new user question, rewrite the
new question as a single, fully standalone question that needs no prior context.

Resolve references such as "it", "them", "those", "that year", "that category",
"compared to before", "the previous one" using the conversation - including concrete
values from previous answers (a year, a category, a metric) when that is what the
user is referring to.

If the new question is about a DIFFERENT subject than the conversation, do NOT graft
the old subject onto it - return the new question unchanged.

If the question is already standalone, return it unchanged.

Return ONLY the rewritten question, nothing else.

CONVERSATION SO FAR:
{history}

NEW QUESTION: {question}

STANDALONE QUESTION:
"""


def _format_history(history: list[dict], max_turns: int = 5) -> str:
    if not history:
        return "(no prior turns)"
    lines = []
    for turn in history[-max_turns:]:
        lines.append(f"user: {turn['question']}")
        lines.append(f"assistant: {turn['answer']}")
    return "\n".join(lines)


def resolve_followup(
    question: str,
    history: list[dict],
    llm=None,
) -> str:
    """
    Return a standalone version of `question`, using `history` (a list
    of {"question": str, "answer": str} dicts, oldest first).

    Falls back to the ORIGINAL question, unchanged, whenever anything
    is uncertain: no history yet, the rewrite call fails, or the
    rewrite looks suspicious. A broken rewrite step must never block
    an investigation - it should just degrade to "use what was typed".
    """
    if not history:
        return question

    if llm is None:
        from src.planner import llm as default_llm  # lazy: reuse the already-configured model
        llm = default_llm

    try:
        rewritten = llm.invoke(
            REWRITE_PROMPT.format(
                history=_format_history(history),
                question=question,
            )
        ).content.strip()
    except Exception:
        return question

    # Distrust suspicious rewrites - same guard src/sql_ag.py's rewrite() uses.
    if not rewritten or len(rewritten) > 300:
        return question

    return rewritten