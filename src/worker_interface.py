"""
The boundary between the controller and the SQL Worker.

The controller depends on this Protocol, not on sql_ag directly. That
keeps the SQL Worker swappable and, importantly, testable: FakeWorker
lets the whole investigation loop run with no database, chromadb, or API
key. SqlAgWorker adapts the real LangGraph agent (imported lazily).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol


@dataclass
class WorkerResult:
    answer: str                 # natural-language answer to the subquestion
    sql: str = ""               # SQL executed (for evidence/audit)
    raw_result: str = ""        # raw rows (for evidence/audit)
    ok: bool = True
    error: str = ""


class SQLWorker(Protocol):
    def run(self, subquestion: str) -> WorkerResult: ...


class FakeWorker:
    """Deterministic worker for tests. Optionally maps subquestion->answer."""

    def __init__(self, answers: Optional[dict] = None):
        self.answers = answers or {}
        self.calls: list[str] = []

    def run(self, subquestion: str) -> WorkerResult:
        self.calls.append(subquestion)
        answer = self.answers.get(
            subquestion,
            f"[fake answer for: {subquestion}]",
        )
        return WorkerResult(
            answer=answer,
            sql="SELECT /* fake */ 1",
            raw_result="[(1,)]",
        )


class SqlAgWorker:
    """
    Adapter over the real sql_ag agent.

    Lazy-imports sql_ag so importing the controller never pulls in
    chromadb/DB/keys. `ask_fn` can be injected to avoid the import
    entirely (or to point at chat_graph instead of the one-shot graph).

    NOTE: this requires sql_ag.py's imports to be package-relative
    (`from src.vectorstore import ...`) so `from src.sql_ag import ask`
    resolves. That conversion is the one remaining integration edit.
    """

    def __init__(self, ask_fn: Optional[Callable] = None):
        self._ask = ask_fn

    def run(self, subquestion: str) -> WorkerResult:
        if self._ask is None:
            from src.sql_ag import ask  # lazy
            self._ask = ask

        final = self._ask(subquestion)

        try:
            msg = final["messages"][-1]
            answer = getattr(msg, "content", str(msg))
            return WorkerResult(
                answer=answer,
                sql=final.get("query", ""),
                raw_result=str(final.get("result", "")),
            )
        except Exception as exc:  # worker returned something unexpected
            return WorkerResult(
                answer="",
                ok=False,
                error=f"worker output parse failed: {exc}",
            )