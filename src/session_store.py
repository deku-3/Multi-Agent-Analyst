"""
Per-investigation session storage - the seam that makes this multi-user
safe.

WHY THIS FILE EXISTS:
The #1 way a multi-user Python web app breaks is holding request/session
data in a module-level global. The moment two users hit the app at once,
their InvestigationState/InvestigationBudget objects collide. Every
investigation here is instead looked up by an opaque investigation_id,
and the frontend (app.py) is written so nothing about a specific user's
investigation ever lives in a global - only in a local variable inside
that user's own connection scope, or in this store, keyed by id.

SCALING PATH:
- Right now: InMemorySessionStore. Fine for one process (one machine).
  Thread-safe via a lock, since NiceGUI's run.io_bound runs each
  investigation in a worker thread.
- Next: swap in a Redis- or Postgres-backed SessionStore that
  serializes InvestigationState/InvestigationBudget as JSON. Same
  .get/.set/.list_recent interface, so app.py does not change. This is
  what actually lets you run more than one server process/machine
  behind a load balancer - investigation state is no longer pinned to
  whichever process happened to handle the first request.
- NiceGUI itself still wants sticky sessions at the load balancer for
  its own websocket connection (any websocket app needs this) - that is
  independent of this store and unavoidable with a live-updating UI.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class SessionRecord:
    investigation_id: str
    question: str
    status: str = "running"          # running | needs_clarification | complete | stopped
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    answer: str = ""
    evidence_count: int = 0
    # Owner tag (e.g. browser client id) - lets a store scope "recent
    # investigations" per user once you have real auth; unused for now.
    owner: str = ""


class SessionStore(Protocol):
    def create(self, question: str, owner: str = "") -> str: ...
    def get(self, investigation_id: str) -> Optional[SessionRecord]: ...
    def update(self, investigation_id: str, **fields) -> None: ...
    def list_recent(self, owner: str = "", limit: int = 20) -> list: ...


class InMemorySessionStore:
    """
    Thread-safe in-memory implementation. See module docstring for the
    scaling path off of this.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._records: dict[str, SessionRecord] = {}

    def create(self, question: str, owner: str = "") -> str:
        investigation_id = uuid.uuid4().hex
        with self._lock:
            self._records[investigation_id] = SessionRecord(
                investigation_id=investigation_id,
                question=question,
                owner=owner,
            )
        return investigation_id

    def get(self, investigation_id: str) -> Optional[SessionRecord]:
        with self._lock:
            return self._records.get(investigation_id)

    def update(self, investigation_id: str, **fields) -> None:
        with self._lock:
            record = self._records.get(investigation_id)
            if record is None:
                return
            for key, value in fields.items():
                setattr(record, key, value)
            record.updated_at = time.time()

    def list_recent(self, owner: str = "", limit: int = 20) -> list:
        with self._lock:
            records = list(self._records.values())
        if owner:
            records = [r for r in records if r.owner == owner]
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records[:limit]


# Process-wide singleton for the demo. In a multi-process deployment this
# becomes a shared Redis/Postgres client instead - everything that uses
# `session_store` below would be unaffected.
session_store: SessionStore = InMemorySessionStore()