"""Session persistence — save/load sessions to disk."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import Message, SessionState, UsageCost, usage_from_dict


@dataclass
class Session:
    """A conversation session with persistence."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    state: SessionState = field(default_factory=SessionState)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def messages(self) -> list[Message]:
        return self.state.messages

    @property
    def total_usage(self) -> UsageCost:
        return self.state.total_usage

    def add_message(self, message: Message) -> None:
        self.state.messages.append(message)
        self.updated_at = datetime.now().isoformat()

    def add_tool_call(self) -> None:
        self.state.tool_call_count += 1

    def add_usage(self, usage: UsageCost) -> None:
        self.state.total_usage = self.state.total_usage + usage

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Session:
        state = SessionState(
            session_id=d["state"].get("session_id", d.get("session_id", "")),
            messages=[Message(**m) for m in d["state"].get("messages", [])],
            total_usage=usage_from_dict(d["state"].get("total_usage", {})),
            tool_call_count=d["state"].get("tool_call_count", 0),
            iteration_count=d["state"].get("iteration_count", 0),
            max_iterations=d["state"].get("max_iterations", 25),
            compacted=d["state"].get("compacted", False),
            metadata=d["state"].get("metadata", {}),
        )
        return cls(
            session_id=d.get("session_id", state.session_id),
            state=state,
            created_at=d.get("created_at", datetime.now().isoformat()),
            updated_at=d.get("updated_at", datetime.now().isoformat()),
        )


class SessionStore:
    """Persists sessions to disk."""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir or Path.home() / ".sentinel" / "sessions")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def save(self, session: Session) -> None:
        path = self._path(session.session_id)
        path.write_text(json.dumps(session.to_dict(), indent=2, default=str))

    def load(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return Session.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError):
            return None

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for path in sorted(self.base_dir.glob("*.json"), reverse=True):
            try:
                d = json.loads(path.read_text())
                sessions.append({
                    "session_id": d.get("session_id", path.stem),
                    "created_at": d.get("created_at", ""),
                    "updated_at": d.get("updated_at", ""),
                    "messages": len(d.get("state", {}).get("messages", [])),
                    "total_cost": d.get("state", {}).get("total_usage", {}).get("cost_usd", 0),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def latest(self) -> Session | None:
        """Get the most recently modified session."""
        sessions = sorted(self.base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sessions:
            return None
        try:
            return Session.from_dict(json.loads(sessions[0].read_text()))
        except (json.JSONDecodeError, KeyError):
            return None
