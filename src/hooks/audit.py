"""Audit hook — logs every tool call with input/output/cost."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: float
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    duration_ms: float
    cost: float = 0.0
    model: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "tool": self.tool_name,
            "input": self.tool_input,
            "output": self.tool_output[:1000],  # Truncate for storage
            "duration_ms": self.duration_ms,
            "cost": self.cost,
            "model": self.model,
            "error": self.error,
        }


class AuditLog:
    """Persistent audit log."""

    def __init__(self, log_dir: str | Path | None = None):
        if log_dir is None:
            log_dir = Path.home() / ".sentinel" / "audit"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[AuditEntry] = []

    def log(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        duration_ms: float,
        cost: float = 0.0,
        model: str = "",
        error: str | None = None,
    ) -> None:
        entry = AuditEntry(
            timestamp=time.time(),
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            duration_ms=duration_ms,
            cost=cost,
            model=model,
            error=error,
        )
        self._entries.append(entry)
        self._append_to_file(entry)

    def _append_to_file(self, entry: AuditEntry) -> None:
        log_file = self.log_dir / f"{int(entry.timestamp)}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def summary(self) -> dict[str, Any]:
        if not self._entries:
            return {"total_calls": 0, "total_cost": 0.0}

        return {
            "total_calls": len(self._entries),
            "total_cost": sum(e.cost for e in self._entries),
            "total_duration_ms": sum(e.duration_ms for e in self._entries),
            "errors": sum(1 for e in self._entries if e.error),
            "tools": list(set(e.tool_name for e in self._entries)),
        }
