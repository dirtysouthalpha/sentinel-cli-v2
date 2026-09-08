"""Structured logging for Sentinel CLI — JSONL format for machine parsing."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
from pathlib import Path


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class EventCategory(Enum):
    AGENT = "agent"
    PROVIDER = "provider"
    TOOL = "tool"
    SESSION = "session"
    ROUTER = "router"
    COST = "cost"
    MEMORY = "memory"
    PLUGIN = "plugin"


@dataclass
class LogEntry:
    """A single structured log entry."""
    timestamp: float
    level: str
    category: str
    event: str
    session_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def to_json(self) -> str:
        return json.dumps({
            "ts": self.timestamp,
            "lvl": self.level,
            "cat": self.category,
            "evt": self.event,
            "sid": self.session_id,
            "data": self.data,
            "id": self.entry_id,
        }, default=str)
    
    @classmethod
    def from_json(cls, line: str) -> "LogEntry":
        d = json.loads(line)
        return cls(
            timestamp=d.get("ts", 0),
            level=d.get("lvl", "info"),
            category=d.get("cat", "agent"),
            event=d.get("evt", ""),
            session_id=d.get("sid", ""),
            data=d.get("data", {}),
            entry_id=d.get("id", ""),
        )


class StructuredLogger:
    """Writes structured JSONL logs to a file."""
    
    def __init__(self, log_dir: str = "~/.sentinel/logs"):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: Optional[Path] = None
        self._file_handle: Optional[Any] = None
    
    def _get_file(self) -> Path:
        """Get the current log file path (rotates daily)."""
        date_str = time.strftime("%Y-%m-%d")
        return self.log_dir / f"sentinel-{date_str}.jsonl"
    
    def _open(self) -> Any:
        """Open the log file for writing."""
        log_file = self._get_file()
        if self._current_file != log_file:
            if self._file_handle:
                self._file_handle.close()
            self._current_file = log_file
            self._file_handle = open(log_file, "a")
        return self._file_handle
    
    def log(
        self,
        event: str,
        category: EventCategory = EventCategory.AGENT,
        level: LogLevel = LogLevel.INFO,
        session_id: str = "",
        **data: Any,
    ) -> LogEntry:
        """Write a log entry."""
        entry = LogEntry(
            timestamp=time.time(),
            level=level.value,
            category=category.value,
            event=event,
            session_id=session_id,
            data=data,
        )
        
        fh = self._open()
        fh.write(entry.to_json() + "\n")
        fh.flush()
        
        return entry
    
    def info(self, event: str, **data: Any) -> LogEntry:
        return self.log(event, level=LogLevel.INFO, **data)
    
    def warn(self, event: str, **data: Any) -> LogEntry:
        return self.log(event, level=LogLevel.WARN, **data)
    
    def error(self, event: str, **data: Any) -> LogEntry:
        return self.log(event, level=LogLevel.ERROR, **data)
    
    def debug(self, event: str, **data: Any) -> LogEntry:
        return self.log(event, level=LogLevel.DEBUG, **data)
    
    def close(self) -> None:
        """Close the log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
    
    def read_logs(
        self,
        session_id: Optional[str] = None,
        category: Optional[EventCategory] = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """Read log entries from the current file."""
        log_file = self._get_file()
        if not log_file.exists():
            return []
        
        entries = []
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = LogEntry.from_json(line)
                    if session_id and entry.session_id != session_id:
                        continue
                    if category and entry.category != category.value:
                        continue
                    entries.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return entries[-limit:]
    
    def summary(self, session_id: str) -> dict[str, Any]:
        """Get a summary of a session's activity."""
        entries = self.read_logs(session_id=session_id, limit=1000)
        
        if not entries:
            return {"session_id": session_id, "events": 0}
        
        categories: dict[str, int] = {}
        event_types: dict[str, int] = {}
        errors = 0
        
        for e in entries:
            categories[e.category] = categories.get(e.category, 0) + 1
            event_types[e.event] = event_types.get(e.event, 0) + 1
            if e.level == "error":
                errors += 1
        
        return {
            "session_id": session_id,
            "events": len(entries),
            "first_event": entries[0].timestamp,
            "last_event": entries[-1].timestamp,
            "categories": categories,
            "event_types": event_types,
            "errors": errors,
        }


# Global logger instance
_logger: Optional[StructuredLogger] = None


def get_logger(log_dir: str = "~/.sentinel/logs") -> StructuredLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger(log_dir)
    return _logger
