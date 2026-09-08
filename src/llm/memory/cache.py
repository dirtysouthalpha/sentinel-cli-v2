"""Memory cache layer — local SQLite with brain API fallback.

Provides fast local caching of memories with TTL, falling back to the
Neuralis brain API (:8000) for misses. All writes go to both layers.
"""

from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryEntry:
    """A single memory entry."""
    key: str
    content: str
    source: str = "local"
    timestamp: float = field(default_factory=time.time)
    ttl: float = 3600.0  # Default 1 hour TTL
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }


class MemoryBackend(ABC):
    """Abstract memory backend."""

    @abstractmethod
    def recall(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Recall memories matching a query."""
        ...

    @abstractmethod
    def remember(self, key: str, content: str, **metadata) -> None:
        """Store a memory."""
        ...

    @abstractmethod
    def forget(self, key: str) -> bool:
        """Remove a memory. Returns True if found and removed."""
        ...

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Search memories by content."""
        ...


class SQLiteCache(MemoryBackend):
    """Local SQLite cache for fast memory access."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".sentinel" / "cache" / "memory.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT DEFAULT 'local',
                    timestamp REAL NOT NULL,
                    ttl REAL DEFAULT 3600,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp
                ON memories(timestamp)
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def recall(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Recall by key match or content match."""
        with self._connect() as conn:
            # First try exact key match
            row = conn.execute(
                "SELECT * FROM memories WHERE key = ? AND (? - timestamp) < ttl",
                (query, time.time())
            ).fetchone()
            if row:
                return [self._row_to_entry(row)]

            # Then try content search
            rows = conn.execute(
                """SELECT * FROM memories
                   WHERE content LIKE ? AND (? - timestamp) < ttl
                   ORDER BY timestamp DESC LIMIT ?""",
                (f"%{query}%", time.time(), limit)
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def remember(self, key: str, content: str, **metadata) -> None:
        ttl = metadata.pop("ttl", 3600.0)
        source = metadata.pop("source", "local")
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memories (key, content, source, timestamp, ttl, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (key, content, source, time.time(), ttl, json.dumps(metadata))
            )

    def forget(self, key: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        return self.recall(query, limit)

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE ? - timestamp > ttl",
                (time.time(),)
            )
            return cursor.rowcount

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            key=row[0],
            content=row[1],
            source=row[2],
            timestamp=row[3],
            ttl=row[4],
            metadata=json.loads(row[5]),
        )


class BrainBackend(MemoryBackend):
    """Neuralis brain API backend (:8000)."""

    def __init__(self, base_url: str = "http://100.86.200.42:8000"):
        self.base_url = base_url

    def recall(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        try:
            import httpx
            resp = httpx.get(
                f"{self.base_url}/recall",
                params={"context": query, "limit": limit},
                timeout=10,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            # Brain /recall returns hits under "direct" (plus associated/
            # anticipated/temporal buckets). Merge them all, direct first.
            results = list(data.get("direct") or [])
            seen = {r.get("id") for r in results if isinstance(r, dict)}
            for bucket in ("associated", "anticipated", "temporal", "results"):
                for r in data.get(bucket) or []:
                    if isinstance(r, dict) and r.get("id") not in seen:
                        seen.add(r.get("id"))
                        results.append(r)
            return [
                MemoryEntry(
                    key=r.get("id", str(i)),
                    content=r.get("content", ""),
                    source="brain",
                    timestamp=r.get("timestamp", time.time()),
                    ttl=86400.0,  # Brain memories last longer
                )
                for i, r in enumerate(results)
            ]
        except Exception:
            return []

    def remember(self, key: str, content: str, **metadata) -> None:
        try:
            import httpx
            httpx.post(
                f"{self.base_url}/neurons",
                json={
                    "content": content,
                    "region": metadata.get("region", "knowledge"),
                    "source": metadata.get("source", "sentinel-cli"),
                    "topic": key,
                },
                timeout=10,
            )
        except Exception:
            pass  # Brain writes are best-effort

    def forget(self, key: str) -> bool:
        # Brain doesn't support deletion via API
        return False

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        try:
            import httpx
            resp = httpx.get(
                f"{self.base_url}/neurons/search",
                params={"q": query, "limit": limit},
                timeout=10,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            results = data.get("results", data.get("neurons", []))
            return [
                MemoryEntry(
                    key=str(r.get("id", i)),
                    content=r.get("content", ""),
                    source="brain",
                    timestamp=r.get("timestamp", time.time()),
                    ttl=86400.0,
                )
                for i, r in enumerate(results)
            ]
        except Exception:
            return []


class TieredMemory(MemoryBackend):
    """Two-tier memory: local SQLite cache + brain API fallback.

    Reads check local first, then brain on miss.
    Writes go to both layers.
    """

    def __init__(
        self,
        cache: SQLiteCache | None = None,
        brain: BrainBackend | None = None,
    ):
        self.cache = cache or SQLiteCache()
        self.brain = brain or BrainBackend()

    @property
    def base_url(self) -> str:
        """Brain API base URL — proxied from the backing store."""
        return self.brain.base_url

    def recall(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        # Check local cache first
        local = self.cache.recall(query, limit)
        if local and not any(e.expired for e in local):
            return local

        # Fall back to brain
        brain_results = self.brain.recall(query, limit)
        for entry in brain_results:
            # Cache brain results locally for next time
            self.cache.remember(
                entry.key, entry.content,
                source="brain", ttl=entry.ttl, **entry.metadata
            )
        return brain_results

    def remember(self, key: str, content: str, **metadata) -> None:
        self.cache.remember(key, content, **metadata)
        self.brain.remember(key, content, **metadata)

    def forget(self, key: str) -> bool:
        local_removed = self.cache.forget(key)
        brain_removed = self.brain.forget(key)
        return local_removed or brain_removed

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        return self.recall(query, limit)
