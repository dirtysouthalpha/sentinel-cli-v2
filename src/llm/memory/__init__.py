"""Memory backends for Sentinel CLI V2.

Tiered: local SQLite cache first (fast, TTL), Neuralis brain API as the
durable store. `TieredMemory` composes both; `create_default_memory()`
builds the standard stack.
"""

from __future__ import annotations

from .cache import (
    BrainBackend,
    MemoryBackend,
    MemoryEntry,
    SQLiteCache,
    TieredMemory,
)

__all__ = [
    "BrainBackend",
    "MemoryBackend",
    "MemoryEntry",
    "SQLiteCache",
    "TieredMemory",
]


def create_default_memory(base_url: str = "http://100.86.200.42:8000") -> TieredMemory:
    """Build the standard memory stack: SQLite cache over the brain API."""
    return TieredMemory(
        cache=SQLiteCache(),
        brain=BrainBackend(base_url=base_url),
    )


# Alias kept so `from src.llm.memory import NeuralisMemory` keeps working.
NeuralisMemory = TieredMemory
