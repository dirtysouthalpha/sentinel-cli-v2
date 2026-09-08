"""Memory integration — connects to Neuralis brain (:8000) for recall."""

from __future__ import annotations

import os
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None


class NeuralisMemory:
    """Talks to the Neuralis brain API for long-term memory."""

    def __init__(self, base_url: str | None = None):
        if httpx is None:
            raise RuntimeError("httpx not installed")
        self.base_url = base_url or os.environ.get("NEURALIS_URL", "http://100.86.200.42:8000")

    def recall(self, context: str, limit: int = 5) -> list[dict[str, Any]]:
        """Recall relevant memories for the given context."""
        try:
            resp = httpx.get(
                f"{self.base_url}/recall",
                params={"context": context, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("neurons", [])
        except Exception:
            return []

    def remember(self, content: str, region: str = "knowledge", topic: str = "sentinel-cli") -> bool:
        """Store a new memory."""
        try:
            resp = httpx.post(
                f"{self.base_url}/neurons",
                json={"content": content, "region": region, "topic": topic},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search the brain for relevant knowledge."""
        try:
            resp = httpx.get(
                f"{self.base_url}/neurons/search",
                params={"q": query},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("neurons", [])
        except Exception:
            return []
