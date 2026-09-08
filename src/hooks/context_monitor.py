"""Strategic compaction — dual-signal (context size + tool-call count)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompactionThreshold:
    context_tokens: int = 160_000  # on a 200K window
    tool_call_count: int = 50


class StrategicCompact:
    """Suggests compaction at phase boundaries, not mid-task."""

    def __init__(self, threshold: CompactionThreshold | None = None):
        self.threshold = threshold or CompactionThreshold()
        self._tool_calls_since_compact = 0
        self._last_compact_context_size = 0

    def should_compact(self, current_context_tokens: int, tool_calls_this_turn: int = 0) -> bool:
        self._tool_calls_since_compact += tool_calls_this_turn
        # Dual signal: context size OR tool-call count
        if current_context_tokens >= self.threshold.context_tokens:
            return True
        if self._tool_calls_since_compact >= self.threshold.tool_call_count:
            return True
        return False

    def reset(self) -> None:
        self._tool_calls_since_compact = 0
        self._last_compact_context_size = 0


# ---------------------------------------------------------------------------
# Context monitor — scope creep / tool-loop / exhaustion warnings
# ---------------------------------------------------------------------------

class ContextMonitor:
    """Monitors context for anti-patterns."""

    def __init__(self):
        self._tool_call_history: list[str] = []

    def check_scope_creep(self, messages: list[dict]) -> str | None:
        """Detect if the conversation has drifted from the original task."""
        if len(messages) < 10:
            return None
        # Simple heuristic: if last 5 messages don't share keywords with the first
        first_content = str(messages[0].get("content", "")).lower()
        recent = " ".join(str(m.get("content", "")).lower() for m in messages[-5:])
        first_words = set(first_content.split()[:20])
        recent_words = set(recent.split()[:50])
        overlap = len(first_words & recent_words)
        if overlap < 3 and len(first_words) > 5:
            return "Scope drift detected — conversation has moved away from original task"
        return None

    def check_tool_loop(self, tool_name: str, window: int = 6) -> str | None:
        """Detect repetitive tool calls (same tool called N times in a row)."""
        self._tool_call_history.append(tool_name)
        if len(self._tool_call_history) >= window:
            recent = self._tool_call_history[-window:]
            if len(set(recent)) == 1:
                return f"Tool loop detected: {tool_name} called {window} times consecutively"
        return None

    def check_context_exhaustion(self, current_tokens: int, max_tokens: int = 200_000) -> str | None:
        """Warn when approaching context limit."""
        ratio = current_tokens / max_tokens
        if ratio >= 0.95:
            return f"CRITICAL: Context at {ratio:.0%} — immediate compaction required"
        if ratio >= 0.85:
            return f"WARNING: Context at {ratio:.0%} — consider compacting"
        return None
