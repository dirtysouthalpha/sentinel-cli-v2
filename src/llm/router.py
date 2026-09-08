"""Model router — haiku/sonnet/opus heuristic with budget flag."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteRule:
    name: str
    pattern: str  # regex on task description
    model: str
    budget_usd: float = 0.50


DEFAULT_RULES = [
    RouteRule("boilerplate", r"\b(boilerplate|template|scaffold|init)\b", "claude-haiku-4-5", 0.10),
    RouteRule("refactor", r"\b(refactor|rename|reorganize)\b", "claude-sonnet-4-5", 0.30),
    RouteRule("impl", r"\b(implement|build|create|add feature)\b", "claude-sonnet-4-5", 0.50),
    RouteRule("review", r"\b(review|audit|security|architecture)\b", "claude-opus-4-5", 1.00),
    RouteRule("debug", r"\b(debug|fix|bug|error)\b", "claude-opus-4-5", 0.80),
]


class ModelRouter:
    """Routes tasks to appropriate models based on heuristics."""

    def __init__(self, rules: list[RouteRule] | None = None):
        import re
        self.rules = rules or DEFAULT_RULES
        self._compiled = [(re.compile(r.pattern, re.I), r) for r in self.rules]

    def route(self, task_description: str) -> tuple[str, float]:
        """Return (model, budget) for the task."""
        for regex, rule in self._compiled:
            if regex.search(task_description):
                return rule.model, rule.budget_usd
        # Default to sonnet for unknown tasks
        return "claude-sonnet-4-5", 0.40
