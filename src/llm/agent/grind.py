"""Grind mode — autonomous execution with human gate.

Inspired by the brain's "grind" instruction (n.128577): give prompts and goals,
the system just grinds until done, then audits the work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..types import Message, Role
from . import ReActAgent, AgentConfig
from ..memory.cache import TieredMemory


class GrindState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


@dataclass
class GrindResult:
    """Result of a grind session."""
    goal: str
    state: GrindState
    iterations: int = 0
    total_cost: float = 0.0
    duration_s: float = 0.0
    output: str = ""
    errors: list[str] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.state == GrindState.DONE and not self.errors


class GrindMode:
    """Autonomous grinding with audit and human gate."""

    def __init__(
        self,
        agent: ReActAgent | None = None,
        memory: TieredMemory | None = None,
        human_gate: bool = True,
        max_iterations: int = 50,
    ):
        self.agent = agent or ReActAgent(AgentConfig())
        self.memory = memory or TieredMemory()
        self.human_gate = human_gate
        self.max_iterations = max_iterations
        self.state = GrindState.IDLE
        self._log: list[dict[str, Any]] = []

    def grind(self, goal: str, context: str = "") -> GrindResult:
        """Execute a goal autonomously until completion."""
        start = time.time()
        self.state = GrindState.RUNNING
        self._log = []

        self.memory.remember(
            f"grind_goal_{int(start)}",
            goal,
            source="grind_mode",
            ttl=86400,
        )

        prompt = self._build_prompt(goal, context)
        self._log_event("start", {"goal": goal, "context": context})

        try:
            output = self.agent.run(prompt)
            self._log_event("complete", {"output_length": len(output)})
            self.state = GrindState.DONE
            audit = self._audit(output, goal)

            return GrindResult(
                goal=goal,
                state=self.state,
                iterations=self.agent.session.state.iteration_count,
                total_cost=self.agent.session.total_usage.cost_usd,
                duration_s=time.time() - start,
                output=output,
                audit=audit,
            )

        except Exception as e:
            self.state = GrindState.ERROR
            self._log_event("error", {"error": str(e)})
            return GrindResult(
                goal=goal,
                state=self.state,
                iterations=self.agent.session.state.iteration_count,
                total_cost=self.agent.session.total_usage.cost_usd,
                duration_s=time.time() - start,
                errors=[str(e)],
            )

    def _build_prompt(self, goal: str, context: str) -> str:
        parts = [f"## Goal\n{goal}\n"]
        if context:
            parts.append(f"## Context\n{context}\n")
        memories = self.memory.recall(goal, limit=3)
        if memories:
            parts.append("## Relevant Memories\n")
            for m in memories:
                parts.append(f"- {m.content[:200]}")
            parts.append("")
        parts.append("## Instructions\n")
        parts.append("Work autonomously to complete this goal. ")
        parts.append("Use tools as needed. Be thorough and verify your work. ")
        parts.append("When done, summarize what you accomplished.\n")
        return "\n".join(parts)

    def _audit(self, output: str, goal: str) -> dict[str, Any]:
        audit = {
            "goal": goal,
            "output_length": len(output),
            "has_output": bool(output.strip()),
            "tool_calls": self.agent.session.state.tool_call_count,
            "iterations": self.agent.session.state.iteration_count,
            "cost": self.agent.session.total_usage.total_cost,
        }
        audit["checks"] = {
            "output_not_empty": audit["has_output"],
            "reasonable_length": len(output) > 50,
            "tools_used": audit["tool_calls"] > 0,
            "within_iterations": audit["iterations"] < self.max_iterations,
        }
        audit["passed"] = all(audit["checks"].values())
        return audit

    def _log_event(self, event: str, data: dict[str, Any]) -> None:
        self._log.append({"event": event, "timestamp": time.time(), **data})

    @property
    def log(self) -> list[dict[str, Any]]:
        return list(self._log)
