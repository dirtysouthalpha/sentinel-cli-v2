"""Core types for Sentinel CLI V2.

Frozen dataclasses — immutable, hashable, clean serialization.
Pattern stolen from ECC's src/llm/types.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class FinishReason(str, Enum):
    STOP = "stop"
    TOOL_USE = "tool_use"
    LENGTH = "length"
    ERROR = "error"


@dataclass(frozen=True)
class ToolDefinition:
    """A tool the agent can call."""
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The output of a tool execution."""
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    """A single message in the conversation."""
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    name: str | None = None  # for tool role

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "input": tc.input}
                for tc in self.tool_calls
            ]
        if self.tool_results:
            d["tool_results"] = [
                {"tool_call_id": tr.tool_call_id, "content": tr.content, "is_error": tr.is_error}
                for tr in self.tool_results
            ]
        if self.name:
            d["name"] = self.name
        return d


@dataclass(frozen=True)
class LLMInput:
    """What we send to the provider."""
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    system_prompt: str | None = None
    model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.tools:
            d["tools"] = [t.to_dict() for t in self.tools]
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        if self.model:
            d["model"] = self.model
        return d


@dataclass(frozen=True)
class LLMOutput:
    """What the provider returns."""
    message: Message
    finish_reason: FinishReason
    usage: UsageCost
    model: str
    raw: dict[str, Any] = field(default_factory=dict)  # provider-specific


@dataclass(frozen=True)
class UsageCost:
    """Token usage and cost for a single call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    provider: str = ""
    model: str = ""

    def __add__(self, other: UsageCost) -> UsageCost:
        return UsageCost(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            provider=self.provider if self.provider == other.provider else "mixed",
            model=self.model if self.model == other.model else "mixed",
        )


@dataclass
class SessionState:
    """Mutable session state — persists across turns."""
    session_id: str = ""
    messages: list[Message] = field(default_factory=list)
    total_usage: UsageCost = field(default_factory=UsageCost)
    tool_call_count: int = 0
    iteration_count: int = 0
    max_iterations: int = 25
    compacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "total_usage": asdict(self.total_usage),
            "tool_call_count": self.tool_call_count,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "compacted": self.compacted,
            "metadata": self.metadata,
        }


def usage_from_dict(d: dict[str, Any]) -> UsageCost:
    return UsageCost(
        prompt_tokens=d.get("prompt_tokens", 0),
        completion_tokens=d.get("completion_tokens", 0),
        total_tokens=d.get("total_tokens", 0),
        cost_usd=d.get("cost_usd", 0.0),
        provider=d.get("provider", ""),
        model=d.get("model", ""),
    )
