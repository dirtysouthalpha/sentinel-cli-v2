"""Echo provider — test provider that simulates tool calls."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from ..base import LLMProvider
from ..config import ProviderConfig
from ..types import LLMInput, LLMOutput, Message, Role, FinishReason, ToolCall


class EchoProvider(LLMProvider):
    """Test provider — echoes input back, simulates tool calls."""

    def __init__(self, config: ProviderConfig, model: str | None = None):
        super().__init__(config, model or "echo")

    def validate(self) -> bool:
        return True

    def resolve_model(self, model: str | None = None) -> str:
        return "echo"

    def generate(self, input: LLMInput) -> LLMOutput:
        last_msg = input.messages[-1] if input.messages else Message(role=Role.USER, content="")
        content = last_msg.content

        # Simulate tool call if content mentions a tool
        tool_calls: list[ToolCall] = []
        if "run" in content.lower() or "execute" in content.lower():
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name="shell",
                input={"command": "echo 'Hello from Sentinel CLI'"},
            ))

        return LLMOutput(
            message=Message(
                role=Role.ASSISTANT,
                content=f"[Echo] {content}",
                tool_calls=tuple(tool_calls) if tool_calls else (),
            ),
            finish_reason=FinishReason.TOOL_USE if tool_calls else FinishReason.STOP,
            usage=self.usage(prompt_tokens=len(content.split()), completion_tokens=10),
            model="echo",
        )

    async def agenerate(self, input: LLMInput) -> LLMOutput:
        return self.generate(input)
