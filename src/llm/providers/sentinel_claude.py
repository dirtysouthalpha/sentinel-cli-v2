"""Claude provider — Anthropic SDK wrapper."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from ..base import LLMProvider
from ..config import ProviderConfig
from ..types import LLMInput, LLMOutput, Message, Role, FinishReason, ToolDefinition, ToolCall, ToolResult


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider using the official SDK."""

    def __init__(self, config: ProviderConfig, model: str | None = None):
        super().__init__(config, model)
        try:
            import anthropic
            self.client = anthropic.Anthropic()  # SDK reads ANTHROPIC_API_KEY env
        except ImportError:
            raise RuntimeError("Install the Anthropic SDK: pip install anthropic")

    def validate(self) -> bool:
        return bool(self.config.api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def resolve_model(self, model: str | None = None) -> str:
        return model or self.model

    def _to_anthropic_messages(self, messages: tuple[Message, ...]) -> list[dict[str, Any]]:
        """Convert internal messages to Anthropic API format."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == Role.TOOL:
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.name,
                            "content": msg.content,
                            "is_error": any(tr.is_error for tr in msg.tool_results),
                        }
                    ],
                })
            elif msg.tool_calls:
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                result.append({"role": "assistant", "content": content})
            else:
                result.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return result

    def _to_anthropic_tools(self, tools: tuple[ToolDefinition, ...]) -> list[dict[str, Any]]:
        """Convert internal tool definitions to Anthropic API format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def _from_anthropic(self, response: Any) -> LLMOutput:
        """Convert Anthropic response to internal format."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish: FinishReason = FinishReason.STOP

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))
                finish = FinishReason.TOOL_USE

        usage = response.usage
        return LLMOutput(
            message=Message(
                role=Role.ASSISTANT,
                content="\n".join(content_parts),
                tool_calls=tuple(tool_calls) if tool_calls else (),
            ),
            finish_reason=finish,
            usage=self.usage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            ),
            model=response.model,
            raw={"response": response.model_dump()},
        )

    def generate(self, input: LLMInput) -> LLMOutput:
        kwargs: dict[str, Any] = {
            "model": self.resolve_model(input.model),
            "max_tokens": input.max_tokens,
            "messages": self._to_anthropic_messages(input.messages),
        }
        if input.system_prompt:
            kwargs["system"] = input.system_prompt
        if input.tools:
            kwargs["tools"] = self._to_anthropic_tools(input.tools)

        response = self.client.messages.create(**kwargs)
        return self._from_anthropic(response)

    async def agenerate(self, input: LLMInput) -> LLMOutput:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, input)
