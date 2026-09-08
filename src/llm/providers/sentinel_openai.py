"""OpenAI provider — OpenAI SDK wrapper (works with OpenRouter, Together, etc.)."""

from __future__ import annotations

import asyncio
from typing import Any

from ..base import LLMProvider
from ..config import ProviderConfig
from ..types import LLMInput, LLMOutput, Message, Role, FinishReason, ToolDefinition, ToolCall


_FINISH_MAP = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_USE,
    "length": FinishReason.LENGTH,
}


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider using the OpenAI SDK."""

    def __init__(self, config: ProviderConfig, model: str | None = None):
        super().__init__(config, model)
        try:
            from openai import OpenAI
            kwargs: dict[str, Any] = {"api_key": config.api_key or "EMPTY"}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            self.client = OpenAI(**kwargs)
        except ImportError:
            raise RuntimeError("Install the OpenAI SDK: pip install openai")

    def validate(self) -> bool:
        return True  # local providers don't need a key

    def resolve_model(self, model: str | None = None) -> str:
        return model or self.model

    def _to_openai_messages(self, messages: tuple[Message, ...]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == Role.TOOL:
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.name,
                    "content": msg.content,
                })
            elif msg.tool_calls:
                result.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": str(tc.input),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                result.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return result

    def _from_openai(self, response: Any) -> LLMOutput:
        choice = response.choices[0]
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=tc.function.arguments if isinstance(tc.function.arguments, dict) else {},
                ))

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return LLMOutput(
            message=Message(
                role=Role.ASSISTANT,
                content=choice.message.content or "",
                tool_calls=tuple(tool_calls) if tool_calls else (),
            ),
            finish_reason=_FINISH_MAP.get(choice.finish_reason, FinishReason.STOP),
            usage=self.usage(prompt_tokens, completion_tokens),
            model=response.model,
            raw={"response": response.model_dump()},
        )

    def generate(self, input: LLMInput) -> LLMOutput:
        kwargs: dict[str, Any] = {
            "model": self.resolve_model(input.model),
            "max_tokens": input.max_tokens,
            "messages": self._to_openai_messages(input.messages),
        }
        if input.system_prompt:
            kwargs["messages"].insert(0, {"role": "system", "content": input.system_prompt})
        if input.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in input.tools
            ]

        response = self.client.chat.completions.create(**kwargs)
        return self._from_openai(response)

    async def agenerate(self, input: LLMInput) -> LLMOutput:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, input)
