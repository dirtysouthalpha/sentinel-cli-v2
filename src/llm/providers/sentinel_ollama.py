"""Ollama provider — OpenAI-compatible, no SDK needed."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

from ..base import LLMProvider
from ..config import ProviderConfig
from ..types import LLMInput, LLMOutput, Message, Role, FinishReason, ToolDefinition, ToolCall


class OllamaProvider(LLMProvider):
    """Ollama / llama-server provider using raw HTTP requests.

    Supports both the Ollama /api/chat and OpenAI-compatible /v1/chat/completions endpoints.
    """

    def __init__(self, config: ProviderConfig, model: str | None = None):
        super().__init__(config, model)
        self.base_url = config.base_url or "http://localhost:11434"

    def validate(self) -> bool:
        return True  # local providers don't need a key

    def resolve_model(self, model: str | None = None) -> str:
        return model or self.model

    def _http_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"HTTP error to {url}: {e}")

    def generate(self, input: LLMInput) -> LLMOutput:
        # Try OpenAI-compatible endpoint first, then fall back to Ollama native
        try:
            return self._generate_openai(input)
        except Exception:
            return self._generate_ollama(input)

    def _generate_openai(self, input: LLMInput) -> LLMOutput:
        messages = []
        if input.system_prompt:
            messages.append({"role": "system", "content": input.system_prompt})
        for msg in input.messages:
            if msg.role == Role.TOOL:
                messages.append({"role": "tool", "tool_call_id": msg.name, "content": msg.content})
            elif msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.input)}}
                        for tc in msg.tool_calls
                    ],
                })
            else:
                messages.append({"role": msg.role.value, "content": msg.content})

        payload: dict[str, Any] = {
            "model": self.resolve_model(input.model),
            "messages": messages,
            "stream": False,
        }
        if input.tools:
            payload["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}} for t in input.tools]

        resp = self._http_post("/v1/chat/completions", payload)
        return self._parse_openai_response(resp)

    def _generate_ollama(self, input: LLMInput) -> LLMOutput:
        messages = []
        if input.system_prompt:
            messages.append({"role": "system", "content": input.system_prompt})
        for msg in input.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        payload: dict[str, Any] = {
            "model": self.resolve_model(input.model),
            "messages": messages,
            "stream": False,
        }
        resp = self._http_post("/api/chat", payload)

        message = resp.get("message", {})
        prompt_tokens = resp.get("prompt_eval_count", 0)
        completion_tokens = resp.get("eval_count", 0)

        return LLMOutput(
            message=Message(
                role=Role.ASSISTANT,
                content=message.get("content", ""),
            ),
            finish_reason=FinishReason.STOP,
            usage=self.usage(prompt_tokens, completion_tokens),
            model=resp.get("model", self.model),
            raw=resp,
        )

    def _parse_openai_response(self, resp: dict[str, Any]) -> LLMOutput:
        choice = resp["choices"][0]
        tool_calls: list[ToolCall] = []
        if choice.get("message", {}).get("tool_calls"):
            for tc in choice["message"]["tool_calls"]:
                args = tc["function"].get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    input=args if isinstance(args, dict) else {},
                ))

        usage = resp.get("usage", {})
        return LLMOutput(
            message=Message(
                role=Role.ASSISTANT,
                content=choice.get("message", {}).get("content", "") or "",
                tool_calls=tuple(tool_calls) if tool_calls else (),
            ),
            finish_reason=FinishReason.TOOL_USE if tool_calls else FinishReason.STOP,
            usage=self.usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
            model=resp.get("model", self.model),
            raw=resp,
        )

    async def agenerate(self, input: LLMInput) -> LLMOutput:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, input)
