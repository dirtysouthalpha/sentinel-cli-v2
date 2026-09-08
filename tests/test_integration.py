"""Integration tests against real endpoints."""

import os
import pytest

from src.llm.types import LLMInput, ToolDefinition, ToolResult, FinishReason, Message, Role
from src.llm.providers import resolve_provider, OllamaProvider, ClaudeProvider


# ---------------------------------------------------------------------------
# Provider wiring
# ---------------------------------------------------------------------------

def test_ollama_provider_resolution():
    """Ollama provider resolves correctly."""
    p = resolve_provider("ollama")
    assert isinstance(p, OllamaProvider)


def test_claude_provider_resolution():
    """Claude provider resolves correctly."""
    p = resolve_provider("claude")
    assert isinstance(p, ClaudeProvider)


def test_local_b60_resolution():
    """local-b60-8083 maps to OllamaProvider."""
    p = resolve_provider("local-b60-8083")
    assert isinstance(p, OllamaProvider)


# ---------------------------------------------------------------------------
# Live API calls (skipped when no key / endpoint unreachable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY",
)
def test_claude_live():
    """Claude provider returns a response."""
    from src.llm.config import ProviderConfig
    config = ProviderConfig(name="claude", api_key_env="ANTHROPIC_API_KEY", default_model="claude-haiku-4-5")
    p = ClaudeProvider(config)
    out = p.generate(LLMInput(
        model="claude-haiku-4-5",
        messages=[Message(role=Role.USER, content="Say 'hi'")],
        system_prompt="You are a concise assistant.",
    ))
    assert out.message.content, "empty response"
    assert out.model
    assert out.usage.prompt_tokens > 0


@pytest.mark.skipif(
    not os.environ.get("LLAMA_SERVER_URL"),
    reason="no LLAMA_SERVER_URL",
)
def test_ollama_live_b60():
    """llama-server on :8083 responds (OpenAI style)."""
    p = resolve_provider("local-b60-8083")
    out = p.generate(LLMInput(
        model="qwen3.8-27b",
        messages=[Message(role=Role.USER, content="Say 'hi'")],
    ))
    assert out.message.content, "empty response"
    assert out.model
