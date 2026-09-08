"""Provider registry — resolve and instantiate providers."""

from __future__ import annotations

from ..config import ProviderConfig, SentinelConfig
from ..base import LLMProvider


def create_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    """Create a provider by name. None -> LLM_PROVIDER env var -> 'claude'."""
    import os

    if name is None:
        name = os.environ.get("LLM_PROVIDER", "claude")

    sentinel_config = SentinelConfig.load()
    config = sentinel_config.get_provider(name)

    if name == "claude":
        from .sentinel_claude import ClaudeProvider
        return ClaudeProvider(config, model)
    elif name in ("openai", "openrouter"):
        from .sentinel_openai import OpenAIProvider
        return OpenAIProvider(config, model)
    elif name in ("ollama", "local-b60-8083", "local-b60-swap", "local-amd-rocm"):
        from .sentinel_ollama import OllamaProvider
        return OllamaProvider(config, model)
    elif name == "echo":
        from .echo import EchoProvider
        return EchoProvider(config, model)
    else:
        raise ValueError(f"No provider implementation for '{name}'")


# Alias — resolve_provider is the public name used by agent.py
resolve_provider = create_provider


# Re-export provider classes for cleaner imports
from .sentinel_claude import ClaudeProvider
from .sentinel_ollama import OllamaProvider
from .sentinel_openai import OpenAIProvider
from .echo import EchoProvider


def list_providers() -> list[str]:
    """List available provider names."""
    return ["claude", "openai", "openrouter", "ollama", "local-b60-8083", "local-b60-swap", "local-amd-rocm", "echo"]
