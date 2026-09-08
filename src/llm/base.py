"""Base provider class.

Frozen — one interface, any backend.
Pattern stolen from ECC's src/llm/providers/.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from .config import ProviderConfig, ModelPricing, get_pricing
from .types import LLMInput, LLMOutput, Message, UsageCost, FinishReason, ToolDefinition


class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, config: ProviderConfig, model: str | None = None):
        self.config = config
        self.model = model or config.default_model
        self._pricing = get_pricing(self.model)

    @abstractmethod
    def generate(self, input: [[PPI-57]]) -> LLMOutput:
        """Generate a response. Synchronous."""
        ...

    @abstractmethod
    async def agenerate(self, input: [[PPI-57]]) -> LLMOutput:
        """Generate a response. Async."""
        ...

    @abstractmethod
    def validate(self) -> bool:
        """Check that the provider is properly configured."""
        ...

    @abstractmethod
    def resolve_model(self, model: str | None = None) -> str:
        """Resolve a model name to a concrete model."""
        ...

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute USD cost for a call."""
        return self._pricing.compute(prompt_tokens, completion_tokens)

    def usage(self, prompt_tokens: int, completion_tokens: int) -> UsageCost:
        """Build a UsageCost object."""
        return UsageCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=self.compute_cost(prompt_tokens, completion_tokens),
            provider=self.config.name,
            model=self.model,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model}>"


def _extract_text(message: Message) -> str:
    """Extract text content from a message."""
    return message.content


def _build_system_prompt(system: str | None, tools: tuple[ToolDefinition, ...]) -> str:
    """Build a system prompt with tool descriptions."""
    parts: list[str] = []
    if system:
        parts.append(system)
    if tools:
        parts.append("\n\nYou have access to the following tools:\n")
        for t in tools:
            parts.append(f"  - {t.name}: {t.description}")
    return "\n".join(parts)
