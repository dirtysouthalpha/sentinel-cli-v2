"""Fleet provider routing — picks the best model/host for each task.

Routes between NUKE's B60, homeserver's llama-swap, ollama, and cloud providers
based on task complexity, current load, and cost.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskTier(Enum):
    """Task complexity tiers."""
    SIMPLE = "simple"       # Boilerplate, formatting, simple queries
    STANDARD = "standard"   # Implementation, refactoring
    COMPLEX = "complex"     # Architecture, debugging, deep analysis


@dataclass
class ProviderCapabilities:
    """What a provider can do."""
    name: str
    host: str
    port: int
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    max_context: int = 128_000
    is_local: bool = False
    cost_per_1m: float = 0.0  # 0 = free (local)
    latency_ms: int = 0
    current_load: float = 0.0  # 0-1 scale


@dataclass
class RoutingDecision:
    """Result of routing a task to a provider."""
    provider: str
    model: str
    host: str
    port: int
    reason: str
    estimated_cost: float = 0.0


# Fleet provider registry — matches your actual infrastructure
FLEET_PROVIDERS: dict[str, ProviderCapabilities] = {
    "local-b60-8083": ProviderCapabilities(
        name="local-b60-8083",
        host="100.86.200.42",
        port=8083,
        supports_tools=True,
        supports_vision=False,
        max_context=48_000,
        is_local=True,
        cost_per_1m=0.0,
        latency_ms=50,
    ),
    "local-b60-swap": ProviderCapabilities(
        name="local-b60-swap",
        host="100.70.240.55",
        port=9090,
        supports_tools=True,
        supports_vision=False,
        max_context=32_000,
        is_local=True,
        cost_per_1m=0.0,
        latency_ms=100,
    ),
    "local-amd-rocm": ProviderCapabilities(
        name="local-amd-rocm",
        host="100.86.200.42",
        port=11434,
        supports_tools=True,
        supports_vision=True,
        max_context=128_000,
        is_local=True,
        cost_per_1m=0.0,
        latency_ms=30,
    ),
    "claude": ProviderCapabilities(
        name="claude",
        host="api.anthropic.com",
        port=443,
        supports_tools=True,
        supports_vision=True,
        max_context=200_000,
        is_local=False,
        cost_per_1m=15.0,
        latency_ms=500,
    ),
    "openai": ProviderCapabilities(
        name="openai",
        host="api.openai.com",
        port=443,
        supports_tools=True,
        supports_vision=True,
        max_context=128_000,
        is_local=False,
        cost_per_1m=10.0,
        latency_ms=400,
    ),
}


# Model mapping per provider
PROVIDER_MODELS: dict[str, dict[TaskTier, str]] = {
    "local-b60-8083": {
        TaskTier.SIMPLE: "qwen2.5-coder-3b",
        TaskTier.STANDARD: "qwen3.8-27b",
        TaskTier.COMPLEX: "qwen3.8-27b",
    },
    "local-b60-swap": {
        TaskTier.SIMPLE: "qwen2.5-coder-3b",
        TaskTier.STANDARD: "qwen2.5-coder-32b",
        TaskTier.COMPLEX: "qwen2.5-coder-32b",
    },
    "local-amd-rocm": {
        TaskTier.SIMPLE: "llama3.2-3b",
        TaskTier.STANDARD: "llama3.1-70b",
        TaskTier.COMPLEX: "llama3.1-70b",
    },
    "claude": {
        TaskTier.SIMPLE: "claude-haiku-4-5",
        TaskTier.STANDARD: "claude-sonnet-4-5",
        TaskTier.COMPLEX: "claude-opus-4-5",
    },
    "openai": {
        TaskTier.SIMPLE: "gpt-4o-mini",
        TaskTier.STANDARD: "gpt-4o",
        TaskTier.COMPLEX: "gpt-4o",
    },
}


class FleetRouter:
    """Routes tasks to the best provider based on complexity, load, and cost."""
    
    def __init__(self, prefer_local: bool = True, budget_usd: float = 1.0):
        self.prefer_local = prefer_local
        self.budget_usd = budget_usd
        self._spent = 0.0
        self._provider_health: dict[str, bool] = {}
    
    def classify_task(self, task: str) -> TaskTier:
        """Classify a task into a complexity tier."""
        task_lower = task.lower()
        
        # Complex indicators
        complex_keywords = [
            "architecture", "design", "refactor", "debug", "optimize",
            "security", "performance", "scale", "migrate", "review",
            "analyze", "investigate", "diagnose", "complex"
        ]
        for kw in complex_keywords:
            if kw in task_lower:
                return TaskTier.COMPLEX
        
        # Simple indicators
        simple_keywords = [
            "format", "lint", "comment", "rename", "typo", "fix typo",
            "simple", "boilerplate", "template", "hello world"
        ]
        for kw in simple_keywords:
            if kw in task_lower:
                return TaskTier.SIMPLE
        
        return TaskTier.STANDARD
    
    def route(self, task: str, require_tools: bool = True) -> RoutingDecision:
        """Route a task to the best provider."""
        tier = self.classify_task(task)
        
        # If budget exhausted, force local
        if self._spent >= self.budget_usd:
            return self._route_local(tier, "budget exhausted")
        
        # If local preferred and available, use it
        if self.prefer_local:
            local = self._route_local(tier)
            if local:
                return local
        
        # Fall back to cloud based on tier
        if tier == TaskTier.COMPLEX:
            # Complex tasks get Claude Opus
            return RoutingDecision(
                provider="claude",
                model=PROVIDER_MODELS["claude"][tier],
                host=FLEET_PROVIDERS["claude"].host,
                port=FLEET_PROVIDERS["claude"].port,
                reason=f"complex task → Claude {PROVIDER_MODELS['claude'][tier]}",
            )
        else:
            # Standard/simple tasks get Sonnet or GPT-4o
            return RoutingDecision(
                provider="claude",
                model=PROVIDER_MODELS["claude"][tier],
                host=FLEET_PROVIDERS["claude"].host,
                port=FLEET_PROVIDERS["claude"].port,
                reason=f"{tier.value} task → Claude {PROVIDER_MODELS['claude'][tier]}",
            )
    
    def _route_local(self, tier: TaskTier, reason: str = "local preferred") -> Optional[RoutingDecision]:
        """Try to route to a local provider."""
        # Priority: B60 > ollama > llama-swap
        for name in ["local-b60-8083", "local-amd-rocm", "local-b60-swap"]:
            provider = FLEET_PROVIDERS.get(name)
            if not provider:
                continue
            if name not in PROVIDER_MODELS:
                continue
            
            model = PROVIDER_MODELS[name].get(tier)
            if not model:
                continue
            
            return RoutingDecision(
                provider=name,
                model=model,
                host=provider.host,
                port=provider.port,
                reason=f"{reason} → {name}/{model}",
            )
        return None
    
    def record_cost(self, cost_usd: float) -> None:
        """Record cost spent."""
        self._spent += cost_usd
    
    @property
    def remaining_budget(self) -> float:
        return max(0, self.budget_usd - self._spent)
