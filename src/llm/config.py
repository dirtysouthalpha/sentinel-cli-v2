"""Configuration management for Sentinel CLI V2."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProviderConfig:
    """Provider configuration."""
    name: str
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# Built-in fleet defaults — matches FLEET_PROVIDERS in fleet_router.py.
# Resolution order: YAML override > env vars > these defaults.
DEFAULT_PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "claude": ProviderConfig(
        name="claude", api_key=None, base_url=None, default_model="claude-sonnet-4-5",
    ),
    "openai": ProviderConfig(
        name="openai", api_key=None, base_url="https://api.openai.com/v1", default_model="gpt-4o",
    ),
    "openrouter": ProviderConfig(
        name="openrouter", api_key=None, base_url="https://openrouter.ai/api/v1", default_model="auto",
    ),
    "ollama": ProviderConfig(
        name="ollama", api_key="", base_url="http://localhost:11434", default_model="llama3.1",
    ),
    "local-b60-8083": ProviderConfig(
        name="local-b60-8083", api_key="", base_url="http://100.86.200.42:8083", default_model="qwen3.8-27b",
    ),
    "local-b60-swap": ProviderConfig(
        name="local-b60-swap", api_key="", base_url="http://100.70.240.55:9090/v1", default_model="gemma4-12b",
    ),
    "local-amd-rocm": ProviderConfig(
        name="local-amd-rocm", api_key="", base_url="http://localhost:11434", default_model="llama3.1",
    ),
    "echo": ProviderConfig(
        name="echo", api_key="", base_url=None, default_model="echo",
    ),
}


@dataclass
class SentinelConfig:
    """Main configuration."""
    default_provider: str = "claude"
    default_model: str = "claude-sonnet-4-5"
    max_iterations: int = 25
    enable_plugins: bool = True
    enable_cost_tracking: bool = True
    enable_compaction: bool = True
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    plugins: list[str] = field(default_factory=list)
    audit_dir: str = "~/.sentinel/audit"
    cache_dir: str = "~/.sentinel/cache"
    sessions_dir: str = "~/.sentinel/sessions"
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)

    def get_provider(self, name: str) -> ProviderConfig:
        """Get provider config: YAML override > env vars > built-in fleet defaults."""
        # 1. YAML config wins
        if name in self.providers:
            return self.providers[name]

        # 2. Env vars (dashes -> underscores: LOCAL_B60_8083_BASE_URL)
        env_prefix = name.upper().replace('-', '_')
        api_key = os.environ.get(f"{env_prefix}_API_KEY")
        base_url = os.environ.get(f"{env_prefix}_BASE_URL")

        # 3. Built-in fleet defaults — no env vars needed for local providers
        default = DEFAULT_PROVIDER_CONFIGS.get(name)
        if default is not None:
            return ProviderConfig(
                name=name,
                api_key=api_key or default.api_key,
                base_url=base_url or default.base_url,
                default_model=default.default_model,
                extra=dict(default.extra),
            )

        return ProviderConfig(name=name, api_key=api_key, base_url=base_url)

    @classmethod
    def load(cls, path: str | Path | None = None) -> SentinelConfig:
        """Load configuration from YAML file."""
        if path is None:
            path = Path.home() / ".sentinel" / "config.yaml"

        config_path = Path(path)
        if not config_path.exists():
            return cls()  # Return defaults

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        providers = {}
        for name, pdata in data.get("providers", {}).items():
            providers[name] = ProviderConfig(name=name, **pdata)

        return cls(
            default_provider=data.get("default_provider", "claude"),
            default_model=data.get("default_model", "claude-sonnet-4-5"),
            max_iterations=data.get("max_iterations", 25),
            enable_plugins=data.get("enable_plugins", True),
            enable_cost_tracking=data.get("enable_cost_tracking", True),
            enable_compaction=data.get("enable_compaction", True),
            providers=providers,
            plugins=data.get("plugins", []),
            audit_dir=data.get("audit_dir", "~/.sentinel/audit"),
            cache_dir=data.get("cache_dir", "~/.sentinel/cache"),
            sessions_dir=data.get("sessions_dir", "~/.sentinel/sessions"),
            mcp_servers=data.get("mcp_servers", []),
        )

    def save(self, path: str | Path | None = None) -> None:
        """Save configuration to YAML file."""
        if path is None:
            path = Path.home() / ".sentinel" / "config.yaml"

        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "max_iterations": self.max_iterations,
            "enable_plugins": self.enable_plugins,
            "enable_cost_tracking": self.enable_cost_tracking,
            "enable_compaction": self.enable_compaction,
            "providers": {
                name: {
                    "api_key": p.api_key,
                    "base_url": p.base_url,
                    "default_model": p.default_model,
                    "extra": p.extra,
                }
                for name, p in self.providers.items()
            },
            "plugins": self.plugins,
            "mcp_servers": self.mcp_servers,
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


@dataclass
class ModelPricing:
    """Pricing for a specific model."""
    prompt_cost_per_1k: float = 0.0
    completion_cost_per_1k: float = 0.0

    def compute(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute cost for a request."""
        return (
            (prompt_tokens / 1000) * self.prompt_cost_per_1k
            + (completion_tokens / 1000) * self.completion_cost_per_1k
        )


def get_pricing(model: str) -> ModelPricing:
    """Get pricing for a model."""
    pricing_map = {
        "claude-sonnet-4-5": ModelPricing(0.003, 0.015),
        "claude-opus-4-5": ModelPricing(0.015, 0.075),
        "claude-haiku-4": ModelPricing(0.00025, 0.00125),
        "gpt-4o": ModelPricing(0.005, 0.015),
        "gpt-4o-mini": ModelPricing(0.00015, 0.0006),
    }
    return pricing_map.get(model, ModelPricing(0.001, 0.002))
