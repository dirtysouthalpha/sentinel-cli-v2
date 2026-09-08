"""Config validation with helpful error messages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ValidationError:
    """A configuration problem."""
    field: str
    message: str
    severity: str = "error"  # error | warning
    suggestion: str = ""


@dataclass
class ValidationResult:
    """Result of config validation."""
    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.valid and not self.warnings:
            lines.append("✓ Configuration valid")
        else:
            for err in self.errors:
                lines.append(f"✗ ERROR [{err.field}]: {err.message}")
                if err.suggestion:
                    lines.append(f"  → {err.suggestion}")
            for warn in self.warnings:
                lines.append(f"⚠ WARN [{warn.field}]: {warn.message}")
                if warn.suggestion:
                    lines.append(f"  → {suggestion}")
        return "\n".join(lines)


class ConfigValidator:
    """Validates Sentinel CLI configuration."""
    
    # Known providers and their required env vars
    PROVIDER_REQUIREMENTS = {
        "claude": {
            "env": ["ANTHROPIC_API_KEY"],
            "models": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5"],
        },
        "openai": {
            "env": ["OPENAI_API_KEY"],
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        },
        "local-b60-8083": {
            "env": [],
            "models": ["qwen3.8-27b", "qwen2.5-coder-3b"],
            "url": "http://100.86.200.42:8083",
        },
        "local-b60-swap": {
            "env": [],
            "models": ["qwen2.5-coder-32b", "qwen2.5-coder-3b"],
            "url": "http://100.70.240.55:9090",
        },
        "local-amd-rocm": {
            "env": [],
            "models": ["llama3.1-70b", "llama3.2-3b"],
            "url": "http://100.86.200.42:11434",
        },
        "echo": {
            "env": [],
            "models": ["echo"],
        },
    }
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.result = ValidationResult()
    
    def validate(self) -> ValidationResult:
        """Run all validation checks."""
        self._validate_provider()
        self._validate_model()
        self._validate_api_keys()
        self._validate_budget()
        self._validate_paths()
        return self.result
    
    def _validate_provider(self) -> None:
        """Check provider is known."""
        provider = self.config.get("provider", "")
        if provider and provider not in self.PROVIDER_REQUIREMENTS:
            known = ", ".join(self.PROVIDER_REQUIREMENTS.keys())
            self.result.errors.append(ValidationError(
                field="provider",
                message=f"Unknown provider: {provider}",
                suggestion=f"Known providers: {known}",
            ))
            self.result.valid = False
    
    def _validate_model(self) -> None:
        """Check model is valid for the provider."""
        provider = self.config.get("provider", "")
        model = self.config.get("model", "")
        
        if not provider or not model:
            return
        
        reqs = self.PROVIDER_REQUIREMENTS.get(provider, {})
        known_models = reqs.get("models", [])
        if known_models and not any(model.startswith(m) or m in model for m in known_models):
            self.result.warnings.append(ValidationError(
                field="model",
                message=f"Model {model!r} not in known models for {provider}",
                suggestion=f"Known: {', '.join(known_models[:5])}",
            ))
    
    def _validate_api_keys(self) -> None:
        """Check required API keys are present."""
        provider = self.config.get("provider", "")
        reqs = self.PROVIDER_REQUIREMENTS.get(provider, {})
        
        for env_var in reqs.get("env", []):
            if not os.environ.get(env_var):
                self.result.errors.append(ValidationError(
                    field="api_key",
                    message=f"Missing environment variable: {env_var}",
                    suggestion=f"export {env_var}=sk-...",
                ))
                self.result.valid = False
    
    def _validate_budget(self) -> None:
        """Check budget settings."""
        budget = self.config.get("budget_usd")
        if budget is not None:
            try:
                b = float(budget)
                if b < 0:
                    raise ValueError
                if b == 0:
                    self.result.warnings.append(ValidationError(
                        field="budget_usd",
                        message="Budget is $0 — cloud providers will be skipped",
                        suggestion="Set a positive budget or use local providers only",
                    ))
            except (TypeError, ValueError):
                self.result.errors.append(ValidationError(
                    field="budget_usd",
                    message=f"Invalid budget: {budget!r}",
                    suggestion="Budget must be a positive number",
                ))
                self.result.valid = False
    
    def _validate_paths(self) -> None:
        """Check writable paths."""
        for path_key in ("session_dir", "log_dir", "cache_dir"):
            path_str = self.config.get(path_key)
            if path_str:
                try:
                    p = Path(path_str).expanduser()
                    p.mkdir(parents=True, exist_ok=True)
                    if not os.access(p, os.W_OK):
                        raise PermissionError
                except (OSError, PermissionError) as e:
                    self.result.errors.append(ValidationError(
                        field=path_key,
                        message=f"Cannot write to {path_str}: {e}",
                        suggestion="Check permissions or use a different path",
                    ))
                    self.result.valid = False


def validate_config(config: dict[str, Any]) -> ValidationResult:
    """Validate a config dict."""
    return ConfigValidator(config).validate()
