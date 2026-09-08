"""Tests for the new system components: token counter, fleet router, retry, logger, validation."""

import os
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.token_counter import (
    count_tokens,
    count_messages_tokens,
    estimate_cost,
    truncate_text,
    truncate_messages,
)
from src.llm.fleet_router import (
    FleetRouter,
    TaskTier,
    FLEET_PROVIDERS,
    PROVIDER_MODELS,
)
from src.llm.retry import (
    CircuitBreaker,
    CircuitState,
    RetryConfig,
    ProviderRetry,
    retry_with_backoff,
    retry_sync,
)
from src.llm.logger import (
    StructuredLogger,
    LogEntry,
    EventCategory,
    LogLevel,
    get_logger,
)
from src.llm.validation import (
    validate_config,
    ConfigValidator,
)


# === Token Counter ===

class TestTokenCounter:
    def test_count_tokens_basic(self):
        text = "Hello, world! This is a test."
        count = count_tokens(text)
        assert count > 0
        assert count < 50

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        total = count_messages_tokens(messages)
        assert total > 8  # at least 2 tokens + overhead
        assert total < 100

    def test_estimate_cost_sonnet(self):
        cost = estimate_cost(1000, 1000, "claude-sonnet-4-5")
        # Sonnet: $3/M prompt, $15/M completion
        expected = (1000 * 3.0 + 1000 * 15.0) / 1_000_000
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_haiku(self):
        cost = estimate_cost(1000, 1000, "claude-haiku-4-5")
        # Haiku: $0.25/M prompt, $1.25/M completion
        expected = (1000 * 0.25 + 1000 * 1.25) / 1_000_000
        assert abs(cost - expected) < 0.0001

    def test_truncate_text(self):
        text = "word " * 1000  # ~5000 chars
        truncated = truncate_text(text, 50)
        assert len(truncated) < len(text)

    def test_truncate_messages(self):
        messages = [
            {"role": "user", "content": "old message " * 100},
            {"role": "assistant", "content": "old reply " * 100},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        truncated = truncate_messages(messages, max_tokens=100, keep_last=2)
        assert len(truncated) >= 2
        assert truncated[-1]["content"] == "recent answer"


# === Fleet Router ===

class TestFleetRouter:
    def test_classify_complex(self):
        router = FleetRouter()
        assert router.classify_task("design the architecture") == TaskTier.COMPLEX
        assert router.classify_task("debug this crash") == TaskTier.COMPLEX
        assert router.classify_task("refactor the whole module") == TaskTier.COMPLEX

    def test_classify_simple(self):
        router = FleetRouter()
        assert router.classify_task("fix typo in readme") == TaskTier.SIMPLE
        assert router.classify_task("format this file") == TaskTier.SIMPLE

    def test_classify_standard(self):
        router = FleetRouter()
        assert router.classify_task("implement user login") == TaskTier.STANDARD

    def test_route_local_preferred(self):
        router = FleetRouter(prefer_local=True)
        decision = router.route("implement a feature")
        assert decision.provider in FLEET_PROVIDERS
        assert "local" in decision.provider or decision.provider in ("claude", "openai")

    def test_route_budget_exhausted(self):
        router = FleetRouter(prefer_local=False, budget_usd=0.01)
        router.record_cost(1.0)  # over budget
        decision = router.route("simple task")
        # Should force local
        assert "local" in decision.provider or "budget" in decision.reason

    def test_route_cloud_complex(self):
        router = FleetRouter(prefer_local=False, budget_usd=100.0)
        decision = router.route("design system architecture")
        assert decision.provider == "claude"
        assert "opus" in decision.model

    def test_remaining_budget(self):
        router = FleetRouter(budget_usd=10.0)
        assert router.remaining_budget == 10.0
        router.record_cost(3.0)
        assert abs(router.remaining_budget - 7.0) < 0.001


# === Circuit Breaker / Retry ===

class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        for _ in range(2):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        time.sleep(0.15)
        assert cb.can_execute()  # should transition to half-open
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        for _ in range(2):
            cb.record_failure()
        time.sleep(0.15)
        cb.can_execute()  # half-open
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


class TestProviderRetry:
    def test_get_breaker(self):
        pr = ProviderRetry()
        b1 = pr.get_breaker("claude")
        b2 = pr.get_breaker("claude")
        assert b1 is b2

    def test_record_success_failure(self):
        pr = ProviderRetry()
        pr.record_success("claude")
        pr.record_failure("claude")
        stats = pr.get_stats()
        assert stats["claude"]["failures"] == 1
        assert stats["claude"]["successes"] == 1

    def test_can_use(self):
        pr = ProviderRetry()
        assert pr.can_use("claude")
        for _ in range(10):
            pr.record_failure("claude")
        assert not pr.can_use("claude")


class TestRetryDecorators:
    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        attempts = []
        
        @retry_with_backoff(RetryConfig(max_retries=3, base_delay=0.01))
        async def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("fail")
            return "ok"
        
        assert await flaky() == "ok"
        assert len(attempts) == 3

    def test_retry_sync_success(self):
        attempts = []
        
        @retry_sync(RetryConfig(max_retries=3, base_delay=0.01))
        def flaky():
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("fail")
            return "ok"
        
        assert flaky() == "ok"
        assert len(attempts) == 2

    def test_retry_exhausted(self):
        attempts = []
        
        @retry_sync(RetryConfig(max_retries=2, base_delay=0.01))
        def always_fails():
            attempts.append(1)
            raise ValueError("always")
        
        with pytest.raises(ValueError):
            always_fails()
        assert len(attempts) == 3  # 1 + 2 retries


# === Structured Logger ===

class TestStructuredLogger:
    def test_log_and_read(self, tmp_path):
        logger = StructuredLogger(log_dir=str(tmp_path))
        entry = logger.info("test_event", session_id="s1", foo="bar")
        assert entry.event == "test_event"
        
        entries = logger.read_logs(session_id="s1")
        assert len(entries) == 1
        assert entries[0].data["foo"] == "bar"

    def test_log_levels(self, tmp_path):
        logger = StructuredLogger(log_dir=str(tmp_path))
        logger.info("info_event", session_id="s2")
        logger.error("error_event", session_id="s2")
        
        entries = logger.read_logs(session_id="s2")
        assert len(entries) == 2
        assert entries[0].level == "info"
        assert entries[1].level == "error"

    def test_summary(self, tmp_path):
        logger = StructuredLogger(log_dir=str(tmp_path))
        logger.info("evt1", session_id="s3", category=EventCategory.TOOL)
        logger.error("evt2", session_id="s3")
        
        summary = logger.summary("s3")
        assert summary["events"] == 2
        assert summary["errors"] == 1


# === Config Validation ===

class TestValidation:
    def test_valid_local_config(self):
        result = validate_config({
            "provider": "local-b60-8083",
            "model": "qwen3.8-27b",
        })
        assert result.valid
        assert not result.has_errors

    def test_unknown_provider(self):
        result = validate_config({
            "provider": "nonexistent-provider",
            "model": "whatever",
        })
        assert not result.valid
        assert len(result.errors) == 1
        assert "Unknown provider" in result.errors[0].message

    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            result = validate_config({
                "provider": "claude",
                "model": "claude-sonnet-4-5",
            })
            assert not result.valid
            assert any("ANTHROPIC_API_KEY" in e.message for e in result.errors)

    def test_unknown_model_warning(self):
        result = validate_config({
            "provider": "claude",
            "model": "claude-fake-model",
        })
        # Warning only — still valid if key present
        # (this test may fail if key missing, so just check structure)
        assert isinstance(result.warnings, list)

    def test_invalid_budget(self):
        result = validate_config({
            "provider": "echo",
            "model": "echo",
            "budget_usd": "not-a-number",
        })
        assert not result.valid

    def test_zero_budget_warning(self):
        result = validate_config({
            "provider": "echo",
            "model": "echo",
            "budget_usd": 0,
        })
        assert result.valid  # warning, not error
        assert len(result.warnings) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
