"""Retry logic with exponential backoff and circuit breaker."""

from __future__ import annotations

import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing if recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for provider calls."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0
    
    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count += 1
        elif self.state == CircuitState.CLOSED:
            self.success_count += 1
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def can_execute(self) -> bool:
        """Check if a call should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN: allow one test call


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """Decorator for retrying async functions with exponential backoff."""
    cfg = config or RetryConfig()
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(cfg.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except cfg.non_retryable_exceptions as e:
                    raise
                except cfg.retryable_exceptions as e:
                    last_exception = e
                    if attempt == cfg.max_retries:
                        break
                    
                    delay = min(
                        cfg.base_delay * (cfg.exponential_base ** attempt),
                        cfg.max_delay,
                    )
                    if cfg.jitter:
                        delay *= (0.5 + random.random())
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def retry_sync(config: Optional[RetryConfig] = None):
    """Decorator for retrying sync functions with exponential backoff."""
    cfg = config or RetryConfig()
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(cfg.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except cfg.non_retryable_exceptions as e:
                    raise
                except cfg.retryable_exceptions as e:
                    last_exception = e
                    if attempt == cfg.max_retries:
                        break
                    
                    delay = min(
                        cfg.base_delay * (cfg.exponential_base ** attempt),
                        cfg.max_delay,
                    )
                    if cfg.jitter:
                        delay *= (0.5 + random.random())
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class ProviderRetry:
    """Per-provider retry tracking with circuit breakers."""
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._retry_counts: dict[str, int] = {}
    
    def get_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider."""
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker()
        return self._breakers[provider]
    
    def record_success(self, provider: str) -> None:
        """Record a successful call to a provider."""
        breaker = self.get_breaker(provider)
        breaker.record_success()
    
    def record_failure(self, provider: str) -> None:
        """Record a failed call to a provider."""
        breaker = self.get_breaker(provider)
        breaker.record_failure()
        self._retry_counts[provider] = self._retry_counts.get(provider, 0) + 1
    
    def can_use(self, provider: str) -> bool:
        """Check if a provider is available."""
        breaker = self.get_breaker(provider)
        return breaker.can_execute()
    
    def get_stats(self) -> dict[str, dict]:
        """Get retry stats for all providers."""
        return {
            provider: {
                "state": breaker.state.value,
                "failures": breaker.failure_count,
                "successes": breaker.success_count,
            }
            for provider, breaker in self._breakers.items()
        }
