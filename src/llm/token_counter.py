"""Real token counting using tiktoken (OpenAI) or character-based fallback."""

from __future__ import annotations

import re
from typing import Optional

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


# Model-specific encoding map
ENCODING_MAP = {
    "claude-sonnet-4-5": "o200k_base",
    "claude-haiku-4-5": "o200k_base",
    "claude-opus-4-5": "o200k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}

# Cache encodings
_encoding_cache: dict[str, any] = {}


def _get_encoding(model: str):
    """Get or create a tiktoken encoding for the model."""
    if not HAS_TIKTOKEN:
        return None
    
    # Find matching encoding
    encoding_name = "cl100k_base"  # default
    for prefix, enc in ENCODING_MAP.items():
        if model.startswith(prefix):
            encoding_name = enc
            break
    
    if encoding_name not in _encoding_cache:
        _encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    
    return _encoding_cache[encoding_name]


def count_tokens(text: str, model: str = "claude-sonnet-4-5") -> int:
    """Count tokens in text for a given model.
    
    Uses tiktoken if available, otherwise falls back to character-based estimate.
    """
    if not text:
        return 0
    
    encoding = _get_encoding(model)
    if encoding:
        return len(encoding.encode(text))
    
    # Fallback: ~4 chars per token for English text
    return len(text) // 4


def count_messages_tokens(messages: list[dict], model: str = "claude-sonnet-4-5") -> int:
    """Count total tokens in a list of messages.
    
    Each message has ~4 tokens of overhead (role, content markers).
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            # Multi-content (text + images)
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += count_tokens(part.get("text", ""), model)
        total += 4  # per-message overhead
    return total


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "claude-sonnet-4-5",
) -> float:
    """Estimate cost in USD for a model call.
    
    Prices are approximate and should be updated periodically.
    """
    # Price per 1M tokens (prompt, completion)
    PRICING = {
        "claude-sonnet-4-5": (3.0, 15.0),
        "claude-haiku-4-5": (0.25, 1.25),
        "claude-opus-4-5": (15.0, 75.0),
        "gpt-4o": (2.5, 10.0),
        "gpt-4o-mini": (0.15, 0.6),
        "gpt-4-turbo": (10.0, 30.0),
        "gpt-4": (30.0, 60.0),
        "gpt-3.5-turbo": (0.5, 1.5),
    }
    
    prompt_price, completion_price = (3.0, 15.0)  # default to sonnet
    for prefix, prices in PRICING.items():
        if model.startswith(prefix):
            prompt_price, completion_price = prices
            break
    
    return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000


def truncate_text(text: str, max_tokens: int, model: str = "claude-sonnet-4-5") -> str:
    """Truncate text to fit within max_tokens."""
    if count_tokens(text, model) <= max_tokens:
        return text
    
    # Binary search for the right length
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid], model) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    
    return text[:lo]


def truncate_messages(
    messages: list[dict],
    max_tokens: int,
    model: str = "claude-sonnet-4-5",
    keep_last: int = 4,
) -> list[dict]:
    """Truncate message list to fit within max_tokens.
    
    Keeps the last `keep_last` messages intact and truncates older ones.
    """
    if not messages:
        return messages
    
    # Always keep the last N messages
    if len(messages) <= keep_last:
        return messages
    
    head = messages[:-keep_last]
    tail = messages[-keep_last:]
    
    tail_tokens = count_messages_tokens(tail, model)
    remaining = max_tokens - tail_tokens
    
    if remaining <= 0:
        return tail
    
    # Truncate head messages from oldest to newest
    result = []
    for msg in head:
        msg_tokens = count_tokens(msg.get("content", ""), model) + 4
        if msg_tokens <= remaining:
            result.append(msg)
            remaining -= msg_tokens
        else:
            # Truncate this message's content
            content = msg.get("content", "")
            if isinstance(content, str):
                truncated = truncate_text(content, remaining, model)
                if truncated:
                    result.append({**msg, "content": truncated})
            break
    
    return result + tail
