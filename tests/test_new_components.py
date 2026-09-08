"""Tests for new Sentinel CLI V2 components."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.memory.cache import (
    SQLiteCache,
    BrainBackend,
    TieredMemory,
    MemoryEntry,
)
from src.llm.agent.grind import GrindMode, GrindState
from src.llm.config import SentinelConfig, ProviderConfig
from src.hooks.audit import AuditLog, AuditEntry


# === Memory Cache Tests ===

class TestSQLiteCache:
    def test_create_and_remember(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteCache(Path(tmpdir) / "test.db")
            cache.remember("test_key", "test content")
            results = cache.recall("test_key")
            assert len(results) == 1
            assert results[0].content == "test content"

    def test_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteCache(Path(tmpdir) / "test.db")
            cache.remember("key1", "hello world")
            cache.remember("key2", "goodbye world")
            results = cache.search("world")
            assert len(results) == 2

    def test_forget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteCache(Path(tmpdir) / "test.db")
            cache.remember("key1", "content")
            assert cache.forget("key1") is True
            assert cache.forget("key1") is False

    def test_expired_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteCache(Path(tmpdir) / "test.db")
            cache.remember("key1", "content", ttl=0.001)
            import time
            time.sleep(0.01)
            results = cache.recall("key1")
            assert len(results) == 0

    def test_cleanup_expired(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteCache(Path(tmpdir) / "test.db")
            cache.remember("key1", "content", ttl=0.001)
            cache.remember("key2", "content", ttl=3600)
            import time
            time.sleep(0.01)
            removed = cache.cleanup_expired()
            assert removed == 1


class TestTieredMemory:
    def test_local_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteCache(Path(tmpdir) / "test.db")
            brain = BrainBackend(base_url="http://localhost:99999")  # unreachable
            mem = TieredMemory(cache=cache, brain=brain)
            mem.remember("key1", "local content")
            results = mem.recall("key1")
            assert len(results) == 1
            assert results[0].content == "local content"


# === Grind Mode Tests ===

class TestGrindMode:
    def test_grind_result_success(self):
        from src.llm.agent import ReActAgent, AgentConfig
        from src.llm.agent.grind import GrindMode, GrindResult, GrindState
        from src.llm.providers.echo import EchoProvider
        from src.llm.tools import ToolExecutor, ToolRegistry

        config = AgentConfig(enable_cost_tracking=False, enable_compaction=False)
        agent = ReActAgent(config)
        mem = TieredMemory(
            cache=SQLiteCache(),
            brain=BrainBackend(base_url="http://localhost:99999"),
        )
        grind = GrindMode(agent=agent, memory=mem, human_gate=False, max_iterations=5)
        result = grind.grind("test goal")
        assert isinstance(result, GrindResult)
        assert result.goal == "test goal"
        assert result.state in (GrindState.DONE, GrindState.ERROR)


# === Config Tests ===

class TestSentinelConfig:
    def test_default_config(self):
        config = SentinelConfig()
        assert config.default_provider == "claude"
        assert config.default_model == "claude-sonnet-4-5"
        assert config.max_iterations == 25

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config = SentinelConfig()
            config.default_model = "claude-opus-4-5"
            config.save(path)

            loaded = SentinelConfig.load(path)
            assert loaded.default_model == "claude-opus-4-5"

    def test_get_provider_from_env(self):
        config = SentinelConfig()
        os.environ["TEST_API_KEY"] = "test_key_123"
        provider = config.get_provider("test")
        assert provider.api_key == "test_key_123"
        del os.environ["TEST_API_KEY"]


# === Audit Log Tests ===

class TestAuditLog:
    def test_log_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog(tmpdir)
            log.log("test_tool", {"arg": "value"}, "output", 150.0, 0.001, "claude")
            summary = log.summary()
            assert summary["total_calls"] == 1
            assert summary["total_cost"] == 0.001
            assert "test_tool" in summary["tools"]

    def test_multiple_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog(tmpdir)
            for i in range(5):
                log.log(f"tool_{i}", {}, f"out_{i}", 100.0 + i, 0.001 * i)
            summary = log.summary()
            assert summary["total_calls"] == 5
            assert summary["errors"] == 0
