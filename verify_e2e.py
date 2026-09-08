#!/usr/bin/env python3
"""End-to-end verification of Sentinel CLI V2 — exercises every subsystem."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = []
FAIL = []


def check(name: str, fn):
    """Run a check, record result."""
    start = time.time()
    try:
        result = fn()
        elapsed = (time.time() - start) * 1000
        PASS.append((name, elapsed, result))
        print(f"  ✓ {name} ({elapsed:.0f}ms)")
        return result
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        FAIL.append((name, str(e)))
        print(f"  ✗ {name}: {e}")
        return None


def main():
    print("=" * 60)
    print("Sentinel CLI V2 — End-to-End Verification")
    print("=" * 60)

    # === 1. Token Counter ===
    print("\n[1/8] Token Counter")
    from src.llm.token_counter import count_tokens, estimate_cost, truncate_messages

    def test_tokens():
        n = count_tokens("Hello, world! " * 100)
        assert n > 100, f"token count too low: {n}"
        return f"{n} tokens counted"

    check("count_tokens", test_tokens)
    check("estimate_cost", lambda: f"${estimate_cost(100000, 50000, 'claude-sonnet-4-5'):.4f}")

    # === 2. Fleet Router ===
    print("\n[2/8] Fleet Router")
    from src.llm.fleet_router import FleetRouter, TaskTier

    def test_router():
        router = FleetRouter()
        assert router.classify_task("refactor architecture") == TaskTier.COMPLEX
        assert router.classify_task("fix typo") == TaskTier.SIMPLE
        decision = router.route("implement feature")
        return f"routed → {decision.provider}/{decision.model}"

    check("classify + route", test_router)

    # === 3. Retry / Circuit Breaker ===
    print("\n[3/8] Retry + Circuit Breaker")
    from src.llm.retry import CircuitBreaker, CircuitState, ProviderRetry

    def test_breaker():
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        return "breaker opens at 3 failures"

    check("circuit breaker", test_breaker)

    # === 4. Structured Logger ===
    print("\n[4/8] Structured Logger")
    from src.llm.logger import get_logger, EventCategory

    def test_logger():
        logger = get_logger()
        logger.info("e2e_test", category=EventCategory.AGENT, session_id="e2e", value=42)
        entries = logger.read_logs(session_id="e2e")
        assert len(entries) >= 1
        return f"{len(entries)} entries logged"

    check("structured logger", test_logger)

    # === 5. Health Checker (LIVE fleet) ===
    print("\n[5/8] Fleet Health (LIVE)")
    from src.llm.health import FleetHealthChecker

    checker = FleetHealthChecker(timeout=5.0)
    summary = check("fleet health probe", lambda: checker.summary())
    if summary:
        for name, status in summary["providers"].items():
            state = "✓" if status["healthy"] else "✗"
            model = status.get("model", "?")
            latency = status.get("latency_ms", 0)
            print(f"      {state} {name}: {model} ({latency:.0f}ms)")

    # === 6. Agent Loop (echo provider, no API needed) ===
    print("\n[6/8] Agent Loop (echo provider)")
    from src.llm.agent import ReActAgent, AgentConfig

    def test_agent():
        config = AgentConfig(
            provider_override="echo",
            enable_cost_tracking=False,
            enable_compaction=False,
        )
        agent = ReActAgent(config)
        output = agent.run("test message for e2e")
        assert output, "agent returned empty"
        return f"loop ran, {len(agent.session.messages)} messages"

    check("agent ReAct loop", test_agent)

    # === 7. Hashline Edits ===
    print("\n[7/8] Hashline Edit System")
    from src.llm.hashline import parse_patch, apply_patch

    TEST_PATCH = """*** Begin Patch
PUT /tmp/sentinel_e2e_test.py
-old_line
+new_line
*** End Patch"""

    def test_hashline():
        Path("/tmp/sentinel_e2e_test.py").write_text("old_line\n")
        hunks = parse_patch(TEST_PATCH)
        assert len(hunks) > 0, "no hunks parsed"
        results = apply_patch(hunks)
        assert all(r.success for r in results), [r.error for r in results]
        content = Path("/tmp/sentinel_e2e_test.py").read_text()
        assert "new_line" in content, content
        return f"patch applied: {content.strip()[:40]}"

    check("hashline patch", test_hashline)

    # === 8. Full Test Suite ===
    print("\n[8/8] Full Test Suite")
    import subprocess

    def test_suite():
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
            capture_output=True,
            timeout=60,
        )
        output = result.stdout.decode()
        passed_line = [l for l in output.split("\n") if "passed" in l]
        assert result.returncode == 0, f"pytest failed: {passed_line}"
        return passed_line[0].strip() if passed_line else "all passed"

    check("pytest full suite", test_suite)

    # === Summary ===
    print("\n" + "=" * 60)
    print(f"RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nFailures:")
        for name, err in FAIL:
            print(f"  ✗ {name}: {err}")
    print("=" * 60)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
