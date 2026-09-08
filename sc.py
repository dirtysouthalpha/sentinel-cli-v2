#!/usr/bin/env python3
"""Sentinel CLI V2 — entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.agent import ReActAgent
from src.llm.plugins import PluginManager, load_builtin_plugins
from src.llm.router import ModelRouter
from src.hooks.cost_tracker import CostTracker
from src.hooks.context_monitor import ContextMonitor, StrategicCompact
from src.llm.memory import NeuralisMemory
from src.llm.session import SessionStore
from src.llm.types import Message, Role


def build_registry() -> None:
    """Legacy — plugins handle tool registration now."""
    pass


def cmd_sessions(args):
    """List sessions."""
    store = SessionStore()
    sessions = store.list_sessions()
    if not sessions:
        print("No sessions found.")
        return
    print(f"{'ID':<12} {'Messages':>8} {'Cost':>10} {'Updated':<20}")
    print("-" * 55)
    for s in sessions:
        print(f"{s['session_id']:<12} {s['messages']:>8} ${s['total_cost']:>8.4f} {s['updated_at'][:19]:<20}")


def cmd_plugins(args):
    """List loaded plugins and their tools."""
    pm = PluginManager()
    for name, plugin in load_builtin_plugins().items():
        pm._plugins[name] = plugin
    pm.load_all()
    
    print(f"{'Plugin':<20} {'Version':<10} {'Tools':<30}")
    print("-" * 65)
    for name, plugin in pm.plugins.items():
        tools = ", ".join(t.name for t in plugin.tools)
        print(f"{name:<20} {plugin.version:<10} {tools:<30}")


def cmd_costs(args):
    """Show cost summary."""
    tracker = CostTracker()
    summary = tracker.summarize()
    print(f"Total cost: ${summary['total']:.4f}")
    print(f"Tokens: {summary['input']:,} in / {summary['output']:,} out")
    if summary['by_model']:
        print("\nBy model:")
        for model, cost in sorted(summary['by_model'].items(), key=lambda x: -x[1]):
            print(f"  {model}: ${cost:.4f}")


def cmd_brain(args):
    """Brain commands — check health, search, recall."""
    mem = NeuralisMemory()
    if args.brain_command == "health":
        try:
            import httpx
            resp = httpx.get(f"{mem.base_url}/health", timeout=5)
            print(f"Brain status: {resp.status_code} ({resp.json().get('status', 'unknown')})")
        except Exception as e:
            print(f"Brain unreachable: {e}")
    elif args.brain_command == "search":
        results = mem.search(args.query or "sentinel cli")
        for r in results[:5]:
            print(f"  - {r.get('content', r)[:80]}...")
    elif args.brain_command == "stats":
        try:
            import httpx
            resp = httpx.get(f"{mem.base_url}/stats", timeout=5)
            stats = resp.json()
            for k, v in stats.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"Failed: {e}")


def main():
    # Manual command parsing — argparse subparsers are greedy and steal
    # the first positional, so we handle commands ourselves.
    argv = sys.argv[1:]
    
    # Known subcommands
    if argv and argv[0] in ("sessions", "plugins", "costs", "brain"):
        cmd = argv[0]
        rest = argv[1:]
        
        if cmd == "sessions":
            return cmd_sessions(argparse.Namespace())
        elif cmd == "plugins":
            return cmd_plugins(argparse.Namespace())
        elif cmd == "costs":
            return cmd_costs(argparse.Namespace())
        elif cmd == "brain":
            bp = argparse.ArgumentParser()
            bp.add_argument("brain_command", choices=["health", "search", "stats"])
            bp.add_argument("query", nargs="?")
            args = bp.parse_args(rest)
            return cmd_brain(args)
    
    # Main prompt mode
    parser = argparse.ArgumentParser(description="Sentinel CLI V2")
    parser.add_argument("prompt", nargs="?", help="Initial prompt")
    parser.add_argument("--model", default=None, help="Model to use")
    parser.add_argument("--provider", default=None, help="Provider override")
    parser.add_argument("--session", default=None, help="Resume a session")
    parser.add_argument("--route", action="store_true", help="Auto-route task to appropriate model")
    parser.add_argument("--system", default="You are Sentinel CLI, a coding assistant.", help="System prompt")
    
    args = parser.parse_args(argv)

    # Build agent
    from src.llm.agent import AgentConfig

    model = args.model or "claude-sonnet-4-5"
    if args.route and args.prompt:
        router = ModelRouter()
        model, budget = router.route(args.prompt)
        print(f"↳ Routed to {model} (budget ${budget})")

    # Fail fast with a clean message if the provider has no credentials
    # instead of an SDK traceback deep inside generate().
    import os as _os
    _provider_name = args.provider or _os.environ.get("LLM_PROVIDER", "claude")
    if _provider_name == "claude" and not _os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: no ANTHROPIC_API_KEY set. Export it or use --provider local-b60-8083 / echo.")
        return 1
    
    config = AgentConfig(
        model=model,
        provider_override=args.provider,
        session_id=args.session,
        system_prompt=args.system,
    )
    agent = ReActAgent(config)

    # Interactive or single-shot
    try:
        if args.prompt:
            result = agent.run(args.prompt)
            print(result)
        else:
            print("Sentinel CLI V2 — interactive mode (Ctrl+C to exit)")
            print(f"Session: {agent.session.session_id}")
            print(f"Model: {model}")
            print(f"Plugins: {list(agent.plugin_manager.plugins.keys()) if agent.plugin_manager else 'none'}")
            print()
            while True:
                try:
                    user_input = input("> ")
                    if user_input.startswith("/"):
                        if user_input == "/quit":
                            break
                        elif user_input == "/costs":
                            summary = agent.cost_tracker.summarize() if agent.cost_tracker else {"total": 0}
                            print(f"Session cost: ${summary['total']:.4f}")
                            continue
                        elif user_input == "/session":
                            print(f"Session ID: {agent.session.session_id}")
                            print(f"Messages: {len(agent.session.messages)}")
                            print(f"Tool calls: {agent.session.state.tool_call_count}")
                            continue
                        elif user_input == "/help":
                            print("Commands: /quit, /costs, /session, /help")
                            continue
                    result = agent.run(user_input)
                    print(result)
                except (KeyboardInterrupt, EOFError):
                    print("\n👋 Session saved.")
                    break
    except KeyboardInterrupt:
        print("\n👋 Session saved.")


if __name__ == "__main__":
    main()
