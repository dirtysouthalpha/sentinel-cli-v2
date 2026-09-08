"""Streaming REPL with history, completion, and real-time output.

Refactored to avoid circular imports — agent components imported lazily.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.styles import Style
    from prompt_toolkit.formatted_text import HTML
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

from ..llm.agent import ReActAgent, AgentConfig
from ..llm.agent.grind import GrindMode
from ..llm.memory.cache import TieredMemory
from ..llm.session import SessionStore


# Cyberpunk-inspired style
STYLE = Style.from_dict({
    'prompt': '#00ffaa bold',
    'command': '#00ccff',
    'error': '#ff3366',
    'success': '#00ff88',
    'info': '#888888',
    'header': '#ffaa00 bold',
}) if HAS_PROMPT_TOOLKIT else None


class SentinelREPL:
    """Interactive REPL for Sentinel CLI."""

    COMMANDS = [
        '/help', '/quit', '/exit', '/model', '/provider',
        '/route', '/grind', '/sessions', '/costs', '/plugins',
        '/brain', '/clear', '/history', '/stats',
    ]

    def __init__(self, agent: ReActAgent | None = None):
        self.agent = agent or ReActAgent(AgentConfig())
        self.memory = TieredMemory()
        self.grind = GrindMode(self.agent, self.memory)
        self.store = SessionStore()
        self._running = True
        if HAS_PROMPT_TOOLKIT:
            self.session = PromptSession(
                history=FileHistory(str(Path.home() / ".sentinel" / "history")),
                completer=WordCompleter(self.COMMANDS, ignore_case=True),
                style=STYLE,
            )

    def run(self) -> None:
        """Main REPL loop."""
        self._print_header()

        while self._running:
            try:
                if HAS_PROMPT_TOOLKIT:
                    text = self.session.prompt(HTML('<prompt>sentinel&gt; </prompt>'))
                else:
                    text = input("sentinel> ")
                text = text.strip()

                if not text:
                    continue
                if text.startswith('/'):
                    self._handle_command(text)
                else:
                    self._handle_chat(text)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break
            except Exception as e:
                self._print(f'<error>Error: {e}</error>')

    def _print(self, text: str) -> None:
        """Print with optional HTML formatting."""
        if HAS_PROMPT_TOOLKIT:
            print_formatted_text(HTML(text), style=STYLE)
        else:
            # Strip HTML tags for plain terminal
            import re
            print(re.sub(r'<[^>]+>', '', text))

    def _print_header(self) -> None:
        self._print('<header>Sentinel CLI V2</header>')
        self._print('<info>Type /help for commands, or just start chatting.</info>')

    def _handle_command(self, cmd: str) -> None:
        """Handle a slash command."""
        parts = cmd.split(maxsplit=1)
        name = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if name in ('/quit', '/exit'):
            self._running = False
        elif name == '/help':
            self._print_help()
        elif name == '/model':
            if arg:
                self.agent.config.model = arg
                self._print(f'<success>Model set to {arg}</success>')
            else:
                self._print(f'<info>Model: {self.agent.config.model}</info>')
        elif name == '/provider':
            if arg:
                try:
                    from ..llm.providers import resolve_provider
                    self.agent.provider = resolve_provider(arg)
                    self._print(f'<success>Provider set to {arg}</success>')
                except Exception as e:
                    self._print(f'<error>Provider error: {e}</error>')
            else:
                self._print(f'<info>Provider: {type(self.agent.provider).__name__}</info>')
        elif name == '/route':
            if arg:
                from ..llm.router import route_task
                model = route_task(arg)
                self._print(f'<info>Route: {arg!r} → {model}</info>')
        elif name == '/grind':
            if arg:
                self._print('<info>Grinding...</info>')
                result = self.grind.grind(arg)
                status = '✓' if result.success else '✗'
                self._print(f'<{ "success" if result.success else "error"}>{status} {result.state.value}</{"success" if result.success else "error"}>')
                if result.output:
                    print(result.output[:2000])
            else:
                self._print('<info>Usage: /grind <goal></info>')
        elif name == '/sessions':
            sessions = self.store.list_sessions()
            if sessions:
                for s in sessions[:10]:
                    self._print(f"<info>{s['session_id'][:8]} — {s['messages']} msgs, ${s['total_cost']:.4f}</info>")
            else:
                self._print('<info>No sessions.</info>')
        elif name == '/costs':
            usage = self.agent.session.total_usage
            self._print(f'<info>Prompt: {usage.prompt_tokens:,} tokens</info>')
            self._print(f'<info>Completion: {usage.completion_tokens:,} tokens</info>')
            self._print(f'<info>Total cost: ${usage.cost_usd:.4f}</info>')
        elif name == '/plugins':
            if self.agent.plugin_manager:
                plugins = list(self.agent.plugin_manager._plugins.keys())
                self._print(f'<info>Loaded: {", ".join(plugins)}</info>')
        elif name == '/brain':
            if arg:
                results = self.memory.search(arg, limit=5)
                for r in results:
                    self._print(f'<info>• {r}</info>')
            else:
                self._print('<info>Usage: /brain <query></info>')
        elif name == '/clear':
            self.agent.session.state.messages.clear()
            self._print('<success>Session cleared.</success>')
        elif name == '/history':
            for m in self.agent.session.messages[-10:]:
                role = m.role.value if hasattr(m.role, 'value') else str(m.role)
                self._print(f'<info>{role}: {m.content[:100]}</info>')
        elif name == '/stats':
            stats = {
                'session': self.agent.session.session_id,
                'messages': len(self.agent.session.messages),
                'tool_calls': self.agent.session.state.tool_call_count,
                'iterations': self.agent.session.state.iteration_count,
                'total_tokens': self.agent.session.total_usage.total_tokens,
                'cost': f"${self.agent.session.total_usage.cost_usd:.4f}",
            }
            for k, v in stats.items():
                self._print(f'<info>{k}: {v}</info>')

    def _print_help(self) -> None:
        self._print('<header>Commands:</header>')
        for cmd in self.COMMANDS:
            self._print(f'<info>  {cmd}</info>')

    def _handle_chat(self, text: str) -> None:
        """Send chat to the agent."""
        try:
            output = self.agent.run(text)
            print(output)
            self.store.save(self.agent.session)
        except Exception as e:
            self._print(f'<error>Agent error: {e}</error>')


def print_formatted_text(*args: Any, **kwargs: Any) -> None:
    """Re-export for convenience."""
    from prompt_toolkit import print_formatted_text
    print_formatted_text(*args, **kwargs)
