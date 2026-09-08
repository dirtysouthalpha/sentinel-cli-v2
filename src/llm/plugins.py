"""Plugin system — extensible tool and hook registry."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .types import ToolDefinition, ToolResult


class ToolHandler(Protocol):
    """Protocol for tool handlers."""
    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass
class Plugin:
    """A plugin that provides tools and hooks."""
    name: str
    version: str = "0.1.0"
    description: str = ""
    tools: list[ToolDefinition] = field(default_factory=list)
    handlers: dict[str, ToolHandler] = field(default_factory=dict)
    hooks: dict[str, list[Callable]] = field(default_factory=dict)

    def register_tool(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        self.tools.append(definition)
        self.handlers[definition.name] = handler

    def add_hook(self, event: str, callback: Callable) -> None:
        self.hooks.setdefault(event, []).append(callback)


class PluginManager:
    """Discovers, loads, and manages plugins."""

    def __init__(self, plugin_dirs: list[str | Path] | None = None):
        self.plugin_dirs: list[Path] = []
        if plugin_dirs:
            for d in plugin_dirs:
                self.plugin_dirs.append(Path(d))
        # Default plugin dirs
        self.plugin_dirs.append(Path.home() / ".sentinel" / "plugins")
        self.plugin_dirs.append(Path("/workspace/sentinel-cli-v2/plugins"))
        self._plugins: dict[str, Plugin] = {}

    @property
    def plugins(self) -> dict[str, Plugin]:
        return self._plugins

    def discover(self) -> list[str]:
        """Find all plugin names in plugin directories."""
        names = []
        for d in self.plugin_dirs:
            if not d.exists():
                continue
            for item in d.iterdir():
                if item.is_dir() and (item / "plugin.py").exists():
                    names.append(item.name)
                elif item.suffix == ".py" and item.name != "__init__.py":
                    names.append(item.stem)
        return names

    def load(self, name: str) -> Plugin | None:
        """Load a plugin by name."""
        if name in self._plugins:
            return self._plugins[name]

        for d in self.plugin_dirs:
            # Try directory plugin
            plugin_dir = d / name
            plugin_file = plugin_dir / "plugin.py"
            if plugin_file.exists():
                plugin = self._load_from_file(plugin_file, name)
                if plugin:
                    self._plugins[name] = plugin
                    return plugin

            # Try single-file plugin
            single_file = d / f"{name}.py"
            if single_file.exists():
                plugin = self._load_from_file(single_file, name)
                if plugin:
                    self._plugins[name] = plugin
                    return plugin

        return None

    def _load_from_file(self, path: Path, name: str) -> Plugin | None:
        """Load a plugin from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location(f"sentinel_plugin_{name}", path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for plugin metadata
            version = getattr(module, "__version__", "0.1.0")
            description = getattr(module, "__doc__", "") or ""

            plugin = Plugin(name=name, version=version, description=description)

            # Look for register function
            register = getattr(module, "register", None)
            if callable(register):
                register(plugin)

            return plugin
        except Exception as e:
            print(f"Failed to load plugin {name}: {e}")
            return None

    def load_all(self) -> int:
        """Load all discovered plugins. Returns count loaded."""
        count = 0
        for name in self.discover():
            if self.load(name):
                count += 1
        return count

    def get_all_tools(self) -> tuple[list[ToolDefinition], dict[str, ToolHandler]]:
        """Get all tools from all loaded plugins."""
        all_tools: list[ToolDefinition] = []
        all_handlers: dict[str, ToolHandler] = {}
        for plugin in self._plugins.values():
            all_tools.extend(plugin.tools)
            all_handlers.update(plugin.handlers)
        return all_tools, all_handlers

    def run_hooks(self, event: str, **kwargs: Any) -> None:
        """Run all hooks for an event."""
        for plugin in self._plugins.values():
            for hook in plugin.hooks.get(event, []):
                try:
                    hook(**kwargs)
                except Exception as e:
                    print(f"Hook error in {plugin.name}/{event}: {e}")


# ---------------------------------------------------------------------------
# Built-in plugins
# ---------------------------------------------------------------------------

def create_filesystem_plugin() -> Plugin:
    """Plugin providing filesystem tools."""
    plugin = Plugin(name="filesystem", description="File system operations")

    def read_file(path: str) -> str:
        return Path(path).read_text()

    def write_file(path: str, content: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def list_dir(path: str = ".") -> str:
        p = Path(path)
        if not p.exists():
            return f"Path not found: {path}"
        items = []
        for item in sorted(p.iterdir()):
            prefix = "📁" if item.is_dir() else "📄"
            items.append(f"{prefix} {item.name}")
        return "\n".join(items)

    plugin.register_tool(ToolDefinition(
        name="read",
        description="Read a file's contents",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ), read_file)

    plugin.register_tool(ToolDefinition(
        name="write",
        description="Write content to a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    ), write_file)

    plugin.register_tool(ToolDefinition(
        name="ls",
        description="List directory contents",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
    ), list_dir)

    return plugin


def create_shell_plugin() -> Plugin:
    """Plugin providing shell execution."""
    plugin = Plugin(name="shell", description="Shell command execution")

    def shell(command: str, timeout: int = 30) -> str:
        import subprocess
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr] {result.stderr}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    plugin.register_tool(ToolDefinition(
        name="shell",
        description="Run a shell command",
        input_schema={"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]},
    ), shell)

    return plugin


def create_memory_plugin() -> Plugin:
    """Plugin providing memory tools via Neuralis brain."""
    plugin = Plugin(name="memory", description="Long-term memory via Neuralis brain")

    def recall(context: str, limit: int = 5) -> str:
        from ..memory.cache import TieredMemory
        mem = TieredMemory()
        entries = mem.recall(context, limit)
        if not entries:
            return "(no memories found)"
        return "\n".join(str(e) for e in entries)

    def remember(content: str, region: str = "knowledge", topic: str = "sentinel-cli") -> str:
        from ..memory.cache import TieredMemory
        mem = TieredMemory()
        ok = mem.remember(content, region, topic)
        return "Stored" if ok else "Failed"

    def search(query: str) -> str:
        from ..memory.cache import TieredMemory
        mem = TieredMemory()
        entries = mem.search(query)
        if not entries:
            return "(no results)"
        return "\n".join(str(e) for e in entries)

    plugin.register_tool(ToolDefinition(
        name="recall",
        description="Recall relevant memories for a context",
        input_schema={"type": "object", "properties": {"context": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["context"]},
    ), recall)

    plugin.register_tool(ToolDefinition(
        name="remember",
        description="Store a new memory",
        input_schema={"type": "object", "properties": {"content": {"type": "string"}, "region": {"type": "string"}, "topic": {"type": "string"}}, "required": ["content"]},
    ), remember)

    plugin.register_tool(ToolDefinition(
        name="search",
        description="Search the memory store",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ), search)

    return plugin


def create_web_plugin() -> Plugin:
    """Plugin providing web fetch tools."""
    plugin = Plugin(name="web", description="Web fetch and search")

    def fetch(url: str) -> str:
        import urllib.request
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Sentinel-CLI/2.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")[:5000]
        except Exception as e:
            return f"Error: {e}"

    plugin.register_tool(ToolDefinition(
        name="fetch",
        description="Fetch a URL and return its content",
        input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    ), fetch)

    return plugin


def load_builtin_plugins() -> dict[str, Plugin]:
    """Load all built-in plugins."""
    return {
        "filesystem": create_filesystem_plugin(),
        "shell": create_shell_plugin(),
        "memory": create_memory_plugin(),
        "web": create_web_plugin(),
    }
