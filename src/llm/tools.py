"""Tool registry + executor for the ReAct agent loop."""

from __future__ import annotations

from typing import Any, Callable

from .types import ToolDefinition, ToolResult


class ToolRegistry:
    """Collects tools and their handlers."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, definition: ToolDefinition, handler: Callable) -> None:
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                tool_call_id=name,
                content=f"Unknown tool: {name}",
                is_error=True,
            )
        try:
            import inspect
            result = handler(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return ToolResult(tool_call_id=name, content=str(result))
        except Exception as e:
            return ToolResult(tool_call_id=name, content=str(e), is_error=True)


class ToolExecutor:
    """Sync wrapper over ToolRegistry for the agent loop."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            # Already in an async context — run in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.registry.execute(name, arguments)).result()
        return loop.run_until_complete(self.registry.execute(name, arguments))
