"""MCP plugin bridge — exposes MCP server tools as Sentinel CLI tools."""

from __future__ import annotations

import json
from typing import Any

from ..types import ToolDefinition
from ...llm.plugins import Plugin


def create_mcp_plugin(server_configs: list[dict[str, Any]] | None = None) -> Plugin:
    """Build a Plugin whose tools are served by MCP servers.

    server_configs: [{"name": ..., "command": ..., "args": [...], "env": {...}}]
    """
    from .client import MCPClient, MCPServer

    plugin = Plugin(name="mcp", description="External MCP tool servers")
    clients: list[MCPClient] = []

    for sc in server_configs or []:
        server = MCPServer(
            name=sc.get("name", "mcp"),
            command=sc.get("command", ""),
            args=sc.get("args", []),
            env=sc.get("env"),
        )
        client = MCPClient(server)
        if client.connect():
            clients.append(client)
            for tool in client.list_tools():
                tname = tool.get("name")
                if not tname:
                    continue
                # Bind the client + tool name into the handler closure
                def _make_handler(cl: MCPClient, tn: str):
                    def handler(**kwargs: Any) -> Any:
                        result = cl.call_tool(tn, kwargs)
                        # MCP returns {"result": {"content": [...]}} or {"error": ...}
                        if "error" in result:
                            return f"MCP error: {result['error']}"
                        content = result.get("result", {}).get("content", [])
                        parts = []
                        for c in content:
                            if isinstance(c, dict) and "text" in c:
                                parts.append(c["text"])
                        return "\n".join(parts) or json.dumps(result)[:2000]
                    return handler

                plugin.register_tool(
                    ToolDefinition(
                        name=f"mcp_{tname}",
                        description=tool.get("description", f"MCP tool {tname}"),
                        input_schema=tool.get("inputSchema", {"type": "object", "properties": {}}),
                    ),
                    _make_handler(client, tname),
                )

    # Lifecycle hook: disconnect all servers when the plugin system shuts down
    def _shutdown(**kwargs: Any) -> None:
        for cl in clients:
            try:
                cl.disconnect()
            except Exception:
                pass

    plugin.add_hook("shutdown", _shutdown)
    return plugin
