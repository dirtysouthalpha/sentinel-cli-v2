"""MCP (Model Context Protocol) client for external tool servers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass
class MCPServer:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None


class MCPClient:
    """Client for connecting to MCP tool servers."""

    def __init__(self, server: MCPServer):
        self.server = server
        self._process: subprocess.Popen | None = None
        self._tools: list[dict[str, Any]] = []

    def connect(self) -> bool:
        """Start the MCP server process."""
        try:
            self._process = subprocess.Popen(
                [self.server.command] + self.server.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.server.env,
            )
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        """Stop the MCP server."""
        if self._process:
            self._process.terminate()
            self._process = None

    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server."""
        if not self._process:
            return []

        # Send list_tools request
        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        })
        self._process.stdin.write((request + "\n").encode())
        self._process.stdin.flush()

        # Read response
        try:
            response = self._process.stdout.readline().decode()
            data = json.loads(response)
            self._tools = data.get("result", {}).get("tools", [])
            return self._tools
        except Exception:
            return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server."""
        if not self._process:
            return {"error": "Not connected"}

        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        self._process.stdin.write((request + "\n").encode())
        self._process.stdin.flush()

        try:
            response = self._process.stdout.readline().decode()
            return json.loads(response)
        except Exception as e:
            return {"error": str(e)}
