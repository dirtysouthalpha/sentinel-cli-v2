"""MCP (Model Context Protocol) client support for Sentinel CLI V2."""

from __future__ import annotations

from .client import MCPClient, MCPServer
from .plugin import create_mcp_plugin

__all__ = ["MCPClient", "MCPServer", "create_mcp_plugin"]
