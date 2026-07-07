"""
MCP client wrapper.

Uses the official `mcp` Python SDK's stdio transport: launches the
server as a subprocess (npx/uvx, per its MCPServerSpec) and speaks the
MCP protocol over its stdin/stdout. This is the only place that touches
the `mcp` SDK directly — everything else in the app deals with plain
Python data (tool name + dict in, dict out).
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from app.core.config.logging_config import get_logger
from app.mcp.registry import MCPServerSpec

log = get_logger(__name__)


@asynccontextmanager
async def mcp_session(spec: MCPServerSpec):
    """Async context manager yielding a connected, initialized ClientSession
    for the given server spec. Usage:

        async with mcp_session(filesystem_server_spec()) as session:
            tools = await session.list_tools()
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_env = {**os.environ, **spec.env}
    params = StdioServerParameters(command=spec.command, args=spec.args, env=server_env)

    log.info("mcp_connecting", server=spec.name, command=spec.command)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            log.info("mcp_connected", server=spec.name)
            yield session


async def list_server_tools(spec: MCPServerSpec) -> List[Dict[str, Any]]:
    """Returns [{"name": ..., "description": ...}, ...] for a server."""
    async with mcp_session(spec) as session:
        result = await session.list_tools()
        return [{"name": t.name, "description": t.description} for t in result.tools]


async def call_server_tool(spec: MCPServerSpec, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Calls one tool on a server and returns its raw result content."""
    async with mcp_session(spec) as session:
        result = await session.call_tool(tool_name, arguments)
        # MCP tool results are a list of content blocks (text/image/etc.) —
        # for our use cases (filesystem/sqlite/github/notion) it's almost
        # always plain text, so flatten that common case for callers.
        texts = [block.text for block in result.content if hasattr(block, "text")]
        return "\n".join(texts) if texts else result.content
