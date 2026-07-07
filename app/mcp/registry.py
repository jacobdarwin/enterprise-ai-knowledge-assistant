"""
MCP server registry.

Each MCP server is described declaratively as an MCPServerSpec (how to
launch it, over what transport). Adding a new server later — Slack,
Jira, whatever — means adding one function here that returns a spec; no
changes needed to the client code in app/mcp/client.py, which only
knows how to speak the MCP protocol, not about any specific server.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from app.core.config.settings import get_settings


@dataclass
class MCPServerSpec:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True


def filesystem_server_spec() -> MCPServerSpec:
    """Local filesystem access, scoped to MCP_FILESYSTEM_ROOT — the reference
    server refuses to touch anything outside the directories it's given."""
    settings = get_settings()
    return MCPServerSpec(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", settings.mcp_filesystem_root],
        description="Read/write access to the local knowledge-base folder.",
        enabled=True,
    )


def sqlite_server_spec() -> MCPServerSpec:
    """Query the app's own SQLite DB (documents/chat history) via MCP —
    useful for a future 'ask questions about your usage/history' agent."""
    settings = get_settings()
    db_path = settings.database_url.split("///")[-1] if "///" in settings.database_url else "./data/app.db"
    return MCPServerSpec(
        name="sqlite",
        command="uvx",
        args=["mcp-server-sqlite", "--db-path", db_path],
        description="Query the app's own SQLite database (documents, chat history).",
        enabled=True,
    )


def github_server_spec() -> MCPServerSpec:
    """Requires MCP_GITHUB_TOKEN — disabled automatically if not set, rather
    than failing at connection time with a confusing auth error."""
    settings = get_settings()
    return MCPServerSpec(
        name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": settings.mcp_github_token},
        description="Read issues/PRs/repos — useful for engineering-team knowledge queries.",
        enabled=bool(settings.mcp_github_token),
    )


def notion_server_spec() -> MCPServerSpec:
    """Requires MCP_NOTION_TOKEN — disabled automatically if not set."""
    settings = get_settings()
    return MCPServerSpec(
        name="notion",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-notion"],
        env={"NOTION_API_KEY": settings.mcp_notion_token},
        description="Read company wiki/docs pages stored in Notion.",
        enabled=bool(settings.mcp_notion_token),
    )


_SPEC_BUILDERS = [filesystem_server_spec, sqlite_server_spec, github_server_spec, notion_server_spec]


def get_all_server_specs() -> Dict[str, MCPServerSpec]:
    """Returns every registered server, including disabled ones (so the
    Settings page can show 'GitHub: disabled — no token configured')."""
    return {builder().name: builder() for builder in _SPEC_BUILDERS}


def get_enabled_server_specs() -> Dict[str, MCPServerSpec]:
    return {name: spec for name, spec in get_all_server_specs().items() if spec.enabled}


def register_server(spec_builder) -> None:
    """Extension point: append a new zero-arg callable returning MCPServerSpec
    to support additional MCP servers (Slack, Jira, ...) without touching
    anything else in this module."""
    _SPEC_BUILDERS.append(spec_builder)
