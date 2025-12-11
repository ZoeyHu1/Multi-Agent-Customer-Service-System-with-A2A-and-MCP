"""
Custom MCP-backed tools for Google ADK agents.

Each tool exposes one of the MCP server's database utilities (get_customer,
list_customers, etc.) using the ADK BaseTool interface so Gemini can call the
tool directly during an A2A request.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import anyio
from google.genai import types as genai_types
from mcp import ClientSessionGroup
from mcp.client.session_group import StreamableHttpParameters

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")


async def _call_mcp_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Invoke an MCP tool over HTTP/SSE and return its textual response."""

    async def _run():
        async with ClientSessionGroup() as group:
            session = await group.connect_to_server(
                StreamableHttpParameters(url=f"{MCP_SERVER_URL}/mcp")
            )
            return await session.call_tool(name=name, arguments=arguments or {})

    result = await _run()
    segments: list[str] = []
    for block in result.content:
        if getattr(block, "type", "") == "text" and hasattr(block, "text"):
            segments.append(block.text)
        else:
            try:
                segments.append(json.dumps(block.model_dump()))
            except Exception:
                segments.append(str(block))
    if result.structuredContent:
        segments.append(json.dumps(result.structuredContent))
    return "\n".join(segments) if segments else "Tool call returned no data."


class MCPTool(BaseTool):
    """BaseTool implementation backed by the FastMCP HTTP server."""

    def __init__(self, *, name: str, description: str, schema: Dict[str, Any]):
        super().__init__(name=name, description=description)
        self._schema = schema

    def _get_declaration(self) -> genai_types.FunctionDeclaration:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self._schema,
        )

    async def run_async(
        self, *, args: Dict[str, Any], tool_context: ToolContext
    ) -> Any:
        return await _call_mcp_tool(self.name, args or {})


def customer_data_tools() -> list[MCPTool]:
    """Tools available to the Customer Data ADK agent."""
    return [
        MCPTool(
            name="get_customer",
            description="Retrieve a customer's profile by ID.",
            schema={
                "type": "object",
                "properties": {"customer_id": {"type": "integer"}},
                "required": ["customer_id"],
            },
        ),
        MCPTool(
            name="list_customers",
            description="List customers optionally filtered by status.",
            schema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["active", "disabled"],
                        "description": "Optional status filter.",
                    }
                },
            },
        ),
        MCPTool(
            name="update_customer",
            description="Update name/email/phone/status for a customer.",
            schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "status": {"type": "string", "enum": ["active", "disabled"]},
                },
                "required": ["customer_id"],
            },
        ),
        MCPTool(
            name="get_customer_history",
            description="Fetch ticket history for a customer, optionally filtered by status.",
            schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "description": "Optional ticket status filter.",
                    },
                },
                "required": ["customer_id"],
            },
        ),
        MCPTool(
            name="list_active_customers_with_open_tickets",
            description="List active customers who currently have open tickets.",
            schema={"type": "object", "properties": {}},
        ),
        MCPTool(
            name="get_high_priority_tickets_by_ids",
            description="Return high-priority tickets for a list of customer IDs.",
            schema={
                "type": "object",
                "properties": {
                    "customer_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of customer IDs.",
                    }
                },
                "required": ["customer_ids"],
            },
        ),
    ]


def support_agent_tools() -> list[MCPTool]:
    """Tools available to the Support ADK agent."""
    return [
        MCPTool(
            name="create_ticket",
            description="Create a support ticket for a customer.",
            schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "issue": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["customer_id", "issue", "priority"],
            },
        ),
    ] + customer_data_tools()

