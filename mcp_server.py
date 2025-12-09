"""
MCP HTTP/SSE server that exposes customer-support tools.

Run with:
    python mcp_server.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, Callable

from flask import Flask, request, Response, jsonify
from flask_cors import CORS

from database_utils import CustomerServiceDB

# -----------------------------------------------------------------------------
# Server bootstrap
# -----------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)
db = CustomerServiceDB()


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def create_sse_message(payload: Dict[str, Any]) -> str:
    """Format a payload as an SSE message."""
    return f"data: {json.dumps(payload)}\n\n"


def format_db_result(raw: str) -> Dict[str, Any]:
    """Normalize outputs from database_utils into structured dicts."""
    try:
        parsed = json.loads(raw)
        return {"success": True, "data": parsed}
    except json.JSONDecodeError:
        upper = raw.upper()
        if upper.startswith("ERROR"):
            return {"success": False, "error": raw}
        return {"success": True, "message": raw}


# -----------------------------------------------------------------------------
# Tool definitions (mirrors ADK tutorial schema)
# -----------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "get_customer",
        "description": "Retrieve a customer's profile by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "Unique customer ID."
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "list_customers",
        "description": "List customers, optionally filtered by status ('active' or 'disabled').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "disabled"],
                    "description": "Optional status filter."
                }
            }
        }
    },
    {
        "name": "update_customer",
        "description": "Update customer contact information or status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "status": {"type": "string", "enum": ["active", "disabled"]}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "create_ticket",
        "description": "Create a new support ticket for a customer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "issue": {"type": "string"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Ticket priority"
                }
            },
            "required": ["customer_id", "issue", "priority"]
        }
    },
    {
        "name": "get_customer_history",
        "description": "Retrieve ticket history for a customer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "status": {"type": "string", "description": "Optional ticket status filter"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "get_high_priority_tickets_by_ids",
        "description": "Get high-priority tickets for a list of customer IDs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_ids": {
                    "type": "array",
                    "description": "List of customer IDs.",
                    "items": {"type": "integer"}
                }
            },
            "required": ["customer_ids"]
        }
    },
    {
        "name": "list_active_customers_with_open_tickets",
        "description": "List active customers who currently have open tickets.",
        "inputSchema": {"type": "object", "properties": {}}
    },
]


def tool_get_customer(customer_id: int, **_: Any) -> Dict[str, Any]:
    return format_db_result(db.get_customer(customer_id))


def tool_list_customers(status: str | None = None, **_: Any) -> Dict[str, Any]:
    return format_db_result(db.list_customers(status))


def tool_update_customer(customer_id: int, **data: Any) -> Dict[str, Any]:
    return format_db_result(db.update_customer(customer_id, data))


def tool_create_ticket(customer_id: int, issue: str, priority: str, **_: Any) -> Dict[str, Any]:
    return format_db_result(db.create_ticket(customer_id, issue, priority))


def tool_get_customer_history(customer_id: int, status: str | None = None, **_: Any) -> Dict[str, Any]:
    return format_db_result(db.get_customer_history(customer_id, status))


def tool_get_high_priority(customer_ids: list[int], **_: Any) -> Dict[str, Any]:
    return format_db_result(db.get_high_priority_tickets_by_ids(customer_ids))


def tool_list_active_with_open(**_: Any) -> Dict[str, Any]:
    return format_db_result(db.list_active_customers_with_open_tickets())


TOOL_FUNCTIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "get_customer": tool_get_customer,
    "list_customers": tool_list_customers,
    "update_customer": tool_update_customer,
    "create_ticket": tool_create_ticket,
    "get_customer_history": tool_get_customer_history,
    "get_high_priority_tickets_by_ids": tool_get_high_priority,
    "list_active_customers_with_open_tickets": tool_list_active_with_open,
}


# -----------------------------------------------------------------------------
# MCP protocol handlers
# -----------------------------------------------------------------------------

def handle_initialize(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "customer-management-mcp-server", "version": "1.0.0"},
        },
    }


def handle_tools_list(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {"tools": MCP_TOOLS},
    }


def handle_tools_call(message: Dict[str, Any]) -> Dict[str, Any]:
    params = message.get("params", {})
    name = params.get("name")
    arguments = params.get("arguments", {}) or {}

    if name not in TOOL_FUNCTIONS:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {"code": -32601, "message": f"Unknown tool: {name}"},
        }

    try:
        result = TOOL_FUNCTIONS[name](**arguments)
        payload = json.dumps(result, indent=2, sort_keys=True)
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {"content": [{"type": "text", "text": payload}]},
        }
    except TypeError as exc:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {"code": -32602, "message": f"Invalid params for {name}: {exc}"},
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {"code": -32603, "message": f"Tool execution error: {exc}"},
        }


def handle_prompts_list(message: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message.get("id"), "result": {"prompts": []}}


def handle_resources_list(message: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message.get("id"), "result": {"resources": []}}


def handle_resources_read(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "error": {"code": -32601, "message": "No resources available"},
    }


def process_mcp_message(message: Dict[str, Any]) -> Dict[str, Any]:
    method = message.get("method")
    if method == "initialize":
        return handle_initialize(message)
    if method == "tools/list":
        return handle_tools_list(message)
    if method == "tools/call":
        return handle_tools_call(message)
    if method == "prompts/list":
        return handle_prompts_list(message)
    if method == "resources/list":
        return handle_resources_list(message)
    if method == "resources/read":
        return handle_resources_read(message)
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


# -----------------------------------------------------------------------------
# Flask routes
# -----------------------------------------------------------------------------

@app.route("/mcp", methods=["POST"])
def mcp_endpoint() -> Response:
    message = request.get_json(force=True, silent=False)

    def generate():
        try:
            response = process_mcp_message(message)
            yield create_sse_message(response)
        except Exception as exc:  # pragma: no cover
            error_payload = {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32700, "message": f"Server error: {exc}"},
            }
            yield create_sse_message(error_payload)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "server": "customer-management-mcp-server", "version": "1.0.0"})


if __name__ == "__main__":
    print("Starting MCP server on http://0.0.0.0:8000 (SSE)")
    app.run(host="0.0.0.0", port=8000, debug=False)
