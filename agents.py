# agents.py - Uses Google ADK pattern (MCPToolSet) for Tool Access

import os
import json
from google import genai
from google.genai.errors import APIError
from typing import Dict, Any, List
import anyio
from mcp import ClientSessionGroup
from mcp.client.session_group import StreamableHttpParameters

# --- IMPORTANT CONFIGURATION ---
# 1. Update this with your actual Ngrok URL for the MCP Server (e.g., https://abcd123.ngrok-free.app)
# This will be updated in the setup guide.
MCP_SERVER_URL = "https://morbifically-unhemmed-danica.ngrok-free.dev"

# Global client initialization
try:
    client = genai.Client()
    MODEL_NAME = "gemini-2.5-flash"
    print("Gemini Client Initialized successfully.")
except Exception as e:
    print(f"ERROR: Gemini Client initialization failed: {e}")
    client = None
    MODEL_NAME = "Unavailable"

# --- Helper functions for MCP invocation ---

def _invoke_mcp_tool(func_name: str, func_args: Dict[str, Any]) -> str:
    async def _call():
        async with ClientSessionGroup() as group:
            session = await group.connect_to_server(
                StreamableHttpParameters(url=f"{MCP_SERVER_URL}/mcp")
            )
            return await session.call_tool(
                name=func_name,
                arguments=func_args or {}
            )

    result = anyio.run(_call)
    segments = []
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

# --- ADK Tool Definition Pattern (MCPToolSet Simulation) ---

def create_mcp_toolset(url: str, tools_to_use: List[str]) -> genai.types.Tool:
    """
    Simulates the ADK's MCPToolSet class creation by defining a Tool
    that points to the MCP server URL.
    """
    if not url.startswith("http"):
        raise ValueError("MCP_SERVER_URL must be a valid HTTP/HTTPS URL.")
    
    # Define schema-aligned function declarations so Gemini knows required args
    schema_map = {
        "get_customer": genai.types.FunctionDeclaration(
            name="get_customer",
            description="Retrieve a single customer's profile by ID.",
            parameters={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "Unique customer ID."
                    }
                },
                "required": ["customer_id"]
            }
        ),
        "list_customers": genai.types.FunctionDeclaration(
            name="list_customers",
            description="List customers optionally filtered by status.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["active", "disabled"],
                        "description": "Optional status filter."
                    }
                }
            }
        ),
        "update_customer": genai.types.FunctionDeclaration(
            name="update_customer",
            description="Update name/email/phone/status for a customer.",
            parameters={
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
        ),
        "create_ticket": genai.types.FunctionDeclaration(
            name="create_ticket",
            description="Create a support ticket for a customer.",
            parameters={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "issue": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Ticket priority level."
                    }
                },
                "required": ["customer_id", "issue", "priority"]
            }
        ),
        "get_customer_history": genai.types.FunctionDeclaration(
            name="get_customer_history",
            description="Fetch ticket history for a customer, optionally filtered by status.",
            parameters={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "status": {"type": "string", "description": "Optional ticket status filter."}
                },
                "required": ["customer_id"]
            }
        ),
        "get_high_priority_tickets_by_ids": genai.types.FunctionDeclaration(
            name="get_high_priority_tickets_by_ids",
            description="Return high-priority tickets for a list of customer IDs.",
            parameters={
                "type": "object",
                "properties": {
                    "customer_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of customer IDs."
                    }
                },
                "required": ["customer_ids"]
            }
        ),
        "list_active_customers_with_open_tickets": genai.types.FunctionDeclaration(
            name="list_active_customers_with_open_tickets",
            description="List active customers who currently have open tickets.",
            parameters={"type": "object", "properties": {}}
        ),
    }

    tool_declarations = [schema_map[name] for name in tools_to_use if name in schema_map]

    # Create the Tool object to expose the allowed function declarations.
    # (The official Google GenAI Tool schema does not accept a URL, so
    # we simply record the declarations the agent may invoke.)
    mcp_tool = genai.types.Tool(function_declarations=tool_declarations)
    return mcp_tool


# --- Agent Base Class and Implementations ---

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools_to_use: List[str]):
        self.name = name
        self.system_prompt = system_prompt
        self.mcp_tool = None
        
        if tools_to_use:
            self.mcp_tool = create_mcp_toolset(MCP_SERVER_URL, tools_to_use)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Processes the input state and executes tool calls via the MCP server URL."""
        
        if not client:
            state['error'] = "Configuration Error: Gemini Client not initialized. Cannot run agent."
            return state

        query = state.get("query", "")
        context = state.get("context", "")
        
        prompt = f"""
        Current Task: {state['task']}
        Customer Query: {query}
        Available Context:
        {context if context else 'None'}
        
        Your role is the {self.name}. Analyze the task and context. 
        If you need to use a tool, output a single tool call.
        If you have enough information, generate the final response or the structured message for the next agent.
        """
        
        print(f"\n--- {self.name} is processing... ---")
        api_contents = [
            genai.types.Part.from_text(text=self.system_prompt),
            genai.types.Part.from_text(text=prompt)
        ]
        
        # Prepare tools for the request
        tools_list: List[Any] = []
        if self.mcp_tool:
            tools_list.append(self.mcp_tool)

        try:
            # 1. First LLM call: decide on tool use
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=api_contents,
                config=genai.types.GenerateContentConfig(
                    tools=tools_list
                )
            )

            # 2. Tool Calling Loop (This is where the agent triggers the remote MCP call)
            if response.function_calls:
                function_call = response.function_calls[0]
                func_name = function_call.name
                func_args = dict(function_call.args)
                print(f"**LOG: {self.name} calling MCP Tool: {func_name}({func_args})**")
                try:
                    tool_result_text = _invoke_mcp_tool(func_name, func_args)
                except Exception as tool_error:
                    error_message = f"MCP tool '{func_name}' failed: {tool_error}"
                    print(f"**ERROR: {error_message}**")
                    state['error'] = error_message
                    state['last_output'] = error_message
                    return state

                api_contents.append(
                    genai.types.Part.from_function_response(
                        name=func_name,
                        response={'result': tool_result_text} 
                    )
                )
                second_response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=api_contents
                )
                final_text = second_response.text.strip() if second_response.text else ""
                state['last_output'] = final_text or tool_result_text
                state['agent_name'] = self.name
                return state
                
            state['last_output'] = response.text
            state['agent_name'] = self.name
            return state

        except APIError as e:
            error_message = f"Gemini API Error in {self.name}: {e}"
            print(f"**ERROR: {error_message}**")
            state['error'] = error_message
            return state
        except Exception as e:
            state['error'] = f"Unhandled Error in {self.name}: {e}"
            return state

class RouterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="RouterAgent (Orchestrator)",
            system_prompt=(
                "You are the Router and Orchestrator. Analyze the query intent. "
                "If data is needed first (get customer, list customers), set next_agent: 'data_agent_then_support' or 'data_agent'."
                "If support/ticket handling is primary, set next_agent: 'support_agent'."
                "Your final output MUST be a JSON object with 'next_agent' and 'task' keys."
            ),
            tools_to_use=[]
        )

    def _determine_route(self, query: str) -> Dict[str, str]:
        """Simple intent classification to keep routing deterministic."""
        q = (query or "").lower()
        if not q:
            return {
                "next_agent": "data_agent",
                "task": "Retrieve any available customer context for the empty query."
            }

        if "active" in q and "open ticket" in q:
            return {
                "next_agent": "data_agent",
                "task": "Call the 'list_active_customers_with_open_tickets' tool to retrieve all active customers that currently have open tickets."
            }

        if "high-priority" in q and "ticket" in q:
            return {
                "next_agent": "data_agent_then_support",
                "task": "List all active customers, share IDs with Support, and instruct Support to summarize high-priority tickets."
            }

        if "update" in q and ("history" in q or "ticket history" in q):
            return {
                "next_agent": "data_agent_then_support",
                "task": "Update the requested customer fields, then provide their ticket history so Support can respond."
            }

        if ("charged" in q or "refund" in q or "billing" in q or "cancel" in q):
            return {
                "next_agent": "support_agent",
                "task": "Handle the urgent billing or cancellation issue and create a high-priority ticket if necessary."
            }

        if "customer id" in q and ("help" in q or "issue" in q or "password" in q):
            return {
                "next_agent": "data_agent_then_support",
                "task": "Fetch the profile for the referenced customer ID before handing off to Support for resolution."
            }

        if "get customer" in q or "customer information" in q or ("show" in q and "customer" in q):
            return {
                "next_agent": "data_agent",
                "task": "Return the requested customer information."
            }

        return {
            "next_agent": "support_agent",
            "task": "Provide a support-focused response using any available context."
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Override BaseAgent logic with deterministic routing heuristics."""
        routing = self._determine_route(state.get("query", ""))
        new_state = dict(state)
        new_state["next_agent"] = routing["next_agent"]
        new_state["task"] = routing["task"]
        new_state["agent_name"] = self.name
        new_state["last_output"] = (
            f"Router decision -> next_agent: {routing['next_agent']} | task: {routing['task']}"
        )
        return new_state

class CustomerDataAgent(BaseAgent):
    def __init__(self):
        # Tools this agent is authorized to use on the MCP Server
        data_tools = [
            'get_customer',
            'list_customers',
            'update_customer',
            'get_customer_history',
            'get_high_priority_tickets_by_ids',
            'list_active_customers_with_open_tickets'
        ]
        super().__init__(
            name="CustomerDataAgent (Specialist)",
            system_prompt=(
                "You are the Customer Data Agent. Your sole purpose is to retrieve or update customer records using the authorized MCP tools. "
                "You MUST call a tool if the task requires data retrieval or update. Your output must be summarized data."
            ),
            tools_to_use=data_tools
        )

class SupportAgent(BaseAgent):
    def __init__(self):
        # Tools this agent is authorized to use on the MCP Server
        support_tools = ['create_ticket', 'get_customer_history', 'get_customer', 'update_customer', 'get_high_priority_tickets_by_ids']
        super().__init__(
            name="SupportAgent (Specialist)",
            system_prompt=(
                "You are the Support Agent. Handle issues, provide solutions, and manage tickets. "
                "Use context from other agents/tools to give a polite, professional, and final response. "
                "Prioritize creating high-priority tickets for urgent issues."
            ),
            tools_to_use=support_tools
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mutable_state = dict(state)
        history_customer_id = mutable_state.get('history_customer_id')
        if history_customer_id:
            try:
                history_text = _invoke_mcp_tool(
                    "get_customer_history",
                    {"customer_id": history_customer_id}
                )
                existing_context = mutable_state.get('context', '')
                addition = f"[Auto Ticket History for {history_customer_id}]: {history_text}"
                mutable_state['context'] = (existing_context + "\n" + addition).strip()
            except Exception as e:
                print(f"**WARNING: Failed to auto-fetch ticket history: {e}**")
        return super().run(mutable_state)
