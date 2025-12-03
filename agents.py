# agents.py - IMPROVED

import os
from google import genai
from google.genai.errors import APIError
from mcp_server import MCP_TOOLS 
from typing import Dict, Any, List

# --- Configuration & Initialization ---
# Set your API Key here or as an environment variable (Recommended)
os.environ['GEMINI_API_KEY'] = 'YOUR_GEMINI_API_KEY_HERE'
client = None
MODEL_NAME = "DummyModel"

try:
    client = genai.Client()
    MODEL_NAME = "gemini-2.5-flash"
    print("Gemini Client Initialized successfully.")
except Exception as e:
    print(f"WARNING: Gemini Client initialization failed ({e}). Using DummyModel.")
    # Define DummyModel to handle tool calls when client fails
    class DummyModel:
        def generate_content(self, contents, config=None, tools=None):
            # Simple simulation of tool-calling based on keywords
            query_part = contents[-1] if isinstance(contents, list) else contents
            query_text = query_part.text if hasattr(query_part, 'text') else str(query_part)
            
            # 1. Update Customer Simulation (for Multi-Intent test)
            if 'update my email' in query_text.lower() and 'id' in query_text.lower():
                customer_id = next((int(s) for s in query_text.split() if s.isdigit()), 3)
                print(f"**LOG: Dummy Agent calling tool: update_customer for ID {customer_id}**")
                # Simulate a successful update and subsequent history check
                MCP_TOOLS['update_customer'](customer_id=customer_id, data={'email': 'bob.new@business.net'})
                return f"Customer ID {customer_id} updated. Now retrieving history: " + MCP_TOOLS['get_customer_history'](customer_id=customer_id)

            # 2. Get Customer Simulation
            if 'get customer' in query_text.lower() and any(c.isdigit() for c in query_text):
                customer_id = next((int(s) for s in query_text.split() if s.isdigit()), 1)
                print(f"**LOG: Dummy Agent calling tool: get_customer with args: {{'customer_id': {customer_id}}}**")
                return MCP_TOOLS['get_customer'](customer_id=customer_id)

            # 3. Create Ticket Simulation
            elif 'charged' in query_text.lower() or 'refund' in query_text.lower():
                customer_id = next((int(s) for s in query_text.split() if s.isdigit()), 1)
                print(f"**LOG: Dummy Agent calling tool: create_ticket (high) for ID {customer_id}**")
                return MCP_TOOLS['create_ticket'](customer_id=customer_id, issue="Billing dispute/refund request", priority="high")
            
            # 4. List Customers Simulation (for Scenario 3)
            elif 'retrieve all active customers' in query_text.lower():
                 print(f"**LOG: Dummy Agent calling tool: list_customers (active)**")
                 return MCP_TOOLS['list_customers'](status='active', limit=100)
            
            # Simple text response
            return f"Dummy response processed for: {query_text[:50]}..."
            
# --- Agent Base Class and Implementations ---

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools: List = None):
        self.name = name
        self.system_prompt = system_prompt
        # Store raw Python functions for the workaround
        self.tool_functions = tools if tools is not None else []
        self.tools = [] # Not used in this workaround, but kept for clarity

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Processes the input state and simulates the LLM interaction and tool use."""
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
        api_contents = [self.system_prompt, prompt]

        if client is None:
            # Dummy Model execution path
            tool_output = DummyModel().generate_content(
                contents=api_contents, 
                # Use tool_functions directly
                tools=self.tool_functions,
                config=genai.types.GenerateContentConfig(
                    tool_config=genai.types.ToolConfig(
                        function_calling_config=genai.types.FunctionCallingConfig(
                            mode="ANY"
                        )
                    )
                )
            )
            state['last_output'] = tool_output
            state['agent_name'] = self.name
            return state

        # Real Gemini API execution
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=api_contents,
                config=genai.types.GenerateContentConfig(
                    # Pass raw functions to the client (WORKAROUND)
                    tools=self.tool_functions, 
                )
            )

            if response.function_calls:
                function_call = response.function_calls[0]
                func_name = function_call.name
                func_args = dict(function_call.args)
                print(f"**LOG: {self.name} calling MCP Tool: {func_name}({func_args})**")
                
                tool_func = MCP_TOOLS.get(func_name)
                if tool_func:
                    tool_result = tool_func(**func_args)
                    print(f"**LOG: Tool Result (abridged): {tool_result[:100]}...**")

                    # Second LLM call: Re-run with the tool result (context)
                    api_contents.append(
                        genai.types.Part.from_function_response(
                            name=func_name,
                            response={'result': tool_result}
                        )
                    )
                    second_response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=api_contents
                    )
                    state['last_output'] = second_response.text
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

class RouterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="RouterAgent (Orchestrator)",
            system_prompt=(
                "You are the Router and Orchestrator. Analyze the query and determine the next step. "
                "Set the 'next_agent' to 'data_agent', 'support_agent', or 'data_agent_then_support'."
            ),
            tools=[]
        )
        
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified router logic based on keywords."""
        query = state.get("query", "").lower()
        
        # Scenario 1: Data fetch needed, then Support
        if ("id" in query and ("help" in query or "upgrade" in query or "password" in query)):
            next_agent = 'data_agent_then_support'
            task = "Fetch customer info (ID from query) and then route to support."
        # Scenario 3: Multi-Step Coordination (List customers, then get tickets)
        elif "status" in query and "ticket" in query and "active customer" in query:
            next_agent = 'data_agent' 
            task = "Action: retrieve all 'active' customers' IDs using list_customers, then pass to support for ticket filtering."
        # Scenario 2/Escalation/Multi-Intent
        elif "cancel" in query or "billing" in query or "refund" in query or "charged" in query or "update my email" in query:
            next_agent = 'support_agent'
            task = "Handle complex multi-intent query (e.g., update record, create ticket, or check history)."
        # Simple Data Query
        elif "customer information" in query or "get customer" in query:
            next_agent = 'data_agent'
            task = "Retrieve customer information."
        # Default to Support
        else:
            next_agent = 'support_agent'
            task = "Handle general support query."
        
        state['next_agent'] = next_agent
        state['task'] = task
        print(f"**LOG: Router decided next agent: {next_agent} for task: {task}**")
        return state

class CustomerDataAgent(BaseAgent):
    def __init__(self):
        data_tools = [MCP_TOOLS['get_customer'], MCP_TOOLS['list_customers'], MCP_TOOLS['update_customer'], MCP_TOOLS['get_customer_history']]
        super().__init__(
            name="CustomerDataAgent (Specialist)",
            system_prompt=(
                "You are the Customer Data Agent. Your sole purpose is to retrieve, list, or update customer records using the MCP tools. "
                "For multi-step tasks (like listing customers), ensure your tool call output is clean JSON for the next agent to consume."
            ),
            tools=data_tools
        )

class SupportAgent(BaseAgent):
    def __init__(self):
        support_tools = [MCP_TOOLS['create_ticket'], MCP_TOOLS['get_customer_history'], MCP_TOOLS['get_customer'], MCP_TOOLS['update_customer']]
        super().__init__(
            name="SupportAgent (Specialist)",
            system_prompt=(
                "You are the Support Agent. Your job is to handle customer issues, provide solutions, and manage tickets. "
                "If the query has multiple intents (e.g., update email AND get history), use multiple tools if necessary and synthesize the final polite response. "
                "You are the final customer-facing agent."
            ),
            tools=support_tools
        )