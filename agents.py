# agents.py

import os
import json
from google import genai
from google.genai.errors import APIError
from mcp_server import MCP_TOOLS
from typing import Dict, Any, List

# --- Configuration ---
# Set your API Key here or as an environment variable
# os.environ['GEMINI_API_KEY'] = 'YOUR_GEMINI_API_KEY'
# client = genai.Client()

# Initialize a dummy client/model if API key is not set for local testing/placeholder
class DummyModel:
    def generate_content(self, contents, config=None, tools=None):
        # Simplifies tool calling for a robust example
        if tools and config and config.tool_config.function_calling_config.mode == "ANY":
            for tool_call in contents.tool_calls:
                func_name = tool_call.function.name
                func_args = dict(tool_call.function.args)
                print(f"**LOG: Dummy Agent calling tool: {func_name} with args: {func_args}**")
                tool_func = MCP_TOOLS.get(func_name)
                if tool_func:
                    result = tool_func(**func_args)
                    return result
        # Simple text response for non-tool calls
        return f"Dummy response for: {contents.text[:50]}..."

try:
    client = genai.Client()
    MODEL_NAME = "gemini-2.5-flash"
    print("Gemini Client Initialized successfully.")
except Exception as e:
    print(f"WARNING: Gemini Client initialization failed ({e}). Using DummyModel.")
    client = None
    MODEL_NAME = "DummyModel"

# --- Agent Base Class and Implementations ---

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools: List = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools if tools is not None else []

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input state and returns an updated state.
        This method simulates the LLM interaction and tool use.
        """
        query = state.get("query", "")
        context = state.get("context", "")
        # The prompt is constructed dynamically based on the current state
        prompt = f"""
        Current Task: {state['task']}
        Customer Query: {query}
        Available Context (Previous Agent Outputs/Data):
        {context if context else 'None'}
        
        Your role is the {self.name}. Analyze the task and context. 
        If you need to use a tool, output a single tool call as your response.
        If you have enough information or the task is complete, formulate a concise final response or a structured message to the next agent.
        """
        
        print(f"\n--- {self.name} is processing... ---")
        print(f"**LOG: {self.name} received task: {state['task']}**")
        print(f"**LOG: {self.name} current context length: {len(context)}**")

        if client is None:
            # Dummy Model execution for tool-use simulation
            result = DummyModel().generate_content(
                contents=[prompt], 
                tools=self.tools,
                config=genai.types.GenerateContentConfig(
                    tool_config=genai.types.ToolConfig(
                        function_calling_config=genai.types.FunctionCallingConfig(
                            mode="ANY"
                        )
                    )
                )
            )
            if isinstance(result, str) and result.startswith("ERROR:"):
                # Simulation of a tool error
                state['error'] = result
                state['last_output'] = f"Error from {self.name}: {result}"
                return state
            
            # The dummy model just returns the tool output directly, 
            # or a dummy text response.
            state['last_output'] = result
            state['agent_name'] = self.name
            return state

        # Real Gemini API execution (Simplified for LangGraph integration)
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    self.system_prompt,
                    prompt
                ],
                config=genai.types.GenerateContentConfig(
                    tools=self.tools,
                )
            )

            if response.function_calls:
                # Agent is calling an MCP tool
                function_call = response.function_calls[0]
                func_name = function_call.name
                func_args = dict(function_call.args)
                print(f"**LOG: {self.name} calling MCP Tool: {func_name}({func_args})**")
                
                tool_func = MCP_TOOLS.get(func_name)
                if tool_func:
                    tool_result = tool_func(**func_args)
                    print(f"**LOG: Tool Result (abridged): {tool_result[:100]}...**")

                    # Re-run the LLM with the tool result to get a final answer or next step
                    second_response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=[
                            self.system_prompt,
                            prompt,
                            genai.types.Part.from_function_response(
                                name=func_name,
                                response={'result': tool_result}
                            )
                        ]
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
                "You are the Router and Orchestrator. Your job is to analyze the user's query and context. "
                "Determine the primary intent (Data, Support, or Both). "
                "Based on the intent, set the 'next_agent' key in the output state to 'data_agent', 'support_agent', or 'data_agent_then_support'."
                "If the query can be answered with current context, set 'next_agent' to 'END'."
                "Your final output should be a structured JSON object with keys: 'next_agent', 'task', and 'context' (updated or summarized)."
            ),
            tools=[]
        )
        
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        The router logic is simplified here to a direct intent mapping for a runnable example.
        In a real scenario, the LLM would dynamically determine the next step.
        """
        query = state.get("query", "").lower()
        context = state.get("context", "")
        
        # Scenario 1: Task Allocation (Data fetch needed, then Support)
        if "id" in query and "help" in query:
            next_agent = 'data_agent_then_support'
            task = "Fetch customer ID and then provide support."
        # Scenario 3: Multi-Step Coordination (Data for status, then Support for tickets)
        elif "status" in query and "ticket" in query:
            next_agent = 'data_agent' 
            task = "First, identify premium customers, then find high-priority tickets for them."
        # Scenario 2: Negotiation/Escalation/Multi-Intent (Complex routing)
        elif "cancel" in query or "billing" in query or "refund" in query or "update" in query:
            next_agent = 'support_agent'
            task = "Complex multi-intent query. Check context, then negotiate with Data Agent if needed."
        # Simple Data Query
        elif "customer information" in query or "get customer" in query:
            next_agent = 'data_agent'
            task = "Retrieve customer information."
        # Simple Support Query
        else:
            next_agent = 'support_agent'
            task = "Handle general support query."
        
        # Update the state to reflect the routing decision
        state['next_agent'] = next_agent
        state['task'] = task
        print(f"**LOG: Router decided next agent: {next_agent} for task: {task}**")
        return state

class CustomerDataAgent(BaseAgent):
    def __init__(self):
        # We only pass the data-related tools to this agent
        data_tools = [MCP_TOOLS['get_customer'], MCP_TOOLS['list_customers'], MCP_TOOLS['update_customer'], MCP_TOOLS['get_customer_history']]
        super().__init__(
            name="CustomerDataAgent (Specialist)",
            system_prompt=(
                "You are the Customer Data Agent. Your only role is to access the customer database using the provided tools (MCP). "
                "DO NOT generate free-form responses. Always attempt a tool call first if the task requires data retrieval or update. "
                "If a tool call is successful, summarize the data returned for the next agent."
            ),
            tools=data_tools
        )

class SupportAgent(BaseAgent):
    def __init__(self):
        # We pass ticket-related tools and data history access
        support_tools = [MCP_TOOLS['create_ticket'], MCP_TOOLS['get_customer_history'], MCP_TOOLS['get_customer']]
        super().__init__(
            name="SupportAgent (Specialist)",
            system_prompt=(
                "You are the Support Agent. Your job is to handle customer support, provide solutions, and create/check tickets. "
                "Use the available tools to complete the task. You are the final customer-facing agent."
                "Always be polite, professional, and directly address the customer's query using the context provided."
            ),
            tools=support_tools
        )