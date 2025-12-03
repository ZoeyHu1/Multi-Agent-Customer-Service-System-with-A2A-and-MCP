# system_orchestrator.py

import os
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from agents import RouterAgent, CustomerDataAgent, SupportAgent
from setup_database import setup_database

# --- 1. Define the Shared State (LangGraph Message Passing) ---
class AgentState(TypedDict):
    """
    Represents the shared state for the LangGraph.
    """
    query: str                       # Original customer query
    context: Annotated[str, lambda x, y: y + "\n" + x]  # Running log of information passed between agents
    task: str                        # The current sub-task determined by the Router
    next_agent: str                  # The name of the next node to execute
    final_answer: str                # The system's final response to the customer
    agent_name: str                  # The last agent to execute
    last_output: str                 # The raw output from the last agent's execution
    error: str                       # Optional error message


# --- 2. Define the Agent Nodes and Conditional Edges ---

# Initialize agents and run database setup
setup_database()
router_agent = RouterAgent()
data_agent = CustomerDataAgent()
support_agent = SupportAgent()

def run_router(state: AgentState) -> AgentState:
    """Router Agent node."""
    new_state = router_agent.run(state)
    # The router's main output is setting 'next_agent' and 'task'
    if new_state.get('error'):
        return {**new_state, 'next_agent': 'END'}
    return new_state

def run_data_agent(state: AgentState) -> AgentState:
    """Customer Data Agent node."""
    new_state = data_agent.run(state)
    # Append the data agent's result to the running context
    new_state['context'] += f"\n[Data Agent Output]: {new_state['last_output']}"
    
    # Simple logic for determining the next step after data retrieval
    if 'support' in state['task'].lower() or state.get('next_agent') == 'data_agent_then_support':
        new_state['next_agent'] = 'support_agent'
        new_state['task'] = 'Use the fetched data to formulate a customer support response.'
    elif 'error' in new_state or 'ERROR' in new_state.get('last_output', ''):
        new_state['next_agent'] = 'router_agent' # Go back to router for error handling
    else:
        # If the query was purely for data (e.g., "get customer info"), it's done
        new_state['final_answer'] = f"Data Retrieval Complete: {new_state['last_output']}"
        new_state['next_agent'] = 'END'
        
    return new_state

def run_support_agent(state: AgentState) -> AgentState:
    """Support Agent node."""
    new_state = support_agent.run(state)
    
    # Support agent has the final word, its output is the final answer
    new_state['final_answer'] = new_state['last_output']
    new_state['next_agent'] = 'END'
    return new_state

def route_agents(state: AgentState):
    """Conditional edge logic based on the Router's decision."""
    agent_name = state['next_agent']
    print(f"\n*** A2A Coordination: Transferring control to {agent_name} ***")
    
    if agent_name == 'data_agent':
        return 'data'
    elif agent_name == 'data_agent_then_support':
        return 'data' # Start with data agent
    elif agent_name == 'support_agent':
        return 'support'
    elif agent_name == 'END':
        return END
    else:
        # Fallback to the router if an agent can't decide or for negotiation/error
        print("Falling back to Router for negotiation/re-routing.")
        return 'router'


# --- 3. Build the LangGraph ---
workflow = StateGraph(AgentState)

# Add nodes (Agent execution functions)
workflow.add_node("router", run_router)
workflow.add_node("data", run_data_agent)
workflow.add_node("support", run_support_agent)

# Set the entry point
workflow.set_entry_point("router")

# Define edges (Flow control)
# Router always routes based on its decision
workflow.add_edge("router", "data", conditional=lambda state: state['next_agent'] in ['data_agent', 'data_agent_then_support'])
workflow.add_edge("router", "support", conditional=lambda state: state['next_agent'] == 'support_agent')
workflow.add_edge("router", END, conditional=lambda state: state['next_agent'] == 'END')

# Data agent decides if it needs to go to support or end
workflow.add_edge("data", "support", conditional=lambda state: state['next_agent'] == 'support_agent')
workflow.add_edge("data", END, conditional=lambda state: state['next_agent'] == 'END')
# Note: For complex negotiation, we might add an edge from 'data' back to 'router'.

# Support agent is the end of the line
workflow.add_edge("support", END)

# Compile the graph
app = workflow.compile()

# --- 4. Test Scenarios ---

def run_scenario(query: str, scenario_name: str):
    """Executes a single customer query through the agent system."""
    print("="*80)
    print(f"🚀 SCENARIO: {scenario_name}")
    print(f"📝 Query: {query}")
    print("="*80)

    # Initial State
    initial_state = AgentState(
        query=query, 
        context="", 
        task="Analyze query intent and route.", 
        next_agent="", 
        final_answer="", 
        agent_name="",
        last_output="",
        error=""
    )
    
    # Run the graph
    # We loop through the steps to see the A2A coordination clearly
    for step in app.stream(initial_state):
        if step:
            node, state = list(step.items())[0]
            print(f"\n[STEP COMPLETED]: {node}")
            
            # Log Agent-to-Agent Communication (A2A)
            if state.get('agent_name') and state.get('last_output'):
                 print(f"**A2A LOG: {state['agent_name']} -> Next Node: {state.get('next_agent', 'END')}**")
                 print(f"**A2A LOG: Task for Next Node: {state.get('task', 'N/A')}**")
                 print(f"**A2A LOG: Information Passed: {state['last_output'][:150]}...**")
            
            if state.get('final_answer'):
                print("\n\n#####################################################")
                print("✅ FINAL CUSTOMER RESPONSE:")
                print(state['final_answer'])
                print("#####################################################")
                return

    print("\n\n#####################################################")
    print("❌ SYSTEM ERROR: Flow did not reach an END state.")
    print("#####################################################")


if __name__ == '__main__':
    
    # Scenario 1: Task Allocation (Router -> Data Agent -> Support Agent)
    run_scenario(
        query="I'm customer 12345 and need help upgrading my account.",
        scenario_name="Scenario 1: Task Allocation (Data Fetch + Support)"
    )

    # Scenario 3: Multi-Step Coordination (Data Agent -> Data Agent tools multiple times, then Support to filter)
    run_scenario(
        query="What's the status of all high-priority tickets for premium customers?",
        scenario_name="Scenario 3: Multi-Step Coordination (Data for list, Support for history)"
    )

    # Coordinated Query: Negotiation/Escalation (Router -> Support Agent -> Data Agent -> Support Agent)
    # Simplified: Router routes to Support, Support uses its get_customer tool and creates a ticket.
    run_scenario(
        query="I've been charged twice, please refund immediately! My ID is 12345.",
        scenario_name="Scenario 2/Escalation: Multi-Intent (Support Handles Ticket Creation)"
    )

    # Simple Query: Single agent, straightforward MCP call
    run_scenario(
        query="Get customer information for ID 5.",
        scenario_name="Test Scenario: Simple Data Fetch"
    )