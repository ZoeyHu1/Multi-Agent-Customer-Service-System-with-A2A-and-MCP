# system_orchestrator.py - FINAL IMPROVEMENT (Robust Scenario 3 Coordination)

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from database_setup import DatabaseSetup
from agents import RouterAgent, CustomerDataAgent, SupportAgent
import json

# --- 1. Define the Shared State (LangGraph Message Passing) ---
class AgentState(TypedDict):
    """
    Represents the shared state for the LangGraph.
    """
    query: str                       # Original customer query
    context: Annotated[str, lambda x, y: y + "\n" + x]  
    task: str                        
    next_agent: str                  
    final_answer: str                
    agent_name: str                  
    last_output: str                 
    error: str                       


# --- Database Setup (unchanged) ---
db_setup = DatabaseSetup("support.db")
db_setup.connect()
db_setup.create_tables()
db_setup.create_triggers()
db_setup.insert_sample_data() 
db_setup.close()
print("\n--- Database Setup Complete for Orchestrator ---")


# --- 2. Define Agent Nodes ---

router_agent = RouterAgent()
data_agent = CustomerDataAgent()
support_agent = SupportAgent()

def run_router(state: AgentState) -> AgentState:
    """Router Agent node."""
    new_state = router_agent.run(state)
    if new_state.get('error'):
        return {**new_state, 'next_agent': 'END'}
    return new_state

def run_data_agent(state: AgentState) -> AgentState:
    """Customer Data Agent node. Contains the crucial A2A routing logic."""
    new_state = data_agent.run(state)
    
    # A2A Transfer: Append the data agent's result to the running context
    new_state['context'] = f"[Data Agent Output from {new_state['agent_name']}]: {new_state['last_output']}"
    
    # Logic for routing to support after data fetch (Scenario 1)
    if state.get('next_agent') == 'data_agent_then_support':
        new_state['next_agent'] = 'support_agent'
        new_state['task'] = 'Use the fetched customer data to handle the support request.'
    
    # --- FIX for Scenario 3 Coordination ---
    # If the task explicitly involves listing customers for later filtering, force the transition.
    elif 'retrieve all active customers' in state['task'].lower():
        # The Data Agent has performed its first step (list_customers). Force transition to Support Agent.
        new_state['next_agent'] = 'support_agent'
        # Provide the next specific instruction to the Support Agent
        new_state['task'] = f"The previous step returned a list of customer IDs in the context. Now, use your tools to find all **high-priority** tickets for those IDs and summarize the status."
        print("\n**LOG: FORCING ROUTE from Data Agent to Support Agent for ticket filtering (Scenario 3)**")
    # --- END FIX ---
            
    elif 'error' in new_state or 'ERROR' in new_state.get('last_output', ''):
        new_state['final_answer'] = f"System Error during Data Fetch: {new_state.get('last_output')}"
        new_state['next_agent'] = 'END'
    else:
        # Pure data query is done (e.g., "Get customer info")
        new_state['final_answer'] = f"Data Retrieval Complete: {new_state['last_output']}"
        new_state['next_agent'] = 'END'
        
    return new_state

def run_support_agent(state: AgentState) -> AgentState:
    """Support Agent node."""
    new_state = support_agent.run(state)
    
    # Support agent provides the final answer
    new_state['final_answer'] = new_state['last_output']
    new_state['next_agent'] = 'END'
    return new_state

# --- 3. Routing Function (unchanged) ---
def route_agents(state: AgentState) -> str:
    """
    Determines the next node (agent) based on the 'next_agent' key.
    """
    next_agent = state.get('next_agent')
    print(f"\n*** A2A Coordination: Router/Agent decided next step: {next_agent} ***")
    
    if next_agent in ['data_agent', 'data_agent_then_support']:
        return 'data'
    elif next_agent == 'support_agent':
        return 'support'
    elif next_agent == 'END':
        return END
    return END


# --- 4. Build the LangGraph (unchanged) ---
workflow = StateGraph(AgentState)

workflow.add_node("router", run_router)
workflow.add_node("data", run_data_agent)
workflow.add_node("support", run_support_agent)

workflow.set_entry_point("router")

workflow.add_conditional_edges("router", route_agents, {"data": "data", "support": "support", END: END})
workflow.add_conditional_edges("data", route_agents, {"support": "support", END: END})
workflow.add_edge("support", END)

app = workflow.compile()

# --- 5. Test Scenarios (unchanged) ---

def run_scenario(query: str, scenario_name: str):
    """Executes a single customer query through the agent system."""
    print("\n" + "="*80)
    print(f"🚀 SCENARIO: {scenario_name}")
    print(f"📝 Query: {query}")
    print("="*80)

    initial_state = AgentState(
        query=query, context="", task="Analyze query intent and route.", 
        next_agent="", final_answer="", agent_name="", last_output="", error=""
    )
    
    for step in app.stream(initial_state):
        if step:
            node, state = list(step.items())[0]
            if node != 'router':
                print(f"\n[STEP COMPLETED]: {node.upper()}")
            
            if state.get('agent_name') and state.get('last_output'):
                 print(f"**A2A LOG: {state['agent_name']} -> Next Node: {state.get('next_agent', 'END')}**")
                 print(f"**A2A LOG: Task for Next Node: {state.get('task', 'N/A')}**")
                 print(f"**A2A LOG: Information Passed (abridged): {state['last_output'][:150]}...**")
            
            if state.get('final_answer'):
                print("\n\n#####################################################")
                print("✅ FINAL CUSTOMER RESPONSE:")
                print(state['final_answer'])
                print("#####################################################")
                return


if __name__ == '__main__':
    
    # Scenario 1: Task Allocation 
    run_scenario(
        query="I need help with my account, customer ID 1. Can you help me with a password reset issue?",
        scenario_name="Scenario 1: Task Allocation (Data Fetch + Support)"
    )

    # Scenario 3: Multi-Step Coordination (FIXED to transition correctly)
    run_scenario(
        query="What's the status of all high-priority tickets for active customers?",
        scenario_name="Scenario 3: Multi-Step Coordination (List Customers + Filter Tickets)"
    )

    # Scenario 2/Escalation 
    run_scenario(
        query="I've been charged twice, please refund immediately! I'm customer ID 4.",
        scenario_name="Scenario 2/Escalation: Multi-Intent (Create High Priority Ticket)"
    )

    # Test Scenario: Multi-Intent (Update Customer Data + Get History)
    run_scenario(
        query="Update my email to bob.new@business.net (ID 3) and show my ticket history",
        scenario_name="Test: Multi-Intent (Update Customer Data + Get History)"
    )