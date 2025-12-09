from typing import TypedDict, Annotated, List, Dict, Any, Optional
import re
from langgraph.graph import StateGraph, END
from agents import RouterAgent, CustomerDataAgent, SupportAgent, MCP_SERVER_URL, client
import json
import os
import time

# --- 1. Define the Shared State (LangGraph Message Passing) ---
class AgentState(TypedDict, total=False):
    """Represents the shared state for the LangGraph."""
    query: str                       
    context: Annotated[str, lambda x, y: y + "\n" + x]  
    task: str                        
    next_agent: str                  
    final_answer: str                
    agent_name: str                  
    last_output: str                 
    error: str                       
    customer_id: Optional[int]
    history_customer_id: Optional[int]


# --- 2. Define Agent Nodes (Routing logic remains the same) ---

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
    """Customer Data Agent node."""
    new_state = data_agent.run(state)
    
    # A2A Transfer: Append the data agent's result to the running context
    previous_context = state.get('context', '')
    data_summary = f"[Data Agent Output from {new_state['agent_name']}]: {new_state['last_output']}"
    if previous_context:
        new_state['context'] = previous_context + "\n" + data_summary
    else:
        new_state['context'] = data_summary
    
    # Logic for routing to support after data fetch (Scenario 1)
    if state.get('next_agent') == 'data_agent_then_support':
        new_state['next_agent'] = 'support_agent'
        customer_id = state.get('customer_id')
        if customer_id:
            new_state['history_customer_id'] = customer_id
            new_state['task'] = (
                f"Customer {customer_id} has been updated. Call 'get_customer_history' "
                f"for customer_id={customer_id} and summarize the results."
            )
        else:
            new_state['task'] = 'Use the fetched customer data to handle the support request.'
    
    # FIX for Scenario 3 Coordination (Data Agent finishes its part, hands off)
    elif 'list_customers' in state['task'].lower():
        # The Data Agent has performed its first step (list_customers). Force transition to Support Agent.
        new_state['next_agent'] = 'support_agent'
        # Provide the next specific instruction to the Support Agent
        new_state['task'] = "The previous step returned a list of active customer IDs in the context. Now, use the 'get_high_priority_tickets_by_ids' tool to filter and summarize."
        print("\n**LOG: FORCING ROUTE from Data Agent to Support Agent for ticket filtering (Scenario 3)**")
            
    elif new_state.get('error') or 'ERROR' in (new_state.get('last_output') or ''):
        new_state['final_answer'] = f"System Error during Data Fetch: {new_state.get('last_output')}"
        new_state['next_agent'] = 'END'
    else:
        # Pure data query is done
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

# --- 3. Routing Function and LangGraph Setup (unchanged) ---
def route_agents(state: AgentState) -> str:
    """Determines the next node (agent) based on the 'next_agent' key."""
    next_agent = state.get('next_agent')
    print(f"\n*** A2A Coordination: Router/Agent decided next step: {next_agent} ***")
    
    if next_agent in ['data_agent', 'data_agent_then_support']:
        return 'data'
    elif next_agent == 'support_agent':
        return 'support'
    elif next_agent == 'END':
        return END
    return END

workflow = StateGraph(AgentState)
workflow.add_node("router", run_router)
workflow.add_node("data", run_data_agent)
workflow.add_node("support", run_support_agent)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_agents, {"data": "data", "support": "support", END: END})
workflow.add_conditional_edges("data", route_agents, {"support": "support", END: END})
workflow.add_edge("support", END)
app = workflow.compile()

# --- 4. Test Scenarios ---

def run_scenario(query: str, scenario_name: str):
    """Executes a single customer query through the agent system."""
    if not client:
        print(f"Skipping scenario '{scenario_name}'. Client is not initialized.")
        return

    print("\n" + "="*80)
    print(f" SCENARIO: {scenario_name}")
    print(f" Query: {query}")
    print("="*80)

    initial_state = AgentState(
        query=query, context="", task="Analyze query intent and route.", 
        next_agent="", final_answer="", agent_name="", last_output="", error=""
    )
    # Extract customer ID early for downstream instructions
    match = re.search(r"(?:customer\s*id|id)\D*(\d+)", query, re.IGNORECASE)
    if match:
        try:
            initial_state["customer_id"] = int(match.group(1))
        except ValueError:
            pass
    
    for step in app.stream(initial_state):
        if step:
            node, state = list(step.items())[0]
            if node != 'router':
                print(f"\n[STEP COMPLETED]: {node.upper()}")
            
            if state.get('agent_name'):
                 print(f"**A2A LOG: {state['agent_name']} -> Next Node: {state.get('next_agent', 'END')}**")
                 print(f"**A2A LOG: Task for Next Node: {state.get('task', 'N/A')}**")
            
            if state.get('final_answer'):
                print("\n\n#####################################################")
                print(" FINAL CUSTOMER RESPONSE:")
                print(state['final_answer'])
                print("#####################################################")
                return


if __name__ == '__main__':
    time.sleep(2) 
    
    # 3 scenarios showing A2A coordination
    run_scenario(
        query="I need help with my account, customer ID 2. Can you check my details and assist?",
        scenario_name="Scenario 1: Task Allocation (Data Fetch + Support)"
    )

    run_scenario(
        query="My customer ID is 3. I want to cancel my subscription but I'm having billing issues",
        scenario_name="Scenario 2: Negotiation/Escalation"
    )

    # run_scenario(
    #     query="What's the status of all high-priority tickets for premium customers?",
    #     scenario_name="Scenario 3: Multi-Step Coordination"
    # )

    # test scenarios
    run_scenario(
        query="Get customer information for ID 5",
        scenario_name="Test: Simple Query"
    )

    run_scenario(
        query="I'm customer 3 and need help upgrading my account",
        scenario_name="Test: Coordinated Query"
    )

    run_scenario(
        query="Show me all active customers who have open tickets",
        scenario_name="Test: Complex Query"
    )

    # run_scenario(
    #     query="I've been charged twice, please refund immediately!",
    #     scenario_name="Test: Escalation"
    # )

    run_scenario(
        query="My customer ID is 4. Update my email to new@email.com and show my ticket history",
        scenario_name="Test: Multi-Intent"
    )
