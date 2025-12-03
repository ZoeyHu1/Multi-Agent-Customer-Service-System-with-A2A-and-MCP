# mcp_server.py

import sqlite3
from datetime import datetime
import json
from typing import List, Dict, Any, Optional

DATABASE_NAME = 'customer_service.db'

def _execute_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Helper to execute a read query and return results as a list of dicts."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def _execute_update(query: str, params: tuple = ()) -> int:
    """Helper to execute an update query and return the row count."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count

# --- MCP Tools Implementation ---

def get_customer(customer_id: int) -> str:
    """
    Retrieves a single customer's information by ID.
    Required Tool: get_customer(customer_id)
    """
    query = "SELECT * FROM customers WHERE id = ?"
    result = _execute_query(query, (customer_id,))
    if not result:
        return f"ERROR: Customer with ID {customer_id} not found."
    return json.dumps(result[0])

def list_customers(status: Optional[str] = None, tier: Optional[str] = None, limit: int = 100) -> str:
    """
    Lists customers based on status ('active', 'disabled') and/or tier ('standard', 'premium').
    Required Tool: list_customers(status, limit) - expanded for tier.
    """
    query = "SELECT id, name, email, status, tier FROM customers WHERE 1=1"
    params = []
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    if tier:
        query += " AND tier = ?"
        params.append(tier)
    
    query += f" LIMIT {limit}"
    
    results = _execute_query(query, tuple(params))
    if not results:
        return "No customers found matching the criteria."
    return json.dumps(results)

def update_customer(customer_id: int, data: Dict[str, Any]) -> str:
    """
    Updates a customer record with new data. Data keys can include: email, phone, status.
    Required Tool: update_customer(customer_id, data)
    """
    updates = []
    params = []
    for key, value in data.items():
        if key in ('name', 'email', 'phone', 'status', 'tier'):
            updates.append(f"{key} = ?")
            params.append(value)
    
    if not updates:
        return "ERROR: No valid fields provided for update."

    params.append(datetime.now().isoformat())
    params.append(customer_id)
    
    query = f"UPDATE customers SET {', '.join(updates)}, updated_at = ? WHERE id = ?"
    count = _execute_update(query, tuple(params))
    
    if count == 0:
        return f"ERROR: Customer with ID {customer_id} not found for update."
    return f"Customer ID {customer_id} successfully updated with new data."

def create_ticket(customer_id: int, issue: str, priority: str) -> str:
    """
    Creates a new support ticket. Priority must be 'low', 'medium', or 'high'.
    Required Tool: create_ticket(customer_id, issue, priority)
    """
    now = datetime.now().isoformat()
    query = """
        INSERT INTO tickets (customer_id, issue, status, priority, created_at)
        VALUES (?, ?, 'open', ?, ?)
    """
    try:
        count = _execute_update(query, (customer_id, issue, priority.lower(), now))
        if count > 0:
            return f"SUCCESS: New '{priority}' priority ticket created for customer ID {customer_id}."
        return "ERROR: Could not create ticket."
    except sqlite3.IntegrityError:
        return f"ERROR: Foreign Key constraint failed. Customer ID {customer_id} may not exist."

def get_customer_history(customer_id: int, status: Optional[str] = None) -> str:
    """
    Retrieves all tickets associated with a customer ID, optionally filtered by status.
    Required Tool: get_customer_history(customer_id)
    """
    query = "SELECT * FROM tickets WHERE customer_id = ?"
    params = [customer_id]
    
    if status:
        query += " AND status = ?"
        params.append(status)
        
    results = _execute_query(query, tuple(params))
    
    if not results:
        return f"No ticket history found for customer ID {customer_id}."
    return json.dumps(results)

# A dictionary to easily map function names to the callable tools
MCP_TOOLS = {
    'get_customer': get_customer,
    'list_customers': list_customers,
    'update_customer': update_customer,
    'create_ticket': create_ticket,
    'get_customer_history': get_customer_history,
}

if __name__ == '__main__':
    # Simple test cases for the MCP tools
    print("--- Testing MCP Tools ---")
    print("Customer 12345:", get_customer(12345))
    print("Active Customers:", list_customers(status='active', limit=2))
    print("Customer 12345 History:", get_customer_history(12345))
    print("Update Customer 5 Email:", update_customer(5, {'email': 'new_bob_email@test.com'}))
    print("New Customer 5 Data:", get_customer(5))
    print("Create New Ticket for 12345:", create_ticket(12345, "Need a new password.", "medium"))
    print("Customer 12345 History (Updated):", get_customer_history(12345))