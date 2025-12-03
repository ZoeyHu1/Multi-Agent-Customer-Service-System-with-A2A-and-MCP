# mcp_server.py
# Implements the Model Context Protocol (MCP) tools for agent use.

import sqlite3
import json
from typing import List, Dict, Any, Optional

# IMPORTANT: Database file name must match the one used in database_setup.py
DATABASE_NAME = 'support.db'

def _execute_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Helper to execute a read query and return results as a list of dicts."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
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
    """
    query = "SELECT id, name, email, phone, status FROM customers WHERE id = ?"
    result = _execute_query(query, (customer_id,))
    if not result:
        return f"ERROR: Customer with ID {customer_id} not found."
    return json.dumps(result[0])

def list_customers(status: Optional[str] = None, limit: int = 100) -> str:
    """
    Lists customers based on status ('active', 'disabled').
    """
    query = "SELECT id, name, email, status FROM customers WHERE 1=1"
    params = []
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    query += f" LIMIT {limit}"
    
    results = _execute_query(query, tuple(params))
    if not results:
        return "No customers found matching the criteria."
    return json.dumps(results)

def update_customer(customer_id: int, data: Dict[str, Any]) -> str:
    """
    Updates a customer record with new data (email, phone, status).
    """
    updates = []
    params = []
    for key, value in data.items():
        if key in ('name', 'email', 'phone', 'status'):
            updates.append(f"{key} = ?")
            params.append(value)
    
    if not updates:
        return "ERROR: No valid fields provided for update."

    # NOTE: The database_setup.py uses an UPDATE trigger for updated_at, 
    # so we don't manually set it here.
    params.append(customer_id)
    
    query = f"UPDATE customers SET {', '.join(updates)} WHERE id = ?"
    count = _execute_update(query, tuple(params))
    
    if count == 0:
        return f"ERROR: Customer with ID {customer_id} not found for update."
    return f"Customer ID {customer_id} successfully updated with new data."

def create_ticket(customer_id: int, issue: str, priority: str) -> str:
    """
    Creates a new support ticket. Priority must be 'low', 'medium', or 'high'.
    """
    query = """
        INSERT INTO tickets (customer_id, issue, status, priority)
        VALUES (?, ?, 'open', ?)
    """
    try:
        count = _execute_update(query, (customer_id, issue, priority.lower()))
        if count > 0:
            # We assume the ID is the last inserted row ID for the ticket
            conn = sqlite3.connect(DATABASE_NAME)
            ticket_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.close()
            return f"SUCCESS: New '{priority}' priority ticket (ID: {ticket_id}) created for customer ID {customer_id}."
        return "ERROR: Could not create ticket."
    except sqlite3.IntegrityError:
        return f"ERROR: Foreign Key constraint failed. Customer ID {customer_id} may not exist."

def get_customer_history(customer_id: int, status: Optional[str] = None) -> str:
    """
    Retrieves all tickets associated with a customer ID, optionally filtered by status.
    """
    query = "SELECT id, issue, status, priority, created_at FROM tickets WHERE customer_id = ?"
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
    # Simple test cases for the MCP tools (Requires support.db to exist)
    print("--- Testing MCP Tools ---")
    print("Customer 1 Info (ID 1):", get_customer(1))
    print("Active Customers:", list_customers(status='active', limit=2))
    print("Customer 2 History:", get_customer_history(2))
    print("Update Customer 1 Email:", update_customer(1, {'email': 'john.new@test.com'}))
    print("New Ticket for Customer 3:", create_ticket(3, "Account login issue", "high"))
    print("Customer 3 History (Updated):", get_customer_history(3))