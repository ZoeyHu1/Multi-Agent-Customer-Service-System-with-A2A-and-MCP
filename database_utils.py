import sqlite3
import json
from typing import List, Dict, Any, Optional

# IMPORTANT: Database file name must match the one used in database_setup.py
DATABASE_NAME = 'support.db'

class CustomerServiceDB:
    """Utility class for all database interactions (Used by MCP Server)."""

    def __init__(self):
        # We handle connection setup here, which FastMCP will instantiate once.
        # Allow the FastAPI worker threads spawned by FastMCP to reuse this connection.
        self.conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cursor = self.conn.cursor()

    # --- REQUIRED MCP Tool Implementations ---

    def get_customer(self, customer_id: int) -> str:
        """Retrieves a single customer's information by ID."""
        query = "SELECT id, name, email, phone, status, updated_at FROM customers WHERE id = ?"
        result = [dict(row) for row in self.cursor.execute(query, (customer_id,)).fetchall()]
        if not result:
            return f"ERROR: Customer with ID {customer_id} not found."
        return json.dumps(result[0])

    def list_customers(self, status: Optional[str] = None, limit: int = 100) -> str:
        """Lists customers based on status ('active', 'disabled')."""
        query = "SELECT id, name, email, status FROM customers WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        query += f" LIMIT {limit}"
        results = [dict(row) for row in self.cursor.execute(query, tuple(params)).fetchall()]
        if not results:
            return "No customers found matching the criteria."
        return json.dumps(results)

    def update_customer(self, customer_id: int, data: Dict[str, Any]) -> str:
        """Updates a customer record with new data (email, phone, status)."""
        updates = []
        params = []
        for key, value in data.items():
            if key in ('name', 'email', 'phone', 'status'):
                updates.append(f"{key} = ?")
                params.append(value)
        
        if not updates:
            return "ERROR: No valid fields provided for update."

        params.append(customer_id)
        query = f"UPDATE customers SET {', '.join(updates)} WHERE id = ?"
        self.cursor.execute(query, tuple(params))
        self.conn.commit()
        
        if self.cursor.rowcount == 0:
            return f"ERROR: Customer with ID {customer_id} not found for update."
        return f"SUCCESS: Customer ID {customer_id} updated."

    def create_ticket(self, customer_id: int, issue: str, priority: str) -> str:
        """Creates a new support ticket. Priority must be 'low', 'medium', or 'high'."""
        query = """
            INSERT INTO tickets (customer_id, issue, status, priority)
            VALUES (?, ?, 'open', ?)
        """
        try:
            self.cursor.execute(query, (customer_id, issue, priority.lower()))
            self.conn.commit()
            ticket_id = self.cursor.lastrowid
            return f"SUCCESS: New '{priority}' priority ticket (ID: {ticket_id}) created for customer ID {customer_id}."
        except sqlite3.IntegrityError as e:
            return f"ERROR: Foreign Key constraint failed. Customer ID {customer_id} may not exist. Details: {e}"

    def get_customer_history(self, customer_id: int, status: Optional[str] = None) -> str:
        """Retrieves all tickets associated with a customer ID, optionally filtered by status."""
        query = "SELECT id, issue, status, priority, created_at FROM tickets WHERE customer_id = ?"
        params = [customer_id]
        if status:
            query += " AND status = ?"
            params.append(status)
            
        results = [dict(row) for row in self.cursor.execute(query, tuple(params)).fetchall()]
        
        if not results:
            return f"No ticket history found for customer ID {customer_id}."
        return json.dumps(results)
    
    # --- Custom Tool for Multi-Step Coordination ---
    def get_high_priority_tickets_by_ids(self, customer_ids: List[int]) -> str:
        """Custom tool for Scenario 3: Gets high-priority tickets for a list of IDs."""
        if not customer_ids:
            return "No customer IDs provided to check."
        
        # Create placeholders for the IN clause
        placeholders = ', '.join('?' for _ in customer_ids)
        query = f"""
            SELECT t.id, c.name, t.issue, t.status, t.priority 
            FROM tickets t
            JOIN customers c ON t.customer_id = c.id
            WHERE t.customer_id IN ({placeholders}) AND t.priority = 'high'
        """
        results = [dict(row) for row in self.cursor.execute(query, tuple(customer_ids)).fetchall()]

        if not results:
            return "No high-priority tickets found for the specified active customers."
        return json.dumps(results)

    def list_active_customers_with_open_tickets(self) -> str:
        """Return active customers who currently have open tickets."""
        query = """
            SELECT c.id, c.name, c.email, c.phone, c.status,
                   COUNT(t.id) as open_ticket_count
            FROM customers c
            JOIN tickets t ON c.id = t.customer_id
            WHERE c.status = 'active' AND t.status = 'open'
            GROUP BY c.id, c.name, c.email, c.phone, c.status
            ORDER BY open_ticket_count DESC, c.name
        """
        results = [dict(row) for row in self.cursor.execute(query).fetchall()]
        if not results:
            return "No active customers currently have open tickets."
        return json.dumps(results)

    def close(self):
        """Close database connection (called automatically on process exit)."""
        if self.conn:
            self.conn.close()
