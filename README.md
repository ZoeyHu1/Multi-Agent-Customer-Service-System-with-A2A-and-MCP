# Customer Support MCP Agents

This project implements a multi-agent customer support workflow that talks to a real database through the Model Context Protocol (MCP). It contains:

- A Flask-based MCP server (`mcp_server.py`) that exposes customer/ticket tools over HTTP + Server-Sent Events.
- Google ADK + A2A agents (`adk_a2a_system.py`, `a2a_mcp_tools.py`) that spin up three servers: a Customer Data agent, a Support agent, and a Router/orchestrator agent. The Router uses `RemoteA2aAgent` transfers (Router → Data → Router → Support → Router) so negotiation and multi-step coordination stay under router control.
- SQLite utilities (`database_setup.py`, `database_utils.py`) and sample data.

The ADK agents call the MCP server via custom BaseTool implementations, emit JSON hand-offs, and expose their `/a2a/{assistant}` endpoints just like the official Google ADK tutorials.

## Requirements

- Python 3.10
- Google Gemini API key (export `GOOGLE_API_KEY`)
- ngrok account (to expose the MCP server)

Create and activate a virtual environment, then install dependencies:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Database Setup

Run once to create `support.db` and seed sample customers/tickets:

```bash
python database_setup.py
```

## Running the MCP Server

The MCP server must be reachable by Gemini. Start it locally and expose with ngrok:

```bash
# Terminal 1: start the server
cd HW5
python mcp_server.py
```

```bash
# Terminal 2: tunnel HTTP traffic
ngrok http 8000
```

Copy the HTTPS forwarding URL (for example `https://abc123.ngrok-free.app`) and export `MCP_SERVER_URL` so the ADK runtime references the public endpoint (the script automatically appends `/mcp`):

```bash
export MCP_SERVER_URL="https://abc123.ngrok-free.app"
```

## Running the ADK A2A System

Export your keys, ensure `MCP_SERVER_URL` points at your ngrok tunnel, and start the ADK servers + demo client:

```bash
export GOOGLE_API_KEY="YOUR_KEY"
export MCP_SERVER_URL="https://abc123.ngrok-free.app"
python adk_a2a_system.py
```

This script:
1. Launches three ADK/A2A servers (data, support, router) on `http://127.0.0.1:1003x`.
2. Streams each agent’s A2A card at `/.well-known/agent-card`.
3. Sends the homework scenarios to the router via the official A2A client so you can watch the Router delegate work, negotiate when support needs more context, and return the final customer answer.

### Capturing A2A Logs

Every request prints a structured “handoff log” so graders can see each transfer:

```
[A2A] Router → Intent: billing_escalation | customer_id=4
[A2A] Router → CustomerData: Task: Provide billing history ...
[A2A] CustomerData → Router: Fetched ticket history for customer 4.
[A2A] Router → Support: ... Context payload (if any): {...}
[A2A] Support → Router: Needs context
[A2A] Router → CustomerData: ...
[A2A] Support → Router: Final customer response JSON
```

Copy/paste these logs (or redirect stdout to a file) when you need to demonstrate a specific scenario. The router enforces JSON hand-offs—if a specialist replies with plain text, the router retries with extra instructions and, as a last resort, coerces the text into the required schema so the orchestration loop always stays deterministic.

### Scenarios Exercised

The default `SCENARIOS` list in `adk_a2a_system.py` covers:

1. Simple lookup (“Get customer information for ID 5”)
2. Upgrade coordination (“I'm customer 3 and need help upgrading my account”)
3. High-priority report (“Show me all active customers who have open tickets”)
4. Multi-intent workflow (“My customer ID is 4. Update my email … and show my ticket history”)
5. Escalation/refund (“I've been charged twice, please refund immediately!”)

Add or modify entries in the `SCENARIOS` list to replay other homework prompts (e.g., substitute customer ID 12345) and capture the logs above as evidence of Router → Data → Support → Router coordination.

## Colab Notebook

The repository also includes `agent_demo.ipynb`, which automates the full flow (install deps, start MCP server in the background, run the orchestrator, and shut everything down). Upload it to Google Colab if you prefer a notebook-based demonstration.

## Repository Structure

```
HW5/
├── a2a_mcp_tools.py         # Google ADK tool wrappers around MCP endpoints
├── adk_a2a_system.py        # Starts the ADK A2A servers and runs demo scenarios
├── database_setup.py        # SQLite schema + sample data installer
├── database_utils.py        # DB helper class consumed by MCP server
├── mcp_server.py            # Flask SSE MCP server
├── agent_demo.ipynb         # End-to-end Colab/Notebook demonstration
├── requirements.txt         # Python dependencies
└── README.md
```

## Deployment Notes

- Keep `mcp_server.py` running as long as the agents need tool access.
- If you restart ngrok, update `MCP_SERVER_URL` accordingly.
- Gemini free tier has strict per-minute and per-day limits; you may need to wait for quota resets during heavy testing.

Enjoy building MCP-powered, ADK-backed customer support flows!
