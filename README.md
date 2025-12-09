# Customer Support MCP Agents

This project implements a multi-agent customer support workflow that talks to a real database through the Model Context Protocol (MCP). It contains:

- A Flask-based MCP server (`mcp_server.py`) that exposes customer/ticket tools over HTTP + Server-Sent Events.
- LangGraph agents (`agents.py`, `system_orchestrator.py`) that orchestrate Router → Data → Support flows using Gemini 2.5 Flash.
- SQLite utilities (`database_setup.py`, `database_utils.py`) and sample data.

The code mirrors the Google ADK MCP tutorials but is packaged as a standalone Python project.

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

Copy the HTTPS forwarding URL (for example `https://abc123.ngrok-free.app`) and set `MCP_SERVER_URL` in `agents.py` / environment vars so the agents reference the public endpoint (remember to append `/mcp`).

## Running the LangGraph Orchestrator

Export your Gemini API key and run the orchestrator script to execute all demo scenarios end-to-end:

```bash
export GOOGLE_API_KEY="YOUR_KEY"
python system_orchestrator.py
```

The script prints each scenario, the agent hand-offs, the MCP tool calls, and the final customer-facing response.

### Test Scenarios Included

1. Task allocation (data fetch + support response)
2. Subscription cancellation + billing escalation
3. High-priority ticket coordination
4. Simple data fetch by ID
5. Active customers with open tickets (custom tool)
6. Multi-intent update + ticket history retrieval

## Colab Notebook

The repository also includes `agent_demo.ipynb`, which automates the full flow (install deps, start MCP server in the background, run the orchestrator, and shut everything down). Upload it to Google Colab if you prefer a notebook-based demonstration.

## Repository Structure

```
HW5/
├── agents.py                # Gemini agent definitions + MCP invocation helpers
├── database_setup.py        # SQLite schema + sample data installer
├── database_utils.py        # DB helper class consumed by MCP server
├── mcp_server.py            # Flask SSE MCP server
├── system_orchestrator.py   # LangGraph workflow with multiple scenarios
├── agent_demo.ipynb         # End-to-end Colab/Notebook demonstration
├── requirements.txt         # Python dependencies
└── README.md
```

## Deployment Notes

- Keep `mcp_server.py` running as long as the agents need tool access.
- If you restart ngrok, update `MCP_SERVER_URL` accordingly.
- Gemini free tier has strict per-minute and per-day limits; you may need to wait for quota resets during heavy testing.

Enjoy building MCP-powered customer support flows!
