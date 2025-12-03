# Multi-Agent Customer Service System (A2A & MCP)

##  Overview
This project implements a multi-agent customer service automation system using the Google Gemini API, coordinated via a **LangGraph** state machine (Agent-to-Agent communication), and integrated with an external data store via the **Model Context Protocol (MCP)**.

The system features three specialized agents:
1.  **Router Agent:** Directs queries based on intent.
2.  **Customer Data Agent:** Accesses customer records and history (via MCP tools).
3.  **Support Agent:** Handles solutions, creates tickets, and synthesizes final responses.

##  Setup Instructions

### 1. Prerequisites

* **Python:** Python 3.9+ installed.
* **Gemini API Key:** You must have a Gemini API key.

### 2. Environment Setup (Using venv)

It is highly recommended to use a virtual environment (`venv`) to isolate project dependencies:

```bash
# 1. Create the virtual environment
python -m venv venv 

# 2. Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows (use backslash)