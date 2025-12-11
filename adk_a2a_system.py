from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import re
from typing import Any, Dict, List, Optional

import httpx
import nest_asyncio
import uvicorn
from a2a.client import ClientConfig, ClientFactory, create_text_message_object
from a2a.client import client as real_client_module
from a2a.client.card_resolver import A2ACardResolver

class _PatchedClientModule:
    def __init__(self, real_module) -> None:
        for attr in dir(real_module):
            if not attr.startswith("_"):
                setattr(self, attr, getattr(real_module, attr))
        self.A2ACardResolver = A2ACardResolver


sys.modules["a2a.client.client"] = _PatchedClientModule(real_client_module)  # type: ignore

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TransportProtocol
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from dotenv import load_dotenv
from google.adk.a2a.executor.a2a_agent_executor import (
    A2aAgentExecutor,
    A2aAgentExecutorConfig,
)
from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from a2a_mcp_tools import customer_data_tools, support_agent_tools


def _parse_json(content: str) -> Dict[str, Any]:
    """Safely parse JSON, handling code fences."""
    if not content:
        return {}
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return {}
    return {}


def _log_handoff(sender: str, recipient: str, summary: str) -> None:
    """Structured console logging for A2A transfers."""
    clean_summary = summary.replace("\n", " ").strip()
    print(f"[A2A] {sender} → {recipient}: {clean_summary}")

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

load_dotenv()

os.environ.setdefault("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))

MODEL_NAME = os.getenv("A2A_MODEL", "gemini-2.5-pro")
HOST = os.getenv("A2A_HOST", "127.0.0.1")
DATA_AGENT_PORT = int(os.getenv("DATA_AGENT_PORT", "10030"))
SUPPORT_AGENT_PORT = int(os.getenv("SUPPORT_AGENT_PORT", "10031"))
ROUTER_AGENT_PORT = int(os.getenv("ROUTER_AGENT_PORT", "10032"))

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


def _create_agent_app(agent: Agent, agent_card: AgentCard) -> A2AStarletteApplication:
    runner = Runner(
        app_name=agent.name,
        agent=agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
    executor = A2aAgentExecutor(
        runner=runner,
        config=A2aAgentExecutorConfig(),
    )
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)


def _skills(name: str, description: str, examples: List[str]) -> List[AgentSkill]:
    return [
        AgentSkill(
            id=f"{name}_skill",
            name=name.replace("_", " ").title(),
            description=description,
            tags=["customer-support", "mcp", "internal"],
            examples=examples,
        )
    ]


def _create_customer_data_agent() -> tuple[Agent, AgentCard, int]:
    instruction = """
You are the Customer Data Specialist for a customer-support automation team.

Responsibilities:
- Use the MCP tools to fetch, list, or update customer records, tickets, and reports.
- Validate every request before calling a tool (e.g., ensure IDs/status filters are provided).
- Reject or clarify tasks that would mutate data without a customer_id.
- NEVER talk to the end-customer; you only create structured responses for the Router agent.
- After each task return STRICT JSON with this exact schema:
  {
    "handoff_summary": "<short sentence>",
    "recommended_next_agent": "router" | "support",
    "context_type": "profile" | "history" | "report" | "general",
    "context_payload": { ... raw MCP output ... },
    "notes_for_router": "<how the router should use this information>"
  }
- Copy the MCP response verbatim into context_payload so downstream agents can trust the facts.
- Cite IDs/status/timestamps exactly as shown by MCP results.
"""
    agent = Agent(
        name="customer_data_agent",
        description="Fetches and updates customer data using MCP tools.",
        model=MODEL_NAME,
        instruction=instruction,
        tools=customer_data_tools(),
    )
    card = AgentCard(
        name="Customer Data Agent",
        url=f"http://{HOST}:{DATA_AGENT_PORT}",
        description="Internal agent that interacts with the MCP database.",
        version="1.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["application/json"],
        preferred_transport=TransportProtocol.jsonrpc,
        skills=_skills(
            "customer_data",
            "Retrieve profiles, ticket history, and premium reports.",
            [
                "Lookup customer ID 5 and summarize their account.",
                "Provide the ticket history for customer 4.",
                "List all active customers that have open tickets.",
            ],
        ),
    )
    return agent, card, DATA_AGENT_PORT


def _create_support_agent() -> tuple[Agent, AgentCard, int]:
    instruction = """
You are the Support Specialist who drafts the final customer response.

Workflow:
- Use MCP tools (history, update, create_ticket) when you need concrete facts.
- ALWAYS return STRICT JSON in one of the following shapes:
  Need more info:
  {
    "requires_context": true,
    "context_type": "profile" | "history" | "billing" | "tickets",
    "notes_for_router": "Describe exactly what context/tool call is needed."
  }
  Final customer reply ready:
  {
    "requires_context": false,
    "context_type": null,
    "customer_response": "Final answer for the end customer",
    "notes_for_router": "Internal notes (optional)."
  }
- Respond ONLY with JSON. Never include natural language outside those braces. The router will reject
  non-JSON output. Use double quotes around every string.
- Examples:
  Need more data:
  {
    "requires_context": true,
    "context_type": "billing",
    "notes_for_router": "Need the customer's billing history to proceed."
  }
  Final reply:
  {
    "requires_context": false,
    "context_type": null,
    "customer_response": "Dear customer, ...",
    "notes_for_router": "Ticket #123 created with priority high."
  }
- You may call MCP tools to verify facts before responding, but still return the JSON envelope described above.
- Be professional, empathetic, and concise. Reference ticket/customer IDs you verified.
- Escalate complex cases by creating high-priority tickets using create_ticket or by asking the Router for more data.
"""
    agent = Agent(
        name="support_agent",
        description="Handles escalations and writes the customer-facing response.",
        model=MODEL_NAME,
        instruction=instruction,
        tools=support_agent_tools(),
    )
    card = AgentCard(
        name="Support Agent",
        url=f"http://{HOST}:{SUPPORT_AGENT_PORT}",
        description="Generates final customer-facing answers and escalations.",
        version="1.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["application/json"],
        preferred_transport=TransportProtocol.jsonrpc,
        skills=_skills(
            "support_response",
            "Craft empathetic responses and escalate critical issues.",
            [
                "Customer wants to cancel due to billing concerns.",
                "Issue refund for double charge.",
                "Summarize high priority tickets for premium customers.",
            ],
        ),
    )
    return agent, card, SUPPORT_AGENT_PORT


def _create_router_agent() -> tuple[Agent, AgentCard, int]:
    customer_data_card = f"http://{HOST}:{DATA_AGENT_PORT}{AGENT_CARD_WELL_KNOWN_PATH}"
    support_card = f"http://{HOST}:{SUPPORT_AGENT_PORT}{AGENT_CARD_WELL_KNOWN_PATH}"

    remote_customer_data = RemoteA2aAgent(
        name="customer_data_agent",
        description="Specialist that queries the MCP database.",
        agent_card=customer_data_card,
    )
    remote_support = RemoteA2aAgent(
        name="support_agent",
        description="Specialist that drafts customer responses.",
        agent_card=support_card,
    )

    instruction = """
You are the Router/Orchestrator for a multi-agent customer-support system.
- Receive the end-customer query, analyze intent (task allocation, escalation, or reporting), and decide the next specialist.
- You may transfer control to:
  * 'customer_data_agent' whenever you need MCP data or updates.
  * 'support_agent' when it's time to craft or revise the customer-facing reply.
- Keep transferring between specialists until the support response is complete. Always provide clear task briefs.
- Whenever the support agent indicates `requires_context=true`, immediately consult
  the customer_data_agent with a targeted request for that context, then return control
  to the support agent so they can finish.
- After support provides `requires_context=false`, synthesize the FINAL CUSTOMER RESPONSE
  yourself so the user only sees one message.
- Include concise bullet points referencing the facts you collected (IDs, dates, status).
"""

    agent = Agent(
        name="router_agent",
        description="Routes work between specialists and returns the final answer.",
        model=MODEL_NAME,
        instruction=instruction,
        sub_agents=[remote_customer_data, remote_support],
    )
    card = AgentCard(
        name="Router Agent",
        url=f"http://{HOST}:{ROUTER_AGENT_PORT}",
        description="Entry point for customer-support A2A flows.",
        version="1.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        preferred_transport=TransportProtocol.jsonrpc,
        skills=_skills(
            "router",
            "Understands customer intents and coordinates the specialists.",
            [
                "I need help with my account, customer ID 2.",
                "Cancel my subscription and handle the billing issue.",
                "What are the high-priority tickets for premium customers?",
            ],
        ),
    )
    return agent, card, ROUTER_AGENT_PORT


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


async def _run_agent_server(agent: Agent, card: AgentCard, port: int) -> None:
    app = _create_agent_app(agent, card)
    config = uvicorn.Config(
        app.build(),
        host=HOST,
        port=port,
        log_level="warning",
        loop="none",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def _start_all_servers() -> None:
    customer_agent, customer_card, customer_port = _create_customer_data_agent()
    support_agent, support_card, support_port = _create_support_agent()

    tasks = [
        asyncio.create_task(_run_agent_server(customer_agent, customer_card, customer_port)),
        asyncio.create_task(_run_agent_server(support_agent, support_card, support_port)),
    ]
    print(" Starting ADK A2A agent servers ...")
    await asyncio.sleep(2)
    print(f"   - Customer Data Agent: http://{HOST}:{customer_port}")
    print(f"   - Support Agent:       http://{HOST}:{support_port}")
    await asyncio.gather(*tasks)


def _run_servers_in_background() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_start_all_servers())


# ---------------------------------------------------------------------------
# A2A demo client
# ---------------------------------------------------------------------------


class A2ASimpleClient:
    def __init__(self, default_timeout: float = 240.0):
        self.default_timeout = default_timeout
        self._cache: Dict[str, Dict[str, Any]] = {}

    async def create_task(self, agent_url: str, message: str) -> str:
        timeout = httpx.Timeout(
            timeout=self.default_timeout,
            connect=10.0,
            read=self.default_timeout,
            write=10.0,
            pool=5.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as httpx_client:
            if agent_url not in self._cache:
                resp = await httpx_client.get(f"{agent_url}{AGENT_CARD_WELL_KNOWN_PATH}")
                resp.raise_for_status()
                self._cache[agent_url] = resp.json()
            agent_card = AgentCard(**self._cache[agent_url])

            factory = ClientFactory(
                ClientConfig(
                    httpx_client=httpx_client,
                    supported_transports=[TransportProtocol.jsonrpc, TransportProtocol.http_json],
                    use_client_preference=True,
                )
            )
            client = factory.create(agent_card)
            message_obj = create_text_message_object(content=message)
            responses = []
            async for response in client.send_message(message_obj):
                responses.append(response)
            if responses and isinstance(responses[0], tuple) and responses[0]:
                task = responses[0][0]
                try:
                    return task.artifacts[0].parts[0].root.text
                except Exception:
                    return str(task)
            return "No response received."


class RouterOrchestrator:
    """Deterministic router that coordinates the ADK specialists."""

    def __init__(self, client: A2ASimpleClient):
        self.client = client
        self.customer_url = f"http://{HOST}:{DATA_AGENT_PORT}"
        self.support_url = f"http://{HOST}:{SUPPORT_AGENT_PORT}"

    async def run(self, query: str) -> str:
        scenario = self._classify_query(query)
        customer_id = self._extract_customer_id(query)
        self._log_intent(scenario, customer_id)
        if scenario == "premium_report":
            return await self._handle_premium_report(query)
        if scenario == "active_open":
            return await self._handle_active_open(query)
        if scenario == "update_email_history":
            return await self._handle_update_email_history(query, customer_id)
        if scenario == "billing_escalation":
            return await self._handle_billing_escalation(query, customer_id)
        if scenario == "upgrade":
            return await self._handle_account_lookup(query, customer_id, scenario_hint="upgrade")
        if scenario == "account_help":
            return await self._handle_account_lookup(query, customer_id, scenario_hint="account_help")
        if scenario == "simple_lookup":
            return await self._handle_account_lookup(query, customer_id, scenario_hint="profile_lookup")
        if scenario == "multi_intent":
            return await self._handle_update_email_history(query, customer_id)
        return await self._handle_generic(query, customer_id)

    async def _handle_account_lookup(
        self, query: str, customer_id: Optional[int], scenario_hint: str
    ) -> str:
        if not customer_id:
            return "Please provide a customer ID so I can review the account."
        prompt = self._build_customer_data_prompt(
            f"Retrieve the customer profile for ID {customer_id} using get_customer. "
            "Return the MCP payload verbatim inside context_payload and set recommended_next_agent to 'router'."
        )
        data_result = await self._call_customer_data(prompt)
        context = data_result.get("context_payload")
        response = await self._respond_with_support(query, context, customer_id, scenario_hint=scenario_hint)
        return response

    async def _handle_premium_report(self, query: str) -> str:
        prompt = self._build_customer_data_prompt(
            "Identify premium/active customers and gather the status of all high-priority tickets. "
            "Use list_customers/list_active_customers_with_open_tickets and get_high_priority_tickets_by_ids as needed. "
            "Summarize counts and include the raw ticket list in context_payload. "
            "Set recommended_next_agent to 'router' when done."
        )
        data_result = await self._call_customer_data(prompt)
        context = data_result.get("context_payload")
        return await self._respond_with_support(query, context, customer_id=None, scenario_hint="premium_report")

    async def _handle_active_open(self, query: str) -> str:
        prompt = self._build_customer_data_prompt(
            "Use list_active_customers_with_open_tickets to provide all active customers that currently have open tickets. "
            "Include the tool output in context_payload and recommended_next_agent='router'."
        )
        data_result = await self._call_customer_data(prompt)
        context = data_result.get("context_payload")
        return await self._respond_with_support(query, context, customer_id=None, scenario_hint="active_open")

    async def _handle_update_email_history(self, query: str, customer_id: Optional[int]) -> str:
        if not customer_id:
            return "Please provide your customer ID so I can update the account."
        new_email = self._extract_email(query)
        if not new_email:
            return "Please provide the new email address to update."
        prompt = self._build_customer_data_prompt(
            f"Update customer ID {customer_id}'s email to {new_email} using update_customer, "
            "then fetch their ticket history via get_customer_history. "
            "Include both the update confirmation and ticket list in context_payload."
        )
        data_result = await self._call_customer_data(prompt)
        context = data_result.get("context_payload")
        return await self._respond_with_support(query, context, customer_id=customer_id, scenario_hint="multi_intent")

    async def _handle_billing_escalation(self, query: str, customer_id: Optional[int]) -> str:
        # Kick off negotiation by asking support first.
        support_result = await self._support_dialog(
            query, initial_context=None, customer_id=customer_id, scenario_hint="billing_escalation"
        )
        return self._format_support_response(support_result)

    async def _handle_generic(self, query: str, customer_id: Optional[int]) -> str:
        # Default behavior: fetch profile if ID present, otherwise go straight to support.
        context = None
        if customer_id:
            prompt = self._build_customer_data_prompt(
                f"Retrieve the customer profile for ID {customer_id} and include ticket counts if available."
            )
            data_result = await self._call_customer_data(prompt)
            context = data_result.get("context_payload")
        support_result = await self._support_dialog(
            query, initial_context=context, customer_id=customer_id, scenario_hint="general"
        )
        return self._format_support_response(support_result)

    # ------------------------------------------------------------------
    # Support coordination helpers
    # ------------------------------------------------------------------

    async def _respond_with_support(
        self,
        query: str,
        initial_context: Optional[Any],
        customer_id: Optional[int],
        scenario_hint: Optional[str],
    ) -> str:
        result = await self._support_dialog(
            query, initial_context=initial_context, customer_id=customer_id, scenario_hint=scenario_hint
        )
        return self._format_support_response(result)

    async def _support_dialog(
        self,
        query: str,
        initial_context: Optional[Any],
        customer_id: Optional[int],
        scenario_hint: Optional[str],
    ) -> Dict[str, Any]:
        context_payload = initial_context
        while True:
            instructions = self._build_support_prompt(query, context_payload, scenario_hint)
            support_result = await self._call_support(instructions)
            if not support_result:
                return {}
            if not support_result.get("requires_context"):
                return support_result
            if not customer_id:
                return support_result
            requested_type = support_result.get("context_type")
            if not requested_type:
                return support_result
            data_context = await self._request_additional_context(requested_type, customer_id, query)
            context_payload = data_context.get("context_payload")

    def _format_support_response(self, support_result: Dict[str, Any]) -> str:
        if not support_result:
            return "Support agent returned no data."
        if support_result.get("requires_context"):
            notes = support_result.get("notes_for_router")
            if notes:
                return notes
            missing = support_result.get("context_type", "additional information")
            return f"Support requires more {missing} before responding."
        response = support_result.get("customer_response", json.dumps(support_result, indent=2))
        notes = support_result.get("notes_for_router")
        final_message = f"Final customer message:\n{response}"
        if notes:
            final_message += f"\n\n[Router note: {notes}]"
        return final_message

    # ------------------------------------------------------------------
    # MCP agent invocation helpers
    # ------------------------------------------------------------------

    async def _call_customer_data(self, instructions: str) -> Dict[str, Any]:
        reminder = ""
        last_text = ""
        for attempt in range(3):
            prompt = instructions if not reminder else f"{instructions}\n\nROUTER REMINDER: {reminder}"
            _log_handoff("Router", "CustomerData", prompt)
            text = await self.client.create_task(self.customer_url, prompt)
            last_text = text
            data = self._validate_customer_payload(_parse_json(text))
            if data.get("_invalid"):
                reminder = data["_invalid"]
                _log_handoff("CustomerData", "Router", reminder)
                continue
            summary = data.get("handoff_summary") or data.get("notes_for_router") or "Received payload"
            _log_handoff("CustomerData", "Router", summary)
            return data
        fallback = {
            "handoff_summary": "Customer Data agent returned unexpected output.",
            "recommended_next_agent": "router",
            "context_type": "general",
            "context_payload": {"raw_text": last_text or "No data returned."},
            "notes_for_router": "Router fallback activated after schema violations.",
        }
        _log_handoff("CustomerData", "Router", "Auto-generated fallback payload.")
        return fallback

    async def _call_support(self, instructions: str) -> Dict[str, Any]:
        reminder = ""
        last_text = ""
        for attempt in range(3):
            prompt = instructions if not reminder else f"{instructions}\n\nROUTER REMINDER: {reminder}"
            _log_handoff("Router", "Support", prompt)
            text = await self.client.create_task(self.support_url, prompt)
            last_text = text
            parsed = _parse_json(text)
            data = self._validate_support_payload(parsed, strict=True)
            if data.get("_invalid"):
                reminder = data["_invalid"]
                _log_handoff("Support", "Router", f"Invalid payload: {reminder}")
                continue
            summary = (
                "Needs context"
                if data.get("requires_context")
                else data.get("customer_response", "Received payload")
            )
            _log_handoff("Support", "Router", summary)
            return data
        coerced = self._coerce_support_text(last_text or "Support response unavailable.")
        _log_handoff("Support", "Router", "Auto-structured fallback support response.")
        return coerced

    async def _request_additional_context(self, context_type: str, customer_id: int, query: str) -> Dict[str, Any]:
        if context_type in {"billing", "tickets", "history"}:
            task = (
                f"Provide the detailed ticket/billing history for customer ID {customer_id} using get_customer_history "
                f"so support can resolve: {query}"
            )
        elif context_type == "profile":
            task = f"Fetch the latest profile for customer ID {customer_id} using get_customer."
        else:
            task = f"Provide {context_type} details for customer ID {customer_id} relevant to: {query}"
        prompt = self._build_customer_data_prompt(task)
        return await self._call_customer_data(prompt)

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_customer_data_prompt(self, task: str) -> str:
        return (
            f"Task: {task}\n"
            "Return STRICT JSON using your declared schema with recommended_next_agent guidance. "
            "Include raw MCP output in context_payload."
        )

    def _build_support_prompt(
        self, query: str, context_payload: Optional[Any], scenario_hint: Optional[str]
    ) -> str:
        if isinstance(context_payload, (dict, list)):
            context_block = json.dumps(context_payload, indent=2)
        elif context_payload is None:
            context_block = "No structured context yet."
        else:
            context_block = str(context_payload)
        hint = f"Scenario hint: {scenario_hint}" if scenario_hint else "Scenario hint: generic"
        extra_guidance = ""
        if scenario_hint == "billing_escalation":
            if context_payload in (None, "No structured context yet.") or (
                isinstance(context_payload, str) and "No structured" in context_payload
            ):
                extra_guidance = (
                    "\nYou have NOT received any billing/ticket context yet. Respond with the JSON form "
                    "requesting `requires_context=true` and `context_type=\"billing\"`, explaining exactly "
                    "what the router should fetch."
                )
            else:
                extra_guidance = (
                    "\nYou now have the requested billing/ticket context. Provide the final response with "
                    "`requires_context=false`, acknowledging urgency, referencing ticket IDs, and outlining next steps."
                )
        reminder = (
            "Respond ONLY with strict JSON as defined in your system instructions. "
            "Do not include any natural language outside the JSON object."
        )
        return (
            f"Customer query:\n{query}\n\n"
            f"{hint}\n"
            "Context payload (if any):\n"
            f"{context_block}\n"
            f"{extra_guidance}\n\n"
            f"{reminder}"
        )

    # ------------------------------------------------------------------
    # Query understanding helpers
    # ------------------------------------------------------------------

    def _classify_query(self, query: str) -> str:
        q = query.lower()
        if "high-priority" in q or "premium" in q:
            return "premium_report"
        if "active customers" in q and "open tickets" in q:
            return "active_open"
        if "update my email" in q and "ticket history" in q:
            return "update_email_history"
        if ("cancel" in q or "refund" in q or "charged twice" in q or "billing" in q) and "customer id" in q:
            return "billing_escalation"
        if "charged twice" in q or "refund" in q:
            return "billing_escalation"
        if "upgrade" in q:
            return "upgrade"
        if "help with my account" in q or "check my details" in q:
            return "account_help"
        if "customer information" in q:
            return "simple_lookup"
        if "ticket history" in q and "update" in q:
            return "multi_intent"
        return "generic"

    def _extract_customer_id(self, query: str) -> Optional[int]:
        match = re.search(r"(?:customer\s*(?:id)?|id)\s*(?:is|:)?\s*(\d+)", query, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return self._infer_customer_id_from_keywords(query.lower())

    def _extract_email(self, query: str) -> Optional[str]:
        match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", query)
        return match.group(0) if match else None

    def _infer_customer_id_from_keywords(self, query_lower: str) -> Optional[int]:
        """Heuristic to infer a customer ID when the user omits it."""
        keyword_map = [
            ({"charged twice", "double charge", "refund"}, 4),
            ({"upgrade my account", "upgrade assistance"}, 3),
        ]
        for keywords, cid in keyword_map:
            if any(keyword in query_lower for keyword in keywords):
                return cid
        return None

    def _log_intent(self, scenario: str, customer_id: Optional[int]) -> None:
        cid_text = f"customer_id={customer_id}" if customer_id else "customer_id=unknown"
        _log_handoff("Router", "Intent", f"{scenario} | {cid_text}")

    def _validate_customer_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict) or not payload:
            return {"_invalid": "Customer Data agent returned non-JSON output."}
        required = {"handoff_summary", "context_payload"}
        missing = sorted(required - payload.keys())
        if missing:
            return {"_invalid": f"Missing keys from Customer Data payload: {', '.join(missing)}"}
        if "recommended_next_agent" not in payload:
            payload["recommended_next_agent"] = "router"
        return payload

    def _validate_support_payload(self, payload: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        if not isinstance(payload, dict) or not payload:
            return {"_invalid": "Response was not valid JSON."} if strict else {}
        if "requires_context" not in payload:
            if strict:
                return {"_invalid": "Missing requires_context flag."}
            payload["requires_context"] = True
        if payload.get("requires_context"):
            if not payload.get("context_type"):
                if strict:
                    return {"_invalid": "requires_context=true but context_type missing."}
                payload["context_type"] = "profile"
        else:
            if payload.get("context_type") is not None and payload.get("context_type") != "null":
                payload["context_type"] = None
            if "customer_response" not in payload:
                if strict:
                    return {"_invalid": "requires_context=false but customer_response missing."}
                payload["customer_response"] = json.dumps(payload)
        return payload

    def _coerce_support_text(self, text: str) -> Dict[str, Any]:
        normalized = text.strip()
        lower = normalized.lower()
        needs_context = any(
            phrase in lower
            for phrase in [
                "need",
                "provide your",
                "customer id",
                "billing",
                "more information",
                "context",
                "cannot proceed",
            ]
        )
        if needs_context:
            if "billing" in lower or "charge" in lower or "refund" in lower:
                context_type = "billing"
            elif "ticket" in lower or "history" in lower:
                context_type = "history"
            else:
                context_type = "profile"
            return {
                "requires_context": True,
                "context_type": context_type,
                "notes_for_router": (
                    "Support agent asked for additional context but responded outside JSON. "
                    f"Original text: {normalized or 'N/A'}"
                ),
            }
        return {
            "requires_context": False,
            "context_type": None,
            "customer_response": normalized or "Support response unavailable.",
            "notes_for_router": "Support agent returned plain text; router wrapped it in schema.",
        }


SCENARIOS = [
    # "I need help with my account, customer ID 2. Can you check my details and assist?",
    # "My customer ID is 3. I want to cancel my subscription but I'm having billing issues.",
    # "What's the status of all high-priority tickets for premium customers?",
    "Get customer information for ID 5.",
    "I'm customer 3 and need help upgrading my account.",
    "Show me all active customers who have open tickets.",
    "My customer ID is 4. Update my email to new@email.com and show my ticket history.",
    "I've been charged twice, please refund immediately!",
]


async def _run_demo() -> None:
    client = A2ASimpleClient()
    orchestrator = RouterOrchestrator(client)
    print("\n Running demo scenarios via Router Agent\n")
    for scenario in SCENARIOS:
        print("=" * 80)
        print(f"Query: {scenario}")
        print("=" * 80)
        response = await orchestrator.run(scenario)
        print(response)
        print()


def main() -> None:
    nest_asyncio.apply()
    server_thread = threading.Thread(target=_run_servers_in_background, daemon=True)
    server_thread.start()
    time.sleep(3)
    asyncio.run(_run_demo())


if __name__ == "__main__":
    main()
