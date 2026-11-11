"""
LangGraph Agent Workflow: Stateful agent with MCP tool integration,
self-correction, and conversation memory.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import math
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Literal, TypedDict, Annotated

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "business.db")
POLICIES_DIR = os.path.join(BASE_DIR, "data", "policies")
load_dotenv(os.path.join(BASE_DIR, ".env"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
USE_MOCK_LLM = not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here"

MAX_CORRECTION_LOOPS = 3


# ── State Definition ───────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """State flowing through the LangGraph workflow."""
    messages: list[dict[str, Any]]
    current_task: str
    plan: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    correction_count: int
    needs_correction: bool
    final_response: str
    error: str
    trace: list[dict[str, Any]]


# ── MCP Tool Implementations (Direct, no subprocess) ──────────

_DESTRUCTIVE_SQL = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|PRAGMA\s+\w+\s*=|VACUUM)\b",
    re.IGNORECASE,
)


def tool_query_business_db(sql_query: str) -> str:
    """Execute a read-only SQL query against the business database."""
    if _DESTRUCTIVE_SQL.search(sql_query):
        return "ERROR: Only SELECT queries are permitted. Destructive operations are blocked."

    stripped = sql_query.strip().rstrip(";").strip()
    if not stripped.upper().startswith("SELECT"):
        return "ERROR: Only SELECT queries are permitted."

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        conn.close()

        if not rows:
            return "Query returned 0 rows."

        header = " | ".join(columns)
        lines = [header, "-+-".join("-" * max(len(c), 8) for c in columns)]
        for row in rows[:100]:
            lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))

        result = "\n".join(lines)
        if len(rows) > 100:
            result += f"\n... ({len(rows) - 100} more rows truncated)"
        return f"Query returned {len(rows)} row(s):\n\n{result}"

    except sqlite3.Error as e:
        return f"SQL_ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_search_policy_docs(query: str) -> str:
    """Search policy documents using TF-IDF."""
    docs: dict[str, dict] = {}
    if not os.path.isdir(POLICIES_DIR):
        return "No policy documents found."

    for fname in os.listdir(POLICIES_DIR):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(POLICIES_DIR, fname), "r", encoding="utf-8") as f:
            content = f.read()

        tokens = re.findall(r"[a-z0-9]+", content.lower())
        tf = Counter(tokens)
        total = len(tokens) or 1
        docs[fname] = {
            "content": content,
            "tf": {t: c / total for t, c in tf.items()},
        }

    if not docs:
        return "No policy documents found."

    # IDF
    num_docs = len(docs)
    all_terms: set[str] = set()
    for d in docs.values():
        all_terms.update(d["tf"].keys())

    idf = {}
    for term in all_terms:
        doc_freq = sum(1 for d in docs.values() if term in d["tf"])
        idf[term] = math.log((num_docs + 1) / (doc_freq + 1)) + 1

    for d in docs.values():
        d["tfidf"] = {t: d["tf"][t] * idf.get(t, 1.0) for t in d["tf"]}

    # Score
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    query_tf = Counter(query_tokens)
    qt = len(query_tokens) or 1
    query_vec = {t: c / qt for t, c in query_tf.items()}

    results = []
    for fname, doc in docs.items():
        common = set(query_vec.keys()) & set(doc["tfidf"].keys())
        if not common:
            continue
        dot = sum(query_vec[t] * doc["tfidf"][t] for t in common)
        q_norm = math.sqrt(sum(v ** 2 for v in query_vec.values()))
        d_norm = math.sqrt(sum(v ** 2 for v in doc["tfidf"].values()))
        if q_norm == 0 or d_norm == 0:
            continue
        sim = dot / (q_norm * d_norm)
        results.append((fname, sim, doc["content"]))

    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return "No relevant policy documents found."

    parts = []
    for fname, score, content in results[:2]:
        sections = re.split(r"\n##\s+", content)
        best = max(sections, key=lambda s: sum(1 for w in query.lower().split() if w in s.lower()))
        if len(best) > 1500:
            best = best[:1500] + "\n... (truncated)"
        parts.append(f"📄 **{fname}** (relevance: {score:.3f})\n\n{best.strip()}")

    return "\n\n---\n\n".join(parts)


def tool_create_action_ticket(
    customer_id: int, issue_summary: str, priority: str, resolution_action: str
) -> str:
    """Create an action ticket in the database."""
    priority = priority.lower().strip()
    if priority not in ("critical", "high", "medium", "low"):
        return f"ERROR: Invalid priority '{priority}'."

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT id, name FROM customers WHERE id = ?", (customer_id,))
        customer = cur.fetchone()
        if not customer:
            conn.close()
            return f"ERROR: Customer with id={customer_id} not found."

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
            "INSERT INTO action_tickets (customer_id, issue_summary, priority, resolution_action, status, created_at) "
            "VALUES (?, ?, ?, ?, 'created', ?)",
            (customer_id, issue_summary, priority, resolution_action, now),
        )
        ticket_id = cur.lastrowid
        conn.commit()
        conn.close()

        return (
            f"✅ Action ticket created successfully.\n"
            f"  Ticket ID: ACT-{ticket_id:04d}\n"
            f"  Customer: {customer['name']} (ID: {customer_id})\n"
            f"  Priority: {priority.upper()}\n"
            f"  Summary: {issue_summary}\n"
            f"  Action: {resolution_action}\n"
            f"  Status: CREATED"
        )
    except Exception as e:
        return f"ERROR: {e}"


def tool_get_schema() -> str:
    """Return database schema."""
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cur.fetchall()]

        parts = ["# Database Schema\n"]
        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            columns = cur.fetchall()
            parts.append(f"\n## {table}")
            for col in columns:
                parts.append(f"  - {col['name']} ({col['type']})")

        conn.close()
        return "\n".join(parts)
    except Exception as e:
        return f"Error: {e}"


# Available tools registry
TOOLS_REGISTRY = {
    "query_business_db": {
        "fn": tool_query_business_db,
        "description": "Execute read-only SQL against the business database.",
        "parameters": ["sql_query"],
    },
    "search_policy_docs": {
        "fn": tool_search_policy_docs,
        "description": "Search company SOP/policy documents.",
        "parameters": ["query"],
    },
    "create_action_ticket": {
        "fn": tool_create_action_ticket,
        "description": "Create an automated action ticket.",
        "parameters": ["customer_id", "issue_summary", "priority", "resolution_action"],
    },
    "get_schema": {
        "fn": tool_get_schema,
        "description": "Get the database table schemas.",
        "parameters": [],
    },
}


# ── Mock LLM for offline/test mode ────────────────────────────
def _mock_sql_for_task(task: str) -> str:
    """Select useful, schema-aware SQL templates for common offline demo tasks."""
    text = task.lower()
    limit_match = re.search(r"\b(?:top|first|last)\s+(\d+)\b", text)
    limit = min(int(limit_match.group(1)), 100) if limit_match else 20

    if "ticket" in text or "sla breach" in text:
        join_customer = "enterprise" in text or "customer name" in text or "customer details" in text
        columns = "t.id, c.name AS customer, t.subject, t.priority, t.status, t.sla_deadline" if join_customer else "id, customer_id, subject, priority, status, sla_deadline"
        source = "support_tickets t JOIN customers c ON c.id = t.customer_id" if join_customer else "support_tickets"
        filters = []
        if "enterprise" in text:
            filters.append("c.plan = 'enterprise'")
        if "critical" in text:
            filters.append("t.priority = 'critical'" if join_customer else "priority = 'critical'")
        if "high" in text:
            filters.append("t.priority = 'high'" if join_customer else "priority = 'high'")
        if "open" in text or "unresolved" in text:
            filters.append("t.status IN ('open', 'in_progress')" if join_customer else "status IN ('open', 'in_progress')")
        if "breach" in text or "overdue" in text:
            filters.append("t.sla_deadline < datetime('now') AND t.status NOT IN ('resolved', 'closed')" if join_customer else "sla_deadline < datetime('now') AND status NOT IN ('resolved', 'closed')")
        where = f" WHERE {' AND '.join(filters)}" if filters else ""
        return f"SELECT {columns} FROM {source}{where} ORDER BY sla_deadline LIMIT {limit}"

    if "order" in text or "revenue" in text or "sales" in text:
        if "top customer" in text or "order count" in text or "most orders" in text:
            return (
                "SELECT c.id, c.name, COUNT(o.id) AS order_count "
                "FROM customers c LEFT JOIN orders o ON o.customer_id = c.id "
                f"GROUP BY c.id, c.name ORDER BY order_count DESC LIMIT {limit}"
            )
        if "revenue" in text or "sales" in text or "total amount" in text:
            return "SELECT status, COUNT(*) AS order_count, ROUND(SUM(amount), 2) AS revenue FROM orders GROUP BY status ORDER BY revenue DESC"
        return f"SELECT id, customer_id, product, amount, status, order_date FROM orders ORDER BY order_date DESC LIMIT {limit}"

    if "customer" in text:
        if "how many" in text or "count" in text or "number of" in text:
            plan = re.search(r"\b(free|pro|enterprise)\b", text)
            if plan:
                return f"SELECT COUNT(*) AS customer_count FROM customers WHERE plan = '{plan.group(1)}'"
            return "SELECT COUNT(*) AS customer_count FROM customers"
        plan = re.search(r"\b(free|pro|enterprise)\b", text)
        condition = f" WHERE plan = '{plan.group(1)}'" if plan else ""
        return f"SELECT id, name, email, plan, region FROM customers{condition} ORDER BY id LIMIT {limit}"

    return f"SELECT id, name, email, plan, region FROM customers ORDER BY id LIMIT {limit}"


class MockLLM:
    """Deterministic mock LLM that generates reasonable tool calls and responses."""

    def invoke(self, messages: list, **kwargs) -> dict:
        last_msg = ""
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                last_msg = last.get("content", "")
            elif hasattr(last, "content"):
                last_msg = last.content or ""

        last_lower = last_msg.lower()

        # Correction messages must be handled before normal tool-result synthesis.
        correction = any(
            isinstance(m, dict) and m.get("role") == "system" and "previous tool call" in m.get("content", "")
            for m in messages
        )

        # If we have tool results in the message history, synthesize a final answer
        has_tool_results = any(
            isinstance(m, dict) and m.get("role") == "tool" for m in messages
        ) and last_lower.startswith("based on the tool results")

        if has_tool_results and not correction:
            last_user_index = max(
                (index for index, message in enumerate(messages[:-1])
                 if isinstance(message, dict) and message.get("role") == "user"),
                default=-1,
            )
            results = "\n\n".join(
                message.get("content", "")
                for message in messages[last_user_index + 1:-1]
                if isinstance(message, dict) and message.get("role") == "tool"
            )
            return {
                "type": "response",
                "content": f"Here are the results from the business operations tools:\n\n{results}",
                "tool_calls": [],
            }

        # Determine which tool to call based on keywords
        if correction:
            return {
                "type": "tool_call",
                "content": "The first attempt failed; retrying with a valid query against the known schema.",
                "tool_calls": [{"tool": "query_business_db", "args": {"sql_query": "SELECT id, name, email, plan FROM customers LIMIT 5"}}],
            }
        if any(kw in last_lower for kw in ["schema", "table structure", "columns"]):
            return {
                "type": "tool_call",
                "content": "I'll retrieve the database schema.",
                "tool_calls": [{"tool": "get_schema", "args": {}}],
            }
        elif any(kw in last_lower for kw in ["create an action", "create action", "create a ticket", "create ticket", "resolve", "fix"]):
            customer_match = re.search(r"\bcustomer(?:\s+id)?\s*#?\s*(\d+)\b", last_lower)
            requested_priority = next(
                (value for value in ("critical", "high", "medium", "low") if value in last_lower),
                "medium",
            )
            return {
                "type": "tool_call",
                "content": "I'll create an action ticket.",
                "tool_calls": [{"tool": "create_action_ticket", "args": {
                    "customer_id": int(customer_match.group(1)) if customer_match else 1,
                    "issue_summary": last_msg[:100],
                    "priority": requested_priority,
                    "resolution_action": "Investigate and resolve the reported issue",
                }}],
            }
        elif (
            any(kw in last_lower for kw in ["policy", "sla", "refund", "escalat", "procedure", "rule", "compliance"])
            and not any(kw in last_lower for kw in ["show", "list", "find", "count", "query", "database", "records", "breach", "overdue"])
        ):
            return {
                "type": "tool_call",
                "content": "I'll search the policy documents.",
                "tool_calls": [{"tool": "search_policy_docs", "args": {"query": last_msg[:200]}}],
            }
        elif any(kw in last_lower for kw in ["sql", "query", "database", "customer", "order", "ticket", "revenue", "count", "select", "show", "list", "find", "how many"]):
            return {
                "type": "tool_call",
                "content": "I'll query the business database to find this information.",
                "tool_calls": [
                    {
                        "tool": "query_business_db",
                        "args": {"sql_query": _mock_sql_for_task(last_msg)},
                    }
                ],
            }
        else:
            return {
                "type": "tool_call",
                "content": "Let me look into this by checking the database.",
                "tool_calls": [
                    {
                        "tool": "query_business_db",
                        "args": {"sql_query": _mock_sql_for_task(last_msg)},
                    }
                ],
            }


def _get_llm():
    """Return the LLM instance — real Gemini or mock."""
    if USE_MOCK_LLM:
        return MockLLM()

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )


# ── LangGraph Node Functions ──────────────────────────────────

def router_planner(state: AgentState) -> AgentState:
    """Router/Planner node: Analyzes the task and determines the execution plan."""
    messages = list(state.get("messages", []))
    trace = list(state.get("trace", []))

    llm = _get_llm()

    system_msg = {
        "role": "system",
        "content": (
            "You are an AI business operations agent with access to these tools:\n"
            "1. query_business_db(sql_query) - Execute read-only SQL on the business DB\n"
            "2. search_policy_docs(query) - Search company policy/SOP documents\n"
            "3. create_action_ticket(customer_id, issue_summary, priority, resolution_action) - Create action tickets\n"
            "4. get_schema() - Read config://schema from the MCP server\n\n"
            "Analyze the latest user request. Return only valid JSON with keys `content` (short plan) "
            "and `tool_calls` (array of objects with `tool` and `args`). Use an empty array only when no tool is needed."
        ),
    }

    # Gemini's chat adapter expects standard roles; internal tool transcript is summarized separately.
    all_messages = [system_msg] + [m for m in messages if m.get("role") in ("user", "assistant", "system")]
    response = llm.invoke(all_messages)

    if isinstance(response, dict):
        plan = response.get("content", "")
        tool_calls = response.get("tool_calls", [])
    else:
        plan = getattr(response, "content", str(response))
        tool_calls = []
        try:
            candidate = plan.strip()
            if candidate.startswith("```"):
                candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate)
            payload = json.loads(candidate)
            plan = payload.get("content", "")
            tool_calls = payload.get("tool_calls", [])
        except (json.JSONDecodeError, AttributeError):
            plan = plan or "Could not parse a tool plan from the model response."

    trace.append({
        "node": "router_planner",
        "timestamp": datetime.utcnow().isoformat(),
        "plan": plan,
        "tool_calls_planned": len(tool_calls),
    })

    return {
        "messages": messages,
        "current_task": state.get("current_task", ""),
        "plan": plan,
        "tool_calls": tool_calls,
        "tool_results": [],
        "correction_count": state.get("correction_count", 0),
        "needs_correction": False,
        "final_response": "",
        "error": "",
        "trace": trace,
    }


async def _invoke_mcp(tool_name: str, args: dict[str, Any]) -> str:
    """Start the local MCP server, discover its capabilities, and invoke one."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=BASE_DIR,
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            if tool_name == "get_schema":
                resource = await session.read_resource("config://schema")
                return "\n".join(
                    item.text for item in resource.contents if hasattr(item, "text")
                )

            available = await session.list_tools()
            if tool_name not in {tool.name for tool in available.tools}:
                return f"TOOL_ERROR: Unknown MCP tool '{tool_name}'"
            result = await session.call_tool(tool_name, arguments=args)
            text_parts = [item.text for item in result.content if hasattr(item, "text")]
            structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None) or {}
            output = "\n".join(text_parts) or json.dumps(structured)
            is_error = getattr(result, "isError", getattr(result, "is_error", False))
            return f"TOOL_ERROR: {output}" if is_error else output


async def tool_executor(state: AgentState) -> AgentState:
    """Tool Executor node: Executes tool calls through the MCP client protocol."""
    tool_calls = state.get("tool_calls", [])
    trace = list(state.get("trace", []))
    tool_results = []
    messages = list(state.get("messages", []))

    for tc in tool_calls:
        tool_name = tc.get("tool", "")
        args = tc.get("args", {})

        trace_entry = {
            "node": "tool_executor",
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "args": args,
        }

        if tool_name in TOOLS_REGISTRY:
            try:
                result = await _invoke_mcp(tool_name, args)
                trace_entry["status"] = "error" if result.startswith(("ERROR:", "SQL_ERROR:", "TOOL_ERROR:", "DATABASE_ERROR:")) else "success"
                trace_entry["result_preview"] = result[:200]
            except Exception as e:
                result = f"TOOL_ERROR: {e}"
                trace_entry["status"] = "error"
                trace_entry["error"] = str(e)
        else:
            result = f"TOOL_ERROR: Unknown tool '{tool_name}'"
            trace_entry["status"] = "error"
            trace_entry["error"] = f"Unknown tool: {tool_name}"

        tool_results.append({"tool": tool_name, "args": args, "result": result})
        trace.append(trace_entry)
        messages.append({"role": "tool", "content": f"[{tool_name}] {result}"})

    return {
        "messages": messages,
        "current_task": state.get("current_task", ""),
        "plan": state.get("plan", ""),
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "correction_count": state.get("correction_count", 0),
        "needs_correction": False,
        "final_response": "",
        "error": "",
        "trace": trace,
    }


def validator(state: AgentState) -> AgentState:
    """Validator/Self-Correction node: Checks tool outputs for errors."""
    tool_results = state.get("tool_results", [])
    correction_count = state.get("correction_count", 0)
    trace = list(state.get("trace", []))
    messages = list(state.get("messages", []))

    needs_correction = False
    error_details = ""

    for tr in tool_results:
        result = tr.get("result", "")
        if isinstance(result, str) and any(
            result.startswith(err) for err in ["ERROR:", "SQL_ERROR:", "TOOL_ERROR:", "DATABASE_ERROR:"]
        ):
            if correction_count < MAX_CORRECTION_LOOPS:
                needs_correction = True
                error_details = result
            break

    trace.append({
        "node": "validator",
        "timestamp": datetime.utcnow().isoformat(),
        "needs_correction": needs_correction,
        "correction_count": correction_count,
        "error_details": error_details[:200] if error_details else "",
    })

    if needs_correction:
        messages.append({
            "role": "system",
            "content": f"The previous tool call resulted in an error: {error_details}. "
                       f"Please fix the query/parameters and try again. Attempt {correction_count + 1}/{MAX_CORRECTION_LOOPS}.",
        })

    return {
        "messages": messages,
        "current_task": state.get("current_task", ""),
        "plan": state.get("plan", ""),
        "tool_calls": state.get("tool_calls", []),
        "tool_results": tool_results,
        "correction_count": correction_count + (1 if needs_correction else 0),
        "needs_correction": needs_correction,
        "final_response": "",
        "error": error_details if needs_correction else "",
        "trace": trace,
    }


def synthesizer(state: AgentState) -> AgentState:
    """Final Synthesizer node: Produces the final human-readable response."""
    messages = list(state.get("messages", []))
    tool_results = state.get("tool_results", [])
    trace = list(state.get("trace", []))

    llm = _get_llm()

    results_summary = ""
    for tr in tool_results:
        tool_name = tr.get("tool", "unknown")
        result = tr.get("result", "")
        if len(result) > 2000:
            result = result[:2000] + "... (truncated)"
        results_summary += f"\n\n### Tool: {tool_name}\n{result}"

    synthesis_msg = {
        "role": "user",
        "content": (
            f"Based on the tool results below, answer this user task clearly: {state.get('current_task', '')}"
            f"\n\nTool Results:{results_summary}"
        ),
    }

    synthesis_messages = messages + [synthesis_msg] if isinstance(llm, MockLLM) else [synthesis_msg]
    response = llm.invoke(synthesis_messages)

    if isinstance(response, dict):
        final_response = response.get("content", "")
    else:
        final_response = getattr(response, "content", str(response))

    if not final_response or final_response.strip() == "":
        parts = ["Here are the results from the business operations analysis:\n"]
        for tr in tool_results:
            parts.append(f"**{tr['tool']}**: {tr['result'][:500]}")
        final_response = "\n\n".join(parts)

    trace.append({
        "node": "synthesizer",
        "timestamp": datetime.utcnow().isoformat(),
        "response_length": len(final_response),
    })

    return {
        "messages": messages + [{"role": "assistant", "content": final_response}],
        "current_task": state.get("current_task", ""),
        "plan": state.get("plan", ""),
        "tool_calls": state.get("tool_calls", []),
        "tool_results": tool_results,
        "correction_count": state.get("correction_count", 0),
        "needs_correction": False,
        "final_response": final_response,
        "error": "",
        "trace": trace,
    }


# ── Conditional Edge ───────────────────────────────────────────

def should_correct(state: AgentState) -> str:
    """Decide whether to loop back for correction or proceed to synthesis."""
    if state.get("needs_correction", False):
        return "correct"
    return "synthesize"


# ── Build the LangGraph Workflow ───────────────────────────────

def build_workflow() -> StateGraph:
    """Construct and compile the LangGraph agent workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router_planner", router_planner)
    workflow.add_node("tool_executor", tool_executor)
    workflow.add_node("validator", validator)
    workflow.add_node("synthesizer", synthesizer)

    # Define edges
    workflow.set_entry_point("router_planner")
    workflow.add_edge("router_planner", "tool_executor")
    workflow.add_edge("tool_executor", "validator")

    # Conditional: correction loop or final synthesis
    workflow.add_conditional_edges(
        "validator",
        should_correct,
        {
            "correct": "router_planner",
            "synthesize": "synthesizer",
        },
    )

    workflow.add_edge("synthesizer", END)

    return workflow


# ── Compiled Graph with Memory ─────────────────────────────────

_memory = MemorySaver()
_workflow = build_workflow()
compiled_graph = _workflow.compile(checkpointer=_memory)


async def run_agent(task: str, thread_id: str = "default") -> dict:
    """Execute the agent workflow for a given task."""
    config = {"configurable": {"thread_id": thread_id}}
    previous = await compiled_graph.aget_state(config)
    messages = list(previous.values.get("messages", [])) if previous and previous.values else []
    messages = [m for m in messages if not (
        m.get("role") == "system" and "previous tool call" in m.get("content", "")
    )]
    messages.append({"role": "user", "content": task})
    initial_state: AgentState = {
        "messages": messages,
        "current_task": task,
        "plan": "",
        "tool_calls": [],
        "tool_results": [],
        "correction_count": 0,
        "needs_correction": False,
        "final_response": "",
        "error": "",
        "trace": [],
    }

    # Use ainvoke for a single run
    final_state = await compiled_graph.ainvoke(initial_state, config=config)

    return {
        "response": final_state.get("final_response", "No response generated."),
        "trace": final_state.get("trace", []),
        "thread_id": thread_id,
        "tool_results": final_state.get("tool_results", []),
    }


async def stream_agent(task: str, thread_id: str = "default"):
    """Stream agent workflow steps as they execute."""
    config = {"configurable": {"thread_id": thread_id}}
    previous = await compiled_graph.aget_state(config)
    messages = list(previous.values.get("messages", [])) if previous and previous.values else []
    messages = [m for m in messages if not (
        m.get("role") == "system" and "previous tool call" in m.get("content", "")
    )]
    messages.append({"role": "user", "content": task})
    initial_state: AgentState = {
        "messages": messages,
        "current_task": task,
        "plan": "",
        "tool_calls": [],
        "tool_results": [],
        "correction_count": 0,
        "needs_correction": False,
        "final_response": "",
        "error": "",
        "trace": [],
    }

    async for event in compiled_graph.astream(initial_state, config=config):
        for node_name, node_output in event.items():
            yield {
                "node": node_name,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "plan": node_output.get("plan", ""),
                    "tool_results": [
                        {"tool": tr["tool"], "result_preview": tr["result"][:300]}
                        for tr in node_output.get("tool_results", [])
                    ],
                    "needs_correction": node_output.get("needs_correction", False),
                    "final_response": node_output.get("final_response", ""),
                },
            }
