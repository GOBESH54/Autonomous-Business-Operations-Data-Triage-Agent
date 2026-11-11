"""
End-to-end tests for the MCP Agentic Workflow Engine.

Tests cover:
1. Database seeding and integrity
2. MCP tool functions (query, search, tickets)
3. Safety checks (SQL injection prevention)
4. LangGraph workflow execution
5. Self-correction loop
6. FastAPI endpoints (health, run, stream)
"""
import asyncio
import json
import os
import sqlite3
import sys

import pytest
import httpx
from fastapi.testclient import TestClient

# ── Ensure project root is on path ────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# ── Run seed_data before any tests ─────────────────────────────
from seed_data import seed_database, seed_policy_docs, DB_PATH, POLICIES_DIR

# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Seed the database and policy docs once for the entire test session."""
    seed_database()
    seed_policy_docs()
    yield
    # Cleanup is optional — leave data for inspection


@pytest.fixture
def db_connection():
    """Provide a read-only database connection."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def test_client():
    """Provide a FastAPI TestClient."""
    from app.main import app
    with TestClient(app) as client:
        yield client


# ════════════════════════════════════════════════════════════════
# 1. Database Seeding Tests
# ════════════════════════════════════════════════════════════════

class TestDatabaseSeeding:
    """Verify the seed script creates all tables with correct data."""

    def test_database_file_exists(self):
        assert os.path.exists(DB_PATH), f"Database not found at {DB_PATH}"

    def test_customers_table_populated(self, db_connection):
        cur = db_connection.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM customers")
        count = cur.fetchone()["cnt"]
        assert count == 10, f"Expected 10 customers, got {count}"

    def test_orders_table_populated(self, db_connection):
        cur = db_connection.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM orders")
        count = cur.fetchone()["cnt"]
        assert count == 30, f"Expected 30 orders, got {count}"

    def test_support_tickets_populated(self, db_connection):
        cur = db_connection.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM support_tickets")
        count = cur.fetchone()["cnt"]
        assert count >= 10, f"Expected at least 10 support tickets, got {count}"

    def test_action_tickets_table_exists(self, db_connection):
        cur = db_connection.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='action_tickets'")
        table = cur.fetchone()
        assert table is not None, "action_tickets table not found"

    def test_customer_fields(self, db_connection):
        cur = db_connection.cursor()
        cur.execute("SELECT * FROM customers WHERE id = 1")
        customer = cur.fetchone()
        assert customer["name"] == "Alice Johnson"
        assert customer["email"] == "alice@acmecorp.com"
        assert customer["plan"] == "enterprise"

    def test_orders_have_valid_customer_ids(self, db_connection):
        cur = db_connection.cursor()
        cur.execute("""
            SELECT COUNT(*) as cnt FROM orders o
            LEFT JOIN customers c ON o.customer_id = c.id
            WHERE c.id IS NULL
        """)
        orphans = cur.fetchone()["cnt"]
        assert orphans == 0, f"Found {orphans} orders with invalid customer_id"

    def test_sla_breach_tickets_exist(self, db_connection):
        cur = db_connection.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM support_tickets WHERE priority = 'critical'")
        count = cur.fetchone()["cnt"]
        assert count >= 2, f"Expected at least 2 critical tickets, got {count}"


# ════════════════════════════════════════════════════════════════
# 2. Policy Document Tests
# ════════════════════════════════════════════════════════════════

class TestPolicyDocs:
    """Verify policy documents are created correctly."""

    def test_policies_directory_exists(self):
        assert os.path.isdir(POLICIES_DIR), f"Policies directory not found at {POLICIES_DIR}"

    def test_all_policy_files_created(self):
        expected_files = [
            "sla_policy.md",
            "refund_policy.md",
            "escalation_procedures.md",
            "data_handling_policy.md",
        ]
        actual_files = os.listdir(POLICIES_DIR)
        for f in expected_files:
            assert f in actual_files, f"Missing policy file: {f}"

    def test_policy_files_not_empty(self):
        for fname in os.listdir(POLICIES_DIR):
            fpath = os.path.join(POLICIES_DIR, fname)
            size = os.path.getsize(fpath)
            assert size > 100, f"Policy file {fname} is too small ({size} bytes)"


# ════════════════════════════════════════════════════════════════
# 3. MCP Tool Function Tests
# ════════════════════════════════════════════════════════════════

class TestMCPTools:
    """Test the MCP tool implementations directly."""

    def test_query_business_db_select(self):
        from agent.workflow import tool_query_business_db
        result = tool_query_business_db("SELECT COUNT(*) as total FROM customers")
        assert "1 row(s)" in result
        assert "total" in result

    def test_query_business_db_join(self):
        from agent.workflow import tool_query_business_db
        result = tool_query_business_db(
            "SELECT c.name, COUNT(o.id) as order_count "
            "FROM customers c LEFT JOIN orders o ON c.id = o.customer_id "
            "GROUP BY c.id ORDER BY order_count DESC LIMIT 3"
        )
        assert "row(s)" in result
        assert "name" in result

    def test_query_blocks_drop(self):
        from agent.workflow import tool_query_business_db
        result = tool_query_business_db("DROP TABLE customers")
        assert "ERROR" in result
        assert "blocked" in result.lower() or "permitted" in result.lower()

    def test_query_blocks_delete(self):
        from agent.workflow import tool_query_business_db
        result = tool_query_business_db("DELETE FROM customers WHERE id = 1")
        assert "ERROR" in result

    def test_query_blocks_update(self):
        from agent.workflow import tool_query_business_db
        result = tool_query_business_db("UPDATE customers SET name = 'hacked' WHERE id = 1")
        assert "ERROR" in result

    def test_query_blocks_insert(self):
        from agent.workflow import tool_query_business_db
        result = tool_query_business_db("INSERT INTO customers (name, email) VALUES ('x', 'x@x.com')")
        assert "ERROR" in result

    def test_query_invalid_sql(self):
        from agent.workflow import tool_query_business_db
        result = tool_query_business_db("SELECT * FROM nonexistent_table")
        assert "SQL_ERROR" in result or "ERROR" in result

    def test_query_empty_result(self):
        from agent.workflow import tool_query_business_db
        result = tool_query_business_db("SELECT * FROM customers WHERE id = 99999")
        assert "0 rows" in result

    def test_search_policy_docs_sla(self):
        from agent.workflow import tool_search_policy_docs
        result = tool_search_policy_docs("SLA breach response time escalation")
        assert "sla_policy.md" in result or "escalation" in result.lower()

    def test_search_policy_docs_refund(self):
        from agent.workflow import tool_search_policy_docs
        result = tool_search_policy_docs("refund cancellation eligibility")
        assert "refund" in result.lower()

    def test_search_policy_docs_data(self):
        from agent.workflow import tool_search_policy_docs
        result = tool_search_policy_docs("data retention PII compliance")
        assert "data" in result.lower()

    def test_create_action_ticket_success(self):
        from agent.workflow import tool_create_action_ticket
        result = tool_create_action_ticket(
            customer_id=1,
            issue_summary="Test issue for unit test",
            priority="medium",
            resolution_action="Verify and close",
        )
        assert "ACT-" in result
        assert "CREATED" in result
        assert "Alice Johnson" in result

    def test_create_action_ticket_invalid_customer(self):
        from agent.workflow import tool_create_action_ticket
        result = tool_create_action_ticket(
            customer_id=99999,
            issue_summary="Invalid customer test",
            priority="low",
            resolution_action="N/A",
        )
        assert "ERROR" in result
        assert "not found" in result.lower()

    def test_create_action_ticket_invalid_priority(self):
        from agent.workflow import tool_create_action_ticket
        result = tool_create_action_ticket(
            customer_id=1,
            issue_summary="Bad priority test",
            priority="urgent",  # Invalid
            resolution_action="N/A",
        )
        assert "ERROR" in result
        assert "priority" in result.lower()

    def test_get_schema(self):
        from agent.workflow import tool_get_schema
        result = tool_get_schema()
        assert "customers" in result
        assert "orders" in result
        assert "support_tickets" in result
        assert "action_tickets" in result


# ════════════════════════════════════════════════════════════════
# 4. LangGraph Workflow Tests
# ════════════════════════════════════════════════════════════════

class TestLangGraphWorkflow:
    """Test the LangGraph agent workflow execution."""

    @pytest.mark.asyncio
    async def test_run_agent_database_query(self):
        from agent.workflow import run_agent
        result = await run_agent(
            task="Show me the top customers by order count",
            thread_id="test-db-query",
        )
        assert "response" in result
        assert result["response"]  # Non-empty
        assert "trace" in result
        assert len(result["trace"]) > 0
        assert result["thread_id"] == "test-db-query"

    @pytest.mark.asyncio
    async def test_run_agent_policy_search(self):
        from agent.workflow import run_agent
        result = await run_agent(
            task="What is the SLA policy for critical tickets?",
            thread_id="test-policy",
        )
        assert result["response"]
        assert len(result["trace"]) > 0

    @pytest.mark.asyncio
    async def test_run_agent_action_ticket(self):
        from agent.workflow import run_agent
        result = await run_agent(
            task="Create an action ticket for customer issue resolution",
            thread_id="test-ticket",
        )
        assert result["response"]
        assert len(result["trace"]) > 0

    @pytest.mark.asyncio
    async def test_run_agent_schema_query(self):
        from agent.workflow import run_agent
        result = await run_agent(
            task="Show me the database schema and table structure",
            thread_id="test-schema",
        )
        assert result["response"]

    @pytest.mark.asyncio
    async def test_workflow_trace_has_all_nodes(self):
        from agent.workflow import run_agent
        result = await run_agent(
            task="Count all customers in the database",
            thread_id="test-trace-nodes",
        )
        trace_nodes = [t["node"] for t in result["trace"]]
        assert "router_planner" in trace_nodes
        assert "tool_executor" in trace_nodes
        assert "validator" in trace_nodes
        assert "synthesizer" in trace_nodes

    @pytest.mark.asyncio
    async def test_self_correction_loop(self):
        """Test that the validator detects errors and can trigger re-planning."""
        from agent.workflow import validator

        # Simulate a state with an error result
        error_state = {
            "messages": [{"role": "user", "content": "test"}],
            "tool_results": [
                {"tool": "query_business_db", "args": {}, "result": "SQL_ERROR: no such table: foo"}
            ],
            "correction_count": 0,
            "trace": [],
            "needs_correction": False,
        }

        result = validator(error_state)
        assert result["needs_correction"] is True
        assert result["correction_count"] == 1

    @pytest.mark.asyncio
    async def test_correction_loop_max_limit(self):
        """Test that correction loop respects MAX_CORRECTION_LOOPS."""
        from agent.workflow import validator, MAX_CORRECTION_LOOPS

        state = {
            "messages": [{"role": "user", "content": "test"}],
            "tool_results": [
                {"tool": "query_business_db", "args": {}, "result": "SQL_ERROR: persistent error"}
            ],
            "correction_count": MAX_CORRECTION_LOOPS,
            "trace": [],
            "needs_correction": False,
        }

        result = validator(state)
        assert result["needs_correction"] is False  # Should NOT correct beyond max

    @pytest.mark.asyncio
    async def test_stream_agent_yields_events(self):
        from agent.workflow import stream_agent
        events = []
        async for event in stream_agent(
            task="List all customers",
            thread_id="test-stream",
        ):
            events.append(event)

        assert len(events) > 0
        node_names = [e["node"] for e in events]
        assert "router_planner" in node_names


# ════════════════════════════════════════════════════════════════
# 5. FastAPI Endpoint Tests
# ════════════════════════════════════════════════════════════════

class TestFastAPIEndpoints:
    """Test the FastAPI REST endpoints."""

    def test_health_endpoint(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "mcp-agentic-workflow-engine"
        assert data["version"] == "1.0.0"

    def test_agent_run_endpoint(self, test_client):
        response = test_client.post(
            "/api/v1/agent/run",
            json={
                "task": "How many customers are on the enterprise plan?",
                "thread_id": "test-api-run",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"]  # Non-empty
        assert data["thread_id"] == "test-api-run"
        assert "trace" in data
        assert "timestamp" in data

    def test_agent_run_auto_thread_id(self, test_client):
        response = test_client.post(
            "/api/v1/agent/run",
            json={"task": "Show customer list"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"]  # Auto-generated, non-empty

    def test_agent_run_with_tool_results(self, test_client):
        response = test_client.post(
            "/api/v1/agent/run",
            json={
                "task": "Query the database for total order revenue",
                "thread_id": "test-tool-results",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "tool_results" in data

    def test_agent_run_empty_task_rejected(self, test_client):
        response = test_client.post(
            "/api/v1/agent/run",
            json={"task": ""},
        )
        assert response.status_code == 422  # Validation error

    def test_agent_stream_endpoint(self, test_client):
        response = test_client.get(
            "/api/v1/agent/stream",
            params={"task": "List all enterprise customers", "thread_id": "test-sse"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE events
        content = response.text
        assert "event:" in content
        assert "data:" in content

    def test_agent_stream_empty_task_rejected(self, test_client):
        response = test_client.get(
            "/api/v1/agent/stream",
            params={"task": ""},
        )
        assert response.status_code == 422

    def test_openapi_docs_available(self, test_client):
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/api/v1/agent/run" in schema["paths"]
        assert "/api/v1/agent/stream" in schema["paths"]


# ════════════════════════════════════════════════════════════════
# 6. Integration Tests
# ════════════════════════════════════════════════════════════════

class TestIntegration:
    """Full integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_workflow_database_to_response(self):
        """Test complete flow: task → planner → SQL tool → validate → synthesize."""
        from agent.workflow import run_agent
        result = await run_agent(
            task="Find all support tickets with critical priority that are still open",
            thread_id="integration-1",
        )
        assert result["response"]
        # Trace should show all workflow nodes executed
        trace_nodes = {t["node"] for t in result["trace"]}
        assert "router_planner" in trace_nodes
        assert "synthesizer" in trace_nodes

    @pytest.mark.asyncio
    async def test_full_workflow_policy_search(self):
        """Test complete flow with policy document retrieval."""
        from agent.workflow import run_agent
        result = await run_agent(
            task="What is our refund policy for enterprise customers?",
            thread_id="integration-2",
        )
        assert result["response"]

    def test_api_multiturn_conversation(self, test_client):
        """Test multi-turn conversation with shared thread_id."""
        thread_id = "multiturn-test"

        # Turn 1
        r1 = test_client.post(
            "/api/v1/agent/run",
            json={"task": "Show me the customers table", "thread_id": thread_id},
        )
        assert r1.status_code == 200
        assert r1.json()["thread_id"] == thread_id

        # Turn 2 (same thread)
        r2 = test_client.post(
            "/api/v1/agent/run",
            json={"task": "Now show me their orders", "thread_id": thread_id},
        )
        assert r2.status_code == 200
        assert r2.json()["thread_id"] == thread_id

