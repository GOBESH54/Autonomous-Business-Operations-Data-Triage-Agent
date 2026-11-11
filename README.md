# MCP Agentic Workflow Engine

A runnable business operations agent built with FastAPI, LangGraph, Google's Gemini model (optional), and the official Python MCP SDK. In the default offline mode, a deterministic local planner keeps the project usable without an API key.

## Architecture

```text
Client
  | REST / SSE
  v
FastAPI (app/main.py)
  |
  v
LangGraph StateGraph -- MemorySaver (thread_id)
  | Planner -> MCP Executor -> Validator --retry--> Planner
  |                                      \----------> Synthesizer
  v
MCP Python Client (stdio / JSON-RPC)
  |
  v
FastMCP server (mcp_server/server.py)
  |-- query_business_db       -> read-only SQLite
  |-- search_policy_docs      -> local TF-IDF retrieval over Markdown
  |-- create_action_ticket    -> SQLite action_tickets table
  `-- config://schema         -> database schema resource
```

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python seed_data.py
```

Copy `.env.example` to `.env` and optionally set `GEMINI_API_KEY`. Without a key, the deterministic mock planner is used. Start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive API documentation is at `http://localhost:8000/docs`.

## API

### `GET /health`

Returns service health and version.

### `POST /api/v1/agent/run`

```json
{
  "task": "What is our SLA policy for critical tickets?",
  "thread_id": "operations-demo"
}
```

Returns the synthesized response, thread ID, tool results, and node/tool execution trace. Omit `thread_id` to start a new conversation.

### `GET /api/v1/agent/stream?task=...&thread_id=...`

Streams workflow node updates as Server-Sent Events. Each event has an `agent_step` type; the stream ends with `complete` or `error`.

## MCP capabilities

| Name | Kind | Purpose |
| --- | --- | --- |
| `query_business_db(sql_query)` | Tool | Executes a single read-only `SELECT` against `data/business.db`. |
| `search_policy_docs(query)` | Tool | Retrieves matching local SOP excerpts with TF-IDF cosine scoring. |
| `create_action_ticket(customer_id, issue_summary, priority, resolution_action)` | Tool | Records a validated workflow action and returns its ID. |
| `config://schema` | Resource | Exposes table and column definitions to MCP clients. |

The database query tool opens SQLite in read-only mode and rejects non-`SELECT` statements and destructive SQL keywords. Ticket creation is the only business database write exposed by the server. Policy retrieval uses a dependency-light TF-IDF implementation, so it works without network access or embedding downloads.

## LangGraph lifecycle

1. **Router / Planner** reads the current task and selects a tool. Gemini is used when configured; otherwise the deterministic mock planner runs.
2. **Tool Executor** launches the MCP server over stdio, initializes an MCP `ClientSession`, discovers tools with `list_tools`, and calls the requested tool. Schema access uses `read_resource` for `config://schema`.
3. **Validator** detects SQL, database, and MCP tool errors. It sends the error back to the planner for up to three correction attempts.
4. **Final Synthesizer** combines the tool result into a user-facing response.
5. **MemorySaver** checkpoints graph state by `thread_id` for conversation continuity in the process lifetime. The in-memory checkpointer is not durable across restarts or multiple workers.

## Seed data and tests

`seed_data.py` creates sample customers, orders, support tickets (including urgent/SLA cases), action ticket schema, and four policy documents. It is safe to run repeatedly; it resets the local sample database.

```bash
python seed_data.py
pytest -q
```

The end-to-end tests cover seeded data, SQL safety, retrieval, ticket creation, MCP stdio discovery and resource reads, graph correction behavior, REST endpoints, and SSE responses.

## Docker

```bash
docker build -t mcp-agentic-workflow-engine .
docker run --rm -p 8000:8000 mcp-agentic-workflow-engine
```

Set `GEMINI_API_KEY` in the container environment to enable Gemini. The image seeds its sample database during build.

## Guide

**MCP client/server lifecycle:** the agent starts the local MCP server as a stdio subprocess, performs protocol initialization, discovers available tools, then invokes a tool or reads a resource. MCP carries structured JSON-RPC messages over the process streams; the agent consumes protocol results rather than importing server functions. The server owns tool validation and data access.

**LangGraph transitions:** the planner writes selected calls into graph state, the executor adds MCP results and trace entries, and the validator routes either back to planning or forward to synthesis. A bounded retry count prevents infinite correction loops. `MemorySaver` associates checkpoints with a conversation's `thread_id`.

**RAG choice:** policy Markdown is tokenized locally and ranked with TF-IDF cosine similarity. The retriever returns excerpts with source filenames. This provides a no-download fallback; an embedding model and FAISS index could replace it when semantic vectors are available.

## Project layout

```text
app/main.py                 FastAPI REST and SSE API
agent/workflow.py           LangGraph nodes, memory, MCP client adapter
mcp_server/server.py        FastMCP tools and schema resource
seed_data.py                Sample SQLite and policy document seeder
data/policies/              Company policy Markdown
tests/test_end_to_end.py    Integration and API tests
```
