"""
MCP Server: Exposes business tools and resources via the Model Context Protocol.
Uses FastMCP from the official Python MCP SDK.
"""
import os
import re
import sqlite3
import math
from collections import Counter
try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # MCP SDK 2.x renamed FastMCP; retain local compatibility.
    from mcp.server.mcpserver import MCPServer

    class FastMCP(MCPServer):
        """Small compatibility adapter; production dependency is pinned to MCP 1.x."""

        def __init__(self, name: str, version: str = "", description: str = ""):
            super().__init__(name=name, version=version, description=description)

        def run(self) -> None:
            import anyio

            anyio.run(self.run_stdio_async)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "business.db")
POLICIES_DIR = os.path.join(BASE_DIR, "data", "policies")

# ── Initialize MCP Server ──────────────────────────────────────
mcp_app = FastMCP(
    "BusinessOps MCP Server",
    version="1.0.0",
    description="MCP server providing business operations tools: DB queries, policy search, and action ticket creation.",
)

# ── Blocked SQL patterns for safety ────────────────────────────
_DESTRUCTIVE_PATTERNS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


def _get_db_connection() -> sqlite3.Connection:
    """Return a read-only SQLite connection."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _get_rw_connection() -> sqlite3.Connection:
    """Return a read-write SQLite connection (for action tickets only)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Tool 1: Query Business Database ───────────────────────────
@mcp_app.tool()
def query_business_db(sql_query: str) -> str:
    """Execute a read-only SQL query against the business database.

    The database contains tables: customers, orders, support_tickets, action_tickets.
    Only SELECT queries are allowed. Destructive operations are blocked.

    Args:
        sql_query: A valid SQL SELECT query.

    Returns:
        Query results as formatted text, or an error message.
    """
    # Safety check: block destructive queries
    if _DESTRUCTIVE_PATTERNS.search(sql_query):
        return "ERROR: Only SELECT queries are permitted. Destructive operations (DROP, DELETE, UPDATE, INSERT, ALTER, etc.) are blocked."

    stripped = sql_query.strip().rstrip(";").strip()
    if not stripped.upper().startswith("SELECT"):
        return "ERROR: Only SELECT queries are permitted. Your query must start with SELECT."

    try:
        conn = _get_db_connection()
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return "Query returned 0 rows."

        # Format output as a readable table
        columns = [desc[0] for desc in cur.description]
        header = " | ".join(columns)
        separator = "-+-".join("-" * max(len(c), 8) for c in columns)
        lines = [header, separator]

        for row in rows[:100]:  # Limit to 100 rows
            line = " | ".join(str(v) if v is not None else "NULL" for v in row)
            lines.append(line)

        result = "\n".join(lines)
        if len(rows) > 100:
            result += f"\n... ({len(rows) - 100} more rows truncated)"

        return f"Query returned {len(rows)} row(s):\n\n{result}"

    except sqlite3.Error as e:
        return f"SQL_ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


# ── Tool 2: Search Policy Documents ───────────────────────────

# Build TF-IDF-like index at module load time
_policy_index: dict[str, dict] | None = None


def _build_policy_index() -> dict[str, dict]:
    """Build a simple TF-IDF index over policy markdown documents."""
    global _policy_index
    if _policy_index is not None:
        return _policy_index

    docs: dict[str, dict] = {}
    if not os.path.isdir(POLICIES_DIR):
        return docs

    for fname in os.listdir(POLICIES_DIR):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(POLICIES_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        # Tokenize
        tokens = re.findall(r"[a-z0-9]+", content.lower())
        tf = Counter(tokens)
        total = len(tokens) or 1
        docs[fname] = {
            "content": content,
            "tokens": tokens,
            "tf": {t: c / total for t, c in tf.items()},
            "path": fpath,
        }

    # Compute IDF across all docs
    num_docs = len(docs) or 1
    all_terms: set[str] = set()
    for d in docs.values():
        all_terms.update(d["tf"].keys())

    idf: dict[str, float] = {}
    for term in all_terms:
        doc_freq = sum(1 for d in docs.values() if term in d["tf"])
        idf[term] = math.log((num_docs + 1) / (doc_freq + 1)) + 1

    for d in docs.values():
        d["tfidf"] = {t: d["tf"][t] * idf.get(t, 1.0) for t in d["tf"]}

    _policy_index = docs
    return docs


def _search_docs(query: str, top_k: int = 3) -> list[tuple[str, float, str]]:
    """Score policy docs against query using TF-IDF cosine similarity."""
    index = _build_policy_index()
    if not index:
        return []

    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    query_tf = Counter(query_tokens)
    total = len(query_tokens) or 1
    query_vec = {t: c / total for t, c in query_tf.items()}

    results = []
    for fname, doc in index.items():
        # Compute cosine similarity
        common_terms = set(query_vec.keys()) & set(doc["tfidf"].keys())
        if not common_terms:
            continue

        dot_product = sum(query_vec[t] * doc["tfidf"][t] for t in common_terms)
        q_norm = math.sqrt(sum(v ** 2 for v in query_vec.values()))
        d_norm = math.sqrt(sum(v ** 2 for v in doc["tfidf"].values()))
        if q_norm == 0 or d_norm == 0:
            continue

        similarity = dot_product / (q_norm * d_norm)
        results.append((fname, similarity, doc["content"]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


@mcp_app.tool()
def search_policy_docs(query: str) -> str:
    """Search company SOP and policy documents for relevant operational rules.

    Uses TF-IDF keyword retrieval over local markdown policy documents
    covering SLA, refund, escalation, and data handling policies.

    Args:
        query: Natural language search query about company policies or procedures.

    Returns:
        Relevant policy excerpts with source document names.
    """
    results = _search_docs(query, top_k=2)

    if not results:
        return "No relevant policy documents found for the given query."

    output_parts = []
    for fname, score, content in results:
        # Extract most relevant section
        sections = re.split(r"\n##\s+", content)
        query_lower = query.lower()
        best_section = max(
            sections,
            key=lambda s: sum(1 for w in query_lower.split() if w in s.lower()),
        )
        # Truncate long sections
        if len(best_section) > 1500:
            best_section = best_section[:1500] + "\n... (truncated)"

        output_parts.append(
            f"📄 **{fname}** (relevance: {score:.3f})\n\n{best_section.strip()}"
        )

    return "\n\n---\n\n".join(output_parts)


# ── Tool 3: Create Action Ticket ──────────────────────────────
@mcp_app.tool()
def create_action_ticket(
    customer_id: int,
    issue_summary: str,
    priority: str,
    resolution_action: str,
) -> str:
    """Create an automated workflow action ticket in the database.

    This records a structured resolution action that the operations team
    should execute. Returns a confirmation with the new ticket ID.

    Args:
        customer_id: ID of the customer this action relates to.
        issue_summary: Brief summary of the issue being addressed.
        priority: Priority level — one of: critical, high, medium, low.
        resolution_action: The specific action to be taken to resolve the issue.

    Returns:
        Confirmation message with the created ticket ID.
    """
    priority = priority.lower().strip()
    if priority not in ("critical", "high", "medium", "low"):
        return f"ERROR: Invalid priority '{priority}'. Must be one of: critical, high, medium, low."

    try:
        conn = _get_rw_connection()
        cur = conn.cursor()

        # Verify customer exists
        cur.execute("SELECT id, name FROM customers WHERE id = ?", (customer_id,))
        customer = cur.fetchone()
        if not customer:
            conn.close()
            return f"ERROR: Customer with id={customer_id} not found."

        from datetime import datetime

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
            f"  Status: CREATED\n"
            f"  Created: {now}"
        )
    except sqlite3.Error as e:
        return f"DATABASE_ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


# ── Resource: Database Schema ──────────────────────────────────
@mcp_app.resource("config://schema")
def get_database_schema() -> str:
    """Return the database table schemas so the agent can inspect tables before writing SQL."""
    try:
        conn = _get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cur.fetchall()]

        schema_parts = ["# Business Database Schema\n"]
        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            columns = cur.fetchall()
            schema_parts.append(f"\n## Table: `{table}`\n")
            schema_parts.append("| Column | Type | Nullable | Default | PK |")
            schema_parts.append("|--------|------|----------|---------|-----|")
            for col in columns:
                name = col["name"]
                col_type = col["type"]
                nullable = "YES" if not col["notnull"] else "NO"
                default = col["dflt_value"] if col["dflt_value"] else "-"
                pk = "✓" if col["pk"] else ""
                schema_parts.append(f"| {name} | {col_type} | {nullable} | {default} | {pk} |")

            # Row count
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            schema_parts.append(f"\n*{count} rows*")

        conn.close()
        return "\n".join(schema_parts)

    except Exception as e:
        return f"Error retrieving schema: {e}"


# ── Entrypoint ─────────────────────────────────────────────────
if __name__ == "__main__":
    mcp_app.run()

