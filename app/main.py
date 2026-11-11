"""
FastAPI Backend: Exposes the LangGraph + MCP agent workflow via REST API and SSE streaming.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.workflow import run_agent, stream_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    print("🚀 MCP Agentic Workflow Engine started")
    print("   Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health")
    yield


# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(
    title="MCP Agentic Workflow Engine",
    description=(
        "Production-style API for an agentic AI system powered by LangGraph, "
        "Model Context Protocol (MCP), and RAG memory. Executes natural language "
        "business operations tasks using autonomous tool orchestration."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ────────────────────────────────────
class AgentRunRequest(BaseModel):
    """Request body for the agent run endpoint."""

    task: str = Field(
        ...,
        description="Natural language business task for the agent to execute.",
        min_length=1,
        max_length=5000,
        json_schema_extra={"examples": ["Show me all enterprise customers with open support tickets"]},
    )
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread ID for multi-turn memory. Auto-generated if not provided.",
    )


class ToolResult(BaseModel):
    """A single MCP tool execution result."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""


class AgentRunResponse(BaseModel):
    """Response from the agent run endpoint."""

    response: str = Field(description="Final synthesized response from the agent.")
    thread_id: str = Field(description="Thread ID for continuing the conversation.")
    trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full trace of agent reasoning steps and MCP tool executions.",
    )
    tool_results: list[ToolResult] = Field(
        default_factory=list,
        description="Results from each MCP tool invocation.",
    )
    timestamp: str = Field(description="Response timestamp in ISO 8601 format.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str = "mcp-agentic-workflow-engine"
    version: str = "1.0.0"
    timestamp: str = ""


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint. Returns service status and version."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/api/v1/agent/run", response_model=AgentRunResponse, tags=["Agent"])
async def agent_run(request: AgentRunRequest):
    """Execute a business operations task through the LangGraph + MCP agent workflow.

    The agent will:
    1. Analyze the task and create an execution plan
    2. Execute relevant MCP tools (database queries, policy search, action tickets)
    3. Validate results and self-correct if errors are detected
    4. Synthesize a final comprehensive response

    Supports multi-turn conversations via `thread_id` for contextual follow-ups.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        result = await run_agent(task=request.task, thread_id=thread_id)

        tool_results = []
        for tr in result.get("tool_results", []):
            tool_results.append(
                ToolResult(
                    tool=tr.get("tool", ""),
                    args=tr.get("args", {}),
                    result=tr.get("result", ""),
                )
            )

        return AgentRunResponse(
            response=result.get("response", "No response generated."),
            thread_id=thread_id,
            trace=result.get("trace", []),
            tool_results=tool_results,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}",
        )


@app.get("/api/v1/agent/stream", tags=["Agent"])
async def agent_stream(
    task: str = Query(
        ...,
        description="Natural language business task for the agent.",
        min_length=1,
    ),
    thread_id: str = Query(
        default="",
        description="Thread ID for multi-turn memory.",
    ),
):
    """Stream real-time agent reasoning steps and MCP tool executions via Server-Sent Events (SSE).

    Each SSE event contains:
    - `node`: The current workflow node (router_planner, tool_executor, validator, synthesizer)
    - `timestamp`: When the step executed
    - `data`: Partial results including plans, tool outputs, and corrections
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())

    async def event_generator():
        try:
            async for step in stream_agent(task=task, thread_id=thread_id):
                event_data = json.dumps(step, default=str)
                yield f"event: agent_step\ndata: {event_data}\n\n"

            # Send completion event
            completion = json.dumps({
                "node": "complete",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"status": "completed", "thread_id": thread_id},
            })
            yield f"event: complete\ndata: {completion}\n\n"

        except Exception as e:
            error_data = json.dumps({
                "node": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"error": str(e)},
            })
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )





# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

