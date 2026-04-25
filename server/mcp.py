"""MCP JSON-RPC 2.0 endpoint for OpenEnv validator compliance."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from constants import PROJECT_NAME, VERSION
from models import CommitmentAction

router = APIRouter()

_CAPABILITIES = {
    "tools": {"listChanged": False},
    "resources": {"subscribe": False, "listChanged": False},
}

# MCP tool names must not collide with OpenEnv reserved names (reset/step/state/close).
# HTTP routes remain POST /reset, POST /step, GET /state — this list is only for MCP discovery.
_TOOLS = [
    {
        "name": "cos_episode_reset",
        "description": "Start a new CommitmentOS episode (maps to HTTP POST /reset)",
        "inputSchema": CommitmentAction.model_json_schema(),
    },
    {
        "name": "cos_environment_step",
        "description": "Execute one tool call in the current episode (maps to HTTP POST /step)",
        "inputSchema": CommitmentAction.model_json_schema(),
    },
    {
        "name": "cos_session_snapshot",
        "description": "Get current episode state (maps to HTTP GET /state)",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _jsonrpc_response(rpc_id: object, result: dict) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": result})


def _jsonrpc_error(rpc_id: object, code: int, message: str) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}})


@router.post("/mcp")
async def mcp_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return _jsonrpc_error(None, -32700, "Parse error")

    rpc_id = body.get("id")
    method = body.get("method", "")

    if method == "initialize":
        return _jsonrpc_response(rpc_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": _CAPABILITIES,
            "serverInfo": {"name": PROJECT_NAME, "version": VERSION},
        })

    if method == "tools/list":
        return _jsonrpc_response(rpc_id, {"tools": _TOOLS})

    return _jsonrpc_error(rpc_id, -32601, f"Method not found: {method}")
