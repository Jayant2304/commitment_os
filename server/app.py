"""FastAPI composition root — wires environment, MCP, and custom endpoints."""

from __future__ import annotations

import os

from openenv.core.env_server import create_fastapi_app

from constants import PROJECT_DESCRIPTION, VERSION
from models import CommitmentAction, CommitmentObservation, CommitmentState
from server.environment import CommitmentEnvironment
from server.mcp import router as mcp_router
from server.tasks import get_scenario_ids_grouped

_shared_env = CommitmentEnvironment()

app = create_fastapi_app(
    env=lambda: _shared_env,
    action_cls=CommitmentAction,
    observation_cls=CommitmentObservation,
)

app.title = "CommitmentOS"
app.description = PROJECT_DESCRIPTION
app.version = VERSION

app.routes[:] = [
    r for r in app.routes
    if not (hasattr(r, "path") and r.path in ("/state", "/mcp"))
]


@app.get("/state", response_model=CommitmentState)
def get_state() -> CommitmentState:
    return _shared_env.state


@app.get("/tasks")
def list_tasks() -> dict[str, list[str]]:
    return get_scenario_ids_grouped()


app.include_router(mcp_router)


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
