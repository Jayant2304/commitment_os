"""FastAPI composition root — wires environment, MCP, and custom endpoints."""

from __future__ import annotations

import os
from threading import Lock

from openenv.core.env_server import create_fastapi_app
from fastapi import Query
from pydantic import BaseModel

from constants import PROJECT_DESCRIPTION, VERSION
from models import CommitmentAction, CommitmentObservation, CommitmentState
from server.environment import CommitmentEnvironment
from server.mcp import router as mcp_router
from server.tasks import get_scenario_ids_grouped

_DEFAULT_SESSION_ID = "default"
_env_store: dict[str, CommitmentEnvironment] = {
    _DEFAULT_SESSION_ID: CommitmentEnvironment(),
}
_env_store_lock = Lock()


def _get_env(session_id: str) -> CommitmentEnvironment:
    """Return a per-session environment instance.

    This avoids cross-user state bleed from a single shared mutable environment.
    Clients can pass ``episode_id`` query param to isolate sessions.
    """
    with _env_store_lock:
        env = _env_store.get(session_id)
        if env is None:
            env = CommitmentEnvironment()
            _env_store[session_id] = env
        return env


class StepPayload(BaseModel):
    action: CommitmentAction

app = create_fastapi_app(
    env=lambda: _get_env(_DEFAULT_SESSION_ID),
    action_cls=CommitmentAction,
    observation_cls=CommitmentObservation,
)

app.title = "CommitmentOS"
app.description = PROJECT_DESCRIPTION
app.version = VERSION

app.routes[:] = [
    r for r in app.routes
    if not (hasattr(r, "path") and r.path in ("/state", "/mcp", "/reset", "/step"))
]


@app.post("/reset")
def reset_episode(
    task_id: str | None = Query(default=None),
    difficulty: str | None = Query(default=None),
    seed: int | None = Query(default=None),
    episode_id: str | None = Query(default=None),
) -> dict[str, object]:
    """Reset endpoint with explicit query-param support.

    The default OpenEnv route did not reliably propagate ``task_id`` from
    query params in this deployment setup, which made scenario selection
    non-deterministic for demos/evaluations.
    """
    session_id = episode_id or _DEFAULT_SESSION_ID
    env = _get_env(session_id)
    obs = env.reset(
        seed=seed,
        episode_id=session_id,
        task_id=task_id,
        difficulty=difficulty,
    )
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward),
        "done": bool(obs.done),
        "episode_id": session_id,
    }


@app.post("/step")
def step_episode(
    payload: StepPayload,
    episode_id: str | None = Query(default=None),
) -> dict[str, object]:
    session_id = episode_id or _DEFAULT_SESSION_ID
    env = _get_env(session_id)
    obs = env.step(payload.action)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward),
        "done": bool(obs.done),
        "episode_id": session_id,
    }


@app.get("/state", response_model=CommitmentState)
def get_state(episode_id: str | None = Query(default=None)) -> CommitmentState:
    session_id = episode_id or _DEFAULT_SESSION_ID
    return _get_env(session_id).state


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
