"""CommitmentOS environment — multi-turn personal task management with
temporal commitment coherence tracking.

Episode lifecycle:
  1. reset()  -> agent receives scenario briefing + calendar + inbox
  2. step()   -> agent makes one tool call per step (done=False)
  3. step(submit_plan) or max_steps reached -> grading + done=True
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from constants import AUTHOR, PROJECT_DESCRIPTION, PROJECT_NAME, VERSION
from models import CommitmentAction, CommitmentObservation, CommitmentState
from server.domain import ScenarioDef
from server.world import WorldState


class CommitmentEnvironment(
    Environment[CommitmentAction, CommitmentObservation, CommitmentState]
):
    def __init__(self) -> None:
        super().__init__()
        self._world: Optional[WorldState] = None
        self._scenario: Optional[ScenarioDef] = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._last_tool_result: str = ""
        self._last_breakdown: dict[str, float] = {}
        self._last_feedback: str = ""

    # ------------------------------------------------------------------
    # Task selection
    # ------------------------------------------------------------------

    def _select_scenario(
        self,
        scenario_id: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> ScenarioDef:
        from server.tasks import get_all_scenarios, get_scenario, get_scenarios_by_difficulty

        if scenario_id:
            s = get_scenario(scenario_id)
            if s is None:
                raise ValueError(f"Unknown scenario_id: {scenario_id}")
            return s
        if difficulty:
            candidates = get_scenarios_by_difficulty(difficulty)
            if not candidates:
                raise ValueError(f"No scenarios for difficulty: {difficulty}")
            return random.choice(candidates)
        return random.choice(list(get_all_scenarios().values()))

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CommitmentObservation:
        if seed is not None:
            random.seed(seed)

        scenario = self._select_scenario(
            scenario_id=kwargs.get("scenario_id") or kwargs.get("task_id"),
            difficulty=kwargs.get("difficulty"),
        )
        self._scenario = scenario
        self._world = WorldState(scenario)
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._last_tool_result = ""
        self._last_breakdown = {}
        self._last_feedback = "New episode started. Read the briefing and use tools to manage the situation."

        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: CommitmentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CommitmentObservation:
        if self._world is None or self._scenario is None:
            raise ValueError("No active episode. Call reset() first.")
        if self._done:
            raise ValueError("Episode already completed. Call reset() to start a new one.")

        self._step_count += 1
        self._world.step_count = self._step_count

        at = action.action_type.lower().strip()

        if at == "submit_plan" or self._step_count >= self._scenario.max_steps:
            return self._finish_episode()

        step_reward = 0.0
        tool_result, dispatch_status = self._dispatch_tool(action, at)
        self._last_tool_result = tool_result

        if dispatch_status == "conflict":
            step_reward = -0.05
        elif dispatch_status == "success" and at in ("schedule_meeting", "reschedule_event", "send_email", "book_restaurant"):
            step_reward = 0.05

        self._cumulative_reward += step_reward
        self._last_feedback = ""
        self._last_breakdown = {}

        return self._build_observation(reward=step_reward, done=False)

    def _finish_episode(self) -> CommitmentObservation:
        from server.graders import grade_scenario

        assert self._world is not None
        assert self._scenario is not None

        total_reward, breakdown, feedback = grade_scenario(
            self._scenario, self._world,
        )
        self._done = True
        self._cumulative_reward += total_reward
        self._last_breakdown = breakdown
        self._last_feedback = feedback
        self._last_tool_result = "Plan submitted. Episode graded."

        return self._build_observation(reward=total_reward, done=True)

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, action: CommitmentAction, at: str) -> tuple[str, str]:
        assert self._world is not None
        turn = self._step_count

        if at == "view_calendar":
            return self._world.view_calendar(action.date), "info"
        elif at == "check_availability":
            return self._world.check_availability(action.person), "info"
        elif at == "search_restaurants":
            return self._world.search_restaurants(
                cuisine=action.cuisine,
                max_price=action.max_price,
                dietary=action.dietary,
                max_distance_miles=action.max_distance_miles,
                near_airport=action.near_airport,
            ), "info"
        elif at == "schedule_meeting":
            result = self._world.schedule_meeting(
                title=action.title,
                date=action.date,
                time=action.time,
                duration_min=action.duration_min,
                participants=action.participants,
                location=action.location,
                turn=turn,
            )
            status = "conflict" if result.startswith("CONFLICT:") else "success"
            return result, status
        elif at == "reschedule_event":
            result = self._world.reschedule_event(
                event_id=action.event_id,
                new_time=action.new_time,
                turn=turn,
            )
            status = "conflict" if result.startswith("CONFLICT:") else ("error" if "not found" in result.lower() else "success")
            return result, status
        elif at == "cancel_event":
            result = self._world.cancel_event(action.event_id, turn=turn)
            status = "error" if "not found" in result.lower() else "success"
            return result, status
        elif at == "send_email":
            return self._world.send_email(
                to=action.to,
                subject=action.subject,
                body=action.body,
                turn=turn,
            ), "success"
        elif at == "book_restaurant":
            result = self._world.book_restaurant(action.restaurant_name, turn=turn)
            status = "error" if "not found" in result.lower() else "success"
            return result, status
        else:
            return (
                f"Unknown action_type: '{at}'. Valid types: view_calendar, check_availability, search_restaurants, schedule_meeting, reschedule_event, cancel_event, send_email, book_restaurant, submit_plan",
                "error",
            )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, *, reward: float, done: bool) -> CommitmentObservation:
        assert self._world is not None
        assert self._scenario is not None

        return CommitmentObservation(
            scenario_id=self._scenario.scenario_id,
            difficulty=self._scenario.difficulty,
            briefing=self._scenario.briefing if self._step_count == 0 else "",
            tool_result=self._last_tool_result,
            calendar_snapshot=self._world.get_calendar_snapshot(),
            inbox=self._world.get_inbox_snapshot(),
            pending_commitments=len(self._world.get_active_commitments()),
            step_number=self._step_count,
            max_steps=self._scenario.max_steps,
            reward=reward,
            reward_breakdown=self._last_breakdown,
            done=done,
            feedback=self._last_feedback,
        )

    # ------------------------------------------------------------------
    # State property
    # ------------------------------------------------------------------

    @property
    def state(self) -> CommitmentState:
        from server.tasks import get_all_scenarios

        violations = self._world.get_silent_violations() if self._world else []
        return CommitmentState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            scenario_id=self._scenario.scenario_id if self._scenario else "",
            difficulty=self._scenario.difficulty if self._scenario else "",
            completed=self._done,
            cumulative_reward=self._cumulative_reward,
            commitment_count=len(self._world.commitment_ledger) if self._world else 0,
            violation_count=len(violations),
            available_tasks=list(get_all_scenarios().keys()),
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name=PROJECT_NAME,
            description=PROJECT_DESCRIPTION,
            version=VERSION,
            author=AUTHOR,
        )
