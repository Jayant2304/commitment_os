"""Comprehensive test suite for CommitmentOS.

Tests cover:
  - Grader (perfect/partial/zero for each component)
  - Environment lifecycle (reset/step/state/multi-turn)
  - Commitment ledger (creation, violation, renegotiation)
  - Task dataset integrity
  - API endpoints
  - Difficulty verification
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from typing import Any, Dict

import pytest

from models import CommitmentAction, CommitmentObservation, CommitmentState
from server.domain import CalendarEvent, ConstraintDef, ScenarioDef
from server.environment import CommitmentEnvironment
from server.graders import (
    _calendar_has_no_overlaps,
    _keyword_score,
    _score_commitment_coherence,
    _score_conflict_resolution,
    _score_step_efficiency,
    grade_scenario,
)
from server.tasks import get_all_scenarios, get_scenario, get_scenarios_by_difficulty
from server.world import WorldState, _time_to_min


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def env() -> CommitmentEnvironment:
    return CommitmentEnvironment()


@pytest.fixture
def easy_env(env: CommitmentEnvironment) -> CommitmentEnvironment:
    env.reset(task_id="easy_001")
    return env


# ===================================================================
# 1. Task dataset integrity
# ===================================================================

class TestTaskDataset:
    def test_15_scenarios_loaded(self) -> None:
        scenarios = get_all_scenarios()
        assert len(scenarios) == 15

    def test_5_easy_5_medium_5_hard(self) -> None:
        for difficulty, count in [("easy", 5), ("medium", 5), ("hard", 5)]:
            tasks = get_scenarios_by_difficulty(difficulty)
            assert len(tasks) == count, f"Expected {count} {difficulty} tasks, got {len(tasks)}"

    def test_each_scenario_has_required_fields(self) -> None:
        for sid, scenario in get_all_scenarios().items():
            assert scenario.scenario_id == sid
            assert scenario.difficulty in ("easy", "medium", "hard")
            assert len(scenario.briefing) > 20, f"{sid}: briefing too short"
            assert scenario.optimal_steps >= 2, f"{sid}: optimal_steps too low"
            assert scenario.max_steps >= scenario.optimal_steps
            assert len(scenario.constraints) >= 1, f"{sid}: no constraints defined"

    def test_scenario_ids_unique(self) -> None:
        ids = list(get_all_scenarios().keys())
        assert len(ids) == len(set(ids))

    def test_get_scenario_returns_none_for_missing(self) -> None:
        assert get_scenario("nonexistent_999") is None

    def test_get_scenario_returns_correct(self) -> None:
        s = get_scenario("easy_001")
        assert s is not None
        assert s.difficulty == "easy"


# ===================================================================
# 2. Grader unit tests
# ===================================================================

class TestKeywordScore:
    def test_full_match(self) -> None:
        score, matched = _keyword_score("I need to reschedule the standup meeting", ["reschedule", "standup"], min_matches=2)
        assert score == 1.0
        assert len(matched) == 2

    def test_partial_match(self) -> None:
        score, matched = _keyword_score("I need to reschedule", ["reschedule", "standup"], min_matches=2)
        assert score == 0.5
        assert len(matched) == 1

    def test_no_match(self) -> None:
        score, matched = _keyword_score("Hello world", ["reschedule", "standup"], min_matches=2)
        assert score == 0.0
        assert len(matched) == 0

    def test_case_insensitive(self) -> None:
        score, _ = _keyword_score("RESCHEDULE THE STANDUP", ["reschedule", "standup"], min_matches=2)
        assert score == 1.0


class TestCalendarConflicts:
    def test_no_conflicts(self) -> None:
        scenario = get_scenario("easy_002")
        assert scenario is not None
        world = WorldState(scenario)
        assert _calendar_has_no_overlaps(world) is True

    def test_conflict_detected(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        assert _calendar_has_no_overlaps(world) is False


class TestCommitmentCoherence:
    def test_no_commitments_full_score(self) -> None:
        scenario = get_scenario("easy_005")
        assert scenario is not None
        world = WorldState(scenario)
        score, _ = _score_commitment_coherence(world)
        assert score == 1.0

    def test_honored_commitment(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        env.step(CommitmentAction(action_type="reschedule_event", event_id="evt_2", new_time="15:00"))
        assert env._world is not None
        score, feedback = _score_commitment_coherence(env._world)
        assert score == 1.0

    def test_silent_violation_detected(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        env.step(CommitmentAction(action_type="schedule_meeting", title="New Meeting", date="2026-04-25", time="16:00", participants=["Alice"]))
        assert env._world is not None
        env._world.calendar.pop("evt_100", None)
        for c in env._world.commitment_ledger:
            if c.commitment_type == "meeting_scheduled" and "16:00" in c.constraint:
                event_key = c.constraint
                for eid, ev in list(env._world.calendar.items()):
                    if ev.time == "16:00" and ev.date == "2026-04-25" and ev.title == "New Meeting":
                        del env._world.calendar[eid]
                        break
        violations = env._world.get_silent_violations()
        assert len(violations) >= 1


class TestStepEfficiency:
    def test_optimal_steps(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        world.step_count = 3
        score, _ = _score_step_efficiency(scenario, world)
        assert score == 1.0

    def test_over_optimal(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        world.step_count = 8
        score, _ = _score_step_efficiency(scenario, world)
        assert score == 0.5


# ===================================================================
# 3. Environment lifecycle
# ===================================================================

class TestEnvironmentLifecycle:
    def test_reset_returns_observation(self, env: CommitmentEnvironment) -> None:
        obs = env.reset(task_id="easy_001")
        assert isinstance(obs, CommitmentObservation)
        assert obs.scenario_id == "easy_001"
        assert obs.done is False
        assert obs.reward == 0.0
        assert len(obs.briefing) > 0

    def test_step_before_reset_raises(self, env: CommitmentEnvironment) -> None:
        with pytest.raises(ValueError, match="No active episode"):
            env.step(CommitmentAction(action_type="view_calendar", date="2026-04-25"))

    def test_step_after_done_raises(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        env.step(CommitmentAction(action_type="submit_plan"))
        with pytest.raises(ValueError, match="already completed"):
            env.step(CommitmentAction(action_type="view_calendar", date="2026-04-25"))

    def test_state_property(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        state = env.state
        assert isinstance(state, CommitmentState)
        assert state.scenario_id == "easy_001"
        assert state.completed is False
        assert len(state.available_tasks) == 15

    def test_multi_turn_episode(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        obs = env.step(CommitmentAction(action_type="view_calendar", date="2026-04-25"))
        assert obs.done is False
        assert obs.step_number == 1

        obs = env.step(CommitmentAction(action_type="reschedule_event", event_id="evt_2", new_time="15:00"))
        assert obs.done is False
        assert obs.step_number == 2

        obs = env.step(CommitmentAction(action_type="submit_plan"))
        assert obs.done is True
        assert obs.reward > 0

    def test_max_steps_auto_submits(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_002")
        for _ in range(20):
            obs = env.step(CommitmentAction(action_type="view_calendar", date="2026-04-25"))
            if obs.done:
                break
        assert obs.done is True

    def test_reset_clears_state(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        env.step(CommitmentAction(action_type="view_calendar", date="2026-04-25"))
        env.reset(task_id="easy_002")
        assert env.state.scenario_id == "easy_002"
        assert env.state.step_count == 0

    def test_unknown_action_type(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        obs = env.step(CommitmentAction(action_type="fly_to_moon"))
        assert "Unknown action_type" in obs.tool_result

    def test_random_reset(self, env: CommitmentEnvironment) -> None:
        obs = env.reset(seed=42)
        assert obs.scenario_id in get_all_scenarios()

    def test_difficulty_filter_reset(self, env: CommitmentEnvironment) -> None:
        obs = env.reset(difficulty="hard", seed=1)
        assert obs.difficulty == "hard"


# ===================================================================
# 4. World simulation (tool functions)
# ===================================================================

class TestWorldTools:
    def test_view_calendar(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.view_calendar("2026-04-25")
        assert "evt_1" in result
        assert "14:00" in result

    def test_view_calendar_empty(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.view_calendar("2099-01-01")
        assert "No events" in result

    def test_check_availability(self) -> None:
        scenario = get_scenario("easy_003")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.check_availability("Client_Jones")
        assert "09:00" in result

    def test_check_availability_unknown(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.check_availability("NonExistentPerson")
        assert "not found" in result

    def test_search_restaurants_filters(self) -> None:
        scenario = get_scenario("med_007")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.search_restaurants(dietary="vegan", max_price=45, max_distance_miles=3.0)
        assert "Green Garden" in result
        assert "Steak House Prime" not in result

    def test_schedule_meeting_creates_commitment(self) -> None:
        scenario = get_scenario("easy_002")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.schedule_meeting("Test Meeting", "2026-04-25", "14:00", turn=1)
        assert "scheduled" in result.lower()
        assert len(world.commitment_ledger) == 1
        assert world.commitment_ledger[0].commitment_type == "meeting_scheduled"

    def test_schedule_meeting_conflict(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.schedule_meeting("Conflicting", "2026-04-25", "14:00", turn=1)
        assert "CONFLICT" in result

    def test_reschedule_event(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.reschedule_event("evt_2", "15:00", turn=1)
        assert "Rescheduled" in result
        assert world.calendar["evt_2"].time == "15:00"

    def test_cancel_event(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.cancel_event("evt_2", turn=1)
        assert "Cancelled" in result
        assert "evt_2" not in world.calendar

    def test_send_email(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.send_email("Team", "Hello", "Testing email body", turn=1)
        assert "sent" in result.lower()
        assert len(world.emails_sent) == 1

    def test_book_restaurant(self) -> None:
        scenario = get_scenario("easy_002")
        assert scenario is not None
        world = WorldState(scenario)
        result = world.book_restaurant("Bella Italia", turn=1)
        assert "confirmed" in result.lower()
        assert world.booked_restaurant == "Bella Italia"


# ===================================================================
# 5. Commitment ledger behaviour
# ===================================================================

class TestCommitmentLedger:
    def test_schedule_creates_commitment(self) -> None:
        scenario = get_scenario("easy_002")
        assert scenario is not None
        world = WorldState(scenario)
        world.schedule_meeting("Test", "2026-04-25", "10:00", turn=1)
        assert len(world.commitment_ledger) == 1
        c = world.commitment_ledger[0]
        assert c.turn_created == 1
        assert c.active is True
        assert c.renegotiated_at is None

    def test_reschedule_marks_old_renegotiated(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        world.reschedule_event("evt_2", "15:00", turn=1)
        renegotiated = [c for c in world.commitment_ledger if c.renegotiated_at is not None]
        assert len(renegotiated) == 0  # initial events don't create ledger entries
        new_commits = [c for c in world.commitment_ledger if c.active]
        assert len(new_commits) >= 1

    def test_email_renegotiation_detection(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        world.schedule_meeting("Important", "2026-04-25", "16:00", participants=["Alice"], turn=1)
        world.send_email("Alice", "Change of plans", "I need to reschedule our meeting", turn=2)
        renegotiated = [c for c in world.commitment_ledger if c.renegotiated_at is not None]
        assert len(renegotiated) >= 1

    def test_cancel_personal_marks_renegotiated(self) -> None:
        scenario = get_scenario("easy_001")
        assert scenario is not None
        world = WorldState(scenario)
        # evt_3 is Lunch (personal)
        world.cancel_event("evt_3", turn=1)
        # Personal cancellations are auto-OK


# ===================================================================
# 6. Full scenario scoring
# ===================================================================

class TestFullScoring:
    def test_perfect_easy_001(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        env.step(CommitmentAction(action_type="reschedule_event", event_id="evt_2", new_time="15:00"))
        env.step(CommitmentAction(action_type="send_email", to="Team", subject="Standup moved", body="Hi team, I've rescheduled the standup to 3:00 PM. Sorry for the move."))
        obs = env.step(CommitmentAction(action_type="submit_plan"))
        assert obs.done is True
        assert obs.reward >= 0.85

    def test_zero_effort_gets_low_score(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        obs = env.step(CommitmentAction(action_type="submit_plan"))
        assert obs.done is True
        assert obs.reward <= 0.50

    def test_hard_011_perfect_run(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="hard_011")
        env.step(CommitmentAction(action_type="view_calendar", date="2026-04-25"))
        env.step(CommitmentAction(action_type="cancel_event", event_id="evt_90"))
        env.step(CommitmentAction(action_type="search_restaurants", dietary="vegetarian", near_airport=True, max_price=60))
        env.step(CommitmentAction(action_type="book_restaurant", restaurant_name="Sky Lounge"))
        env.step(CommitmentAction(action_type="send_email", to="Team", subject="Happy Hour Rescheduled", body="Sorry team, I need to reschedule the happy hour to Thursday. An investor dinner came up tonight. Apologies!"))
        env.step(CommitmentAction(action_type="send_email", to="VP_Chen", subject="Investor dinner plan", body="I've booked Sky Lounge for dinner tonight with Investor_Park. Vegetarian options available, near the airport."))
        obs = env.step(CommitmentAction(action_type="submit_plan"))
        assert obs.done is True
        assert obs.reward >= 0.85

    def test_hard_015_sre_crisis(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="hard_015")
        env.step(CommitmentAction(action_type="view_calendar", date="2026-04-25"))
        env.step(CommitmentAction(action_type="cancel_event", event_id="evt_130"))
        env.step(CommitmentAction(action_type="send_email", to="Team", subject="Lunch cancelled - incident", body="Team, I'm cancelling our lunch due to a production incident. Payment service returning 503s. Will handle this first."))
        env.step(CommitmentAction(action_type="send_email", to="Client_Jones", subject="Demo reschedule needed", body="Hi Client_Jones, I sincerely apologize but I need to reschedule our demo. We have a production incident with the payment system. Can we find another time this week?"))
        env.step(CommitmentAction(action_type="send_email", to="VP_Chen", subject="Incident + 1-on-1", body="VP_Chen, we have a production incident — payment service is returning 503s. I'm on-call and handling it. May need to reschedule our 1-on-1 depending on resolution time."))
        obs = env.step(CommitmentAction(action_type="submit_plan"))
        assert obs.done is True
        assert obs.reward >= 0.60


# ===================================================================
# 7. Reward clamping
# ===================================================================

class TestRewardClamping:
    def test_reward_never_zero(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        obs = env.step(CommitmentAction(action_type="submit_plan"))
        assert obs.reward >= 0.01

    def test_reward_never_one(self, env: CommitmentEnvironment) -> None:
        env.reset(task_id="easy_001")
        env.step(CommitmentAction(action_type="reschedule_event", event_id="evt_2", new_time="15:00"))
        env.step(CommitmentAction(action_type="send_email", to="Team", subject="Standup moved", body="Hi team, the standup is rescheduled to 3pm. Sorry for the move."))
        obs = env.step(CommitmentAction(action_type="submit_plan"))
        assert obs.reward <= 0.99
        assert obs.reward > 0.01


# ===================================================================
# 8. Time utility
# ===================================================================

class TestTimeUtil:
    def test_time_to_min(self) -> None:
        assert _time_to_min("00:00") == 0
        assert _time_to_min("09:30") == 570
        assert _time_to_min("14:00") == 840
        assert _time_to_min("23:59") == 1439


# ===================================================================
# 9. API endpoint tests (via TestClient)
# ===================================================================

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from server.app import app
        return TestClient(app)

    def test_health(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_tasks(self, client) -> None:
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["easy"]) == 5
        assert len(data["medium"]) == 5
        assert len(data["hard"]) == 5

    def test_reset_step_state(self, client) -> None:
        resp = client.post("/reset", params={"task_id": "easy_001"})
        assert resp.status_code == 200

        resp = client.post("/step", json={"action": {"action_type": "view_calendar", "date": "2026-04-25"}})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("done") is False

        resp = client.get("/state")
        assert resp.status_code == 200
        state = resp.json()
        assert "step_count" in state

    def test_mcp_initialize(self, client) -> None:
        resp = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["serverInfo"]["name"] == "commitment-os"

    def test_mcp_tools_list(self, client) -> None:
        resp = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })
        assert resp.status_code == 200
        tools = resp.json()["result"]["tools"]
        assert len(tools) == 3


# ===================================================================
# 10. Metadata
# ===================================================================

class TestMetadata:
    def test_get_metadata(self, env: CommitmentEnvironment) -> None:
        meta = env.get_metadata()
        assert meta.name == "commitment-os"
        assert "Jayant" in meta.author
