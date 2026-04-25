"""Environment factory for TRL GRPOTrainer integration.

Wraps CommitmentOS as a callable that accepts model completions and
returns rewards, making it compatible with TRL's ``environment_factory``
pattern for multi-turn RL training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.domain import ScenarioDef
from server.environment import CommitmentEnvironment
from server.tasks import get_all_scenarios
from models import CommitmentAction


TOOL_DESCRIPTIONS = """Available tools (respond with JSON):
- {"action_type": "view_calendar", "date": "2026-04-25"}
- {"action_type": "check_availability", "person": "Name"}
- {"action_type": "search_restaurants", "cuisine": "...", "max_price": 50, "dietary": "..."}
- {"action_type": "schedule_meeting", "title": "...", "date": "...", "time": "HH:MM", "participants": [...]}
- {"action_type": "reschedule_event", "event_id": "evt_X", "new_time": "HH:MM"}
- {"action_type": "cancel_event", "event_id": "evt_X"}
- {"action_type": "send_email", "to": "Name", "subject": "...", "body": "..."}
- {"action_type": "book_restaurant", "restaurant_name": "..."}
- {"action_type": "submit_plan"}"""


def build_system_prompt() -> str:
    return (
        "You are an expert executive assistant AI managing calendars, emails, and "
        "dining reservations. For each turn, respond with EXACTLY ONE JSON tool call.\n\n"
        f"{TOOL_DESCRIPTIONS}\n\n"
        "Rules:\n"
        "1. Respond with ONLY JSON, no markdown or explanation\n"
        "2. Handle higher-priority items first\n"
        "3. When cancelling/rescheduling commitments, ALWAYS email affected parties\n"
        "4. Call submit_plan when all issues are resolved\n"
        "5. Never silently drop a commitment"
    )


def build_initial_prompt(scenario: ScenarioDef) -> str:
    """Build the user message for the first turn of an episode."""
    from server.world import WorldState

    world = WorldState(scenario)
    calendar = json.dumps(world.get_calendar_snapshot(), indent=2)
    inbox = json.dumps(world.get_inbox_snapshot(), indent=2)

    return (
        f"SCENARIO: {scenario.briefing}\n\n"
        f"CALENDAR:\n{calendar}\n\n"
        f"INBOX:\n{inbox}\n\n"
        "What is your first action? Respond with a JSON tool call."
    )


def parse_action_from_text(text: str) -> Dict[str, Any]:
    """Extract a JSON action from model output, with fallback to submit."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action_type" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "action_type" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                continue

    return {"action_type": "submit_plan"}


class CommitmentOSEnvFactory:
    """Wraps CommitmentOS for use with TRL's GRPOTrainer.

    Usage with TRL::

        from training.env_factory import CommitmentOSEnvFactory

        factory = CommitmentOSEnvFactory(max_turns=8)

        trainer = GRPOTrainer(
            ...
            environment_factory=factory,
        )
    """

    def __init__(
        self,
        max_turns: int = 8,
        scenario_ids: Optional[List[str]] = None,
    ) -> None:
        self.max_turns = max_turns
        self.scenario_ids = scenario_ids or list(get_all_scenarios().keys())
        self.system_prompt = build_system_prompt()

    def __call__(self, completions: List[str], **kwargs: Any) -> List[float]:
        """Evaluate a batch of model completions.

        Each completion is treated as a full multi-turn transcript where
        each line is one JSON action. Returns a list of final rewards.
        """
        rewards: List[float] = []
        for completion in completions:
            reward = self._evaluate_single(completion)
            rewards.append(reward)
        return rewards

    def _evaluate_single(self, completion: str) -> float:
        import random

        env = CommitmentEnvironment()
        scenario_id = random.choice(self.scenario_ids)
        env.reset(task_id=scenario_id)

        actions = completion.strip().split("\n")
        last_reward = 0.01

        for i, action_text in enumerate(actions[: self.max_turns]):
            action_dict = parse_action_from_text(action_text)
            try:
                action = CommitmentAction(**action_dict)
                obs = env.step(action)
                last_reward = obs.reward
                if obs.done:
                    break
            except Exception:
                # Invalid action payloads should be penalized, not silently ignored.
                last_reward = 0.01
                break

        if not env._done:
            obs = env.step(CommitmentAction(action_type="submit_plan"))
            last_reward = obs.reward

        return float(last_reward)

    def get_prompt(self, scenario_id: Optional[str] = None) -> List[Dict[str, str]]:
        """Build chat messages for a scenario."""
        import random
        from server.tasks import get_scenario

        sid = scenario_id or random.choice(self.scenario_ids)
        scenario = get_scenario(sid)
        if scenario is None:
            raise ValueError(f"Unknown scenario: {sid}")

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_initial_prompt(scenario)},
        ]
