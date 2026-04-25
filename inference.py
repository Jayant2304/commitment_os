"""Baseline inference script for CommitmentOS.

Uses an OpenAI-compatible LLM to play through all 15 scenarios.
Multi-turn: the agent gets the briefing, makes tool calls, then submits.

Required environment variables:
  API_BASE_URL  — OpenAI-compatible endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — API key (also checked as OPENAI_API_KEY)
  ENV_BASE_URL  — CommitmentOS server URL (default: HF Space)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import requests
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://jayant2304-commitment-os.hf.space")

MAX_STEPS = 12

SYSTEM_PROMPT = """You are an expert executive assistant AI. You manage calendars, emails, and dining reservations.

You will be given a scenario briefing describing a situation with calendar conflicts, emails, or planning tasks.

For each turn, you must respond with EXACTLY ONE JSON object choosing a tool to call:

Available tools:
- {"action_type": "view_calendar", "date": "2026-04-25"}
- {"action_type": "check_availability", "person": "Client_Jones"}
- {"action_type": "search_restaurants", "cuisine": "Italian", "max_price": 50, "dietary": "vegetarian", "max_distance_miles": 3.0, "near_airport": false}
- {"action_type": "schedule_meeting", "title": "Demo", "date": "2026-04-25", "time": "14:00", "duration_min": 60, "participants": ["Client_Jones"], "location": "Room A"}
- {"action_type": "reschedule_event", "event_id": "evt_1", "new_time": "15:00"}
- {"action_type": "cancel_event", "event_id": "evt_1"}
- {"action_type": "send_email", "to": "VP_Chen", "subject": "Meeting update", "body": "Hi, I need to reschedule..."}
- {"action_type": "book_restaurant", "restaurant_name": "Sky Lounge"}
- {"action_type": "submit_plan"}

IMPORTANT RULES:
1. Respond with ONLY a JSON object, no markdown, no explanation
2. Handle higher-priority items before lower-priority ones
3. When cancelling or rescheduling commitments, ALWAYS send an email to affected parties BEFORE submitting
4. Call submit_plan when you have resolved all issues
5. Never silently drop a commitment — always notify the affected person"""


# ---------------------------------------------------------------------------
# Logging helpers — exact format required by hackathon evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={'true' if done else 'false'} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={'true' if success else 'false'} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Environment interaction
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("observation", data)


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    obs = data.get("observation", data)
    obs["done"] = data.get("done", obs.get("done", False))
    obs["reward"] = data.get("reward", obs.get("reward", 0.0))
    return obs


def get_task_ids() -> List[str]:
    resp = requests.get(f"{ENV_BASE_URL}/tasks", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    ids: List[str] = []
    for difficulty in ["easy", "medium", "hard"]:
        ids.extend(data.get(difficulty, []))
    return ids


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
        stream=False,
    )
    return response.choices[0].message.content.strip()


def parse_action(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"action_type": "submit_plan"}


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> Dict[str, Any]:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    try:
        obs = env_reset(task_id)
        log_start(task=task_id, env="commitment-os", model=MODEL_NAME)

        briefing = obs.get("briefing", "")
        calendar = json.dumps(obs.get("calendar_snapshot", []), indent=2)
        inbox = json.dumps(obs.get("inbox", []), indent=2)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SCENARIO: {briefing}\n\nCALENDAR:\n{calendar}\n\nINBOX:\n{inbox}\n\nWhat is your first action?"},
        ]

        for step_num in range(1, MAX_STEPS + 1):
            llm_output = call_llm(client, messages)
            action = parse_action(llm_output)

            step_data = env_step(action)
            reward = float(step_data.get("reward", 0.0) or 0.0)
            done = step_data.get("done", False)
            steps_taken = step_num
            rewards.append(reward)

            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=step_num, action=action_str, reward=reward, done=done)

            if done:
                score = max(0.01, min(0.99, reward))
                success = score > 0.01
                break

            tool_result = step_data.get("tool_result", "")
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"TOOL RESULT: {tool_result}\n\nWhat is your next action?"})

        if not done:
            step_data = env_step({"action_type": "submit_plan"})
            reward = float(step_data.get("reward", 0.0) or 0.0)
            steps_taken += 1
            rewards.append(reward)
            score = max(0.01, min(0.99, reward))
            success = score > 0.01
            log_step(step=steps_taken, action='{"action_type":"submit_plan"}', reward=reward, done=True)

    except Exception as exc:
        steps_taken = max(steps_taken, 1)
        if not rewards:
            rewards.append(0.01)
        log_step(step=steps_taken, action="error", reward=0.01, done=True, error=str(exc))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "reward": score, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or OPENAI_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_ids = get_task_ids()

    results: List[Dict[str, Any]] = []
    for tid in task_ids:
        result = run_task(client, tid)
        results.append(result)

    total = len(results)
    successes = sum(1 for r in results if r["success"])
    mean_reward = sum(r["reward"] for r in results) / total if total > 0 else 0.0
    print(f"\n# Summary: {successes}/{total} tasks succeeded, mean_reward={mean_reward:.3f}", flush=True)


if __name__ == "__main__":
    main()
