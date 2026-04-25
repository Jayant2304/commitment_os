"""Evaluate base vs RL-trained LLM checkpoints on CommitmentOS.

This script runs the SAME protocol for two model names:
- baseline model (pre-RL)
- trained model (post-RL checkpoint)

It writes judge-friendly artifacts under artifacts/evals_llm/.
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from pathlib import Path
from statistics import mean, median
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI

ARTIFACT_DIR = Path("artifacts/evals_llm")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://jayant2304-commitment-os.hf.space")

BASELINE_MODEL = os.getenv("BASELINE_MODEL_NAME", "").strip()
TRAINED_MODEL = os.getenv("TRAINED_MODEL_NAME", "").strip()

EVAL_SEED = int(os.getenv("EVAL_SEED", "42"))
MAX_STEPS = int(os.getenv("EVAL_MAX_STEPS", "12"))
TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("EVAL_TOP_P", "1.0"))
MAX_NEW_TOKENS = int(os.getenv("EVAL_MAX_NEW_TOKENS", "256"))
SUCCESS_THRESHOLD = float(os.getenv("EVAL_SUCCESS_THRESHOLD", "0.6"))

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


def _require_env() -> None:
    if not API_KEY:
        raise RuntimeError("Set HF_TOKEN or OPENAI_API_KEY")
    if not BASELINE_MODEL:
        raise RuntimeError("Set BASELINE_MODEL_NAME")
    if not TRAINED_MODEL:
        raise RuntimeError("Set TRAINED_MODEL_NAME")


def _get_task_ids() -> list[str]:
    resp = requests.get(f"{ENV_BASE_URL}/tasks", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    task_ids: list[str] = []
    for difficulty in ("easy", "medium", "hard"):
        task_ids.extend(data.get(difficulty, []))
    return task_ids


def _parse_action(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[0]
    try:
        action = json.loads(text)
        if isinstance(action, dict) and action.get("action_type"):
            return action
    except json.JSONDecodeError:
        pass
    return {"action_type": "submit_plan"}


def _llm_action(client: OpenAI, model_name: str, messages: list[dict[str, str]]) -> tuple[dict[str, Any], str]:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        stream=False,
    )
    raw = (response.choices[0].message.content or "").strip()
    return _parse_action(raw), raw


def _env_reset(task_id: str, episode_id: str) -> dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        params={"task_id": task_id, "seed": EVAL_SEED, "episode_id": episode_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("observation", data)


def _env_step(action: dict[str, Any], episode_id: str) -> dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        params={"episode_id": episode_id},
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    obs = data.get("observation", data)
    obs["done"] = data.get("done", obs.get("done", False))
    obs["reward"] = float(data.get("reward", obs.get("reward", 0.0)) or 0.0)
    return obs


def _env_state(episode_id: str) -> dict[str, Any]:
    resp = requests.get(f"{ENV_BASE_URL}/state", params={"episode_id": episode_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_task(client: OpenAI, model_name: str, task_id: str) -> dict[str, Any]:
    episode_id = f"eval-{model_name.replace('/', '-')}-{task_id}-{uuid.uuid4().hex[:8]}"
    obs = _env_reset(task_id, episode_id)

    briefing = obs.get("briefing", "")
    calendar = json.dumps(obs.get("calendar_snapshot", []), indent=2)
    inbox = json.dumps(obs.get("inbox", []), indent=2)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"SCENARIO: {briefing}\n\nCALENDAR:\n{calendar}\n\nINBOX:\n{inbox}\n\nWhat is your first action?"},
    ]

    trace: list[dict[str, Any]] = []
    step_num = 0
    done = False
    final_obs: dict[str, Any] = obs

    for step_num in range(1, MAX_STEPS + 1):
        action, raw = _llm_action(client, model_name, messages)
        step_obs = _env_step(action, episode_id)
        final_obs = step_obs
        done = bool(step_obs.get("done", False))
        trace.append(
            {
                "step": step_num,
                "action": action,
                "raw_model_output": raw,
                "reward": float(step_obs.get("reward", 0.0)),
                "done": done,
                "tool_result": step_obs.get("tool_result", ""),
            }
        )
        if done:
            break
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"TOOL RESULT: {step_obs.get('tool_result', '')}\n\nWhat is your next action?"})

    if not done:
        final_obs = _env_step({"action_type": "submit_plan"}, episode_id)
        step_num += 1
        trace.append(
            {
                "step": step_num,
                "action": {"action_type": "submit_plan"},
                "raw_model_output": '{"action_type":"submit_plan"}',
                "reward": float(final_obs.get("reward", 0.0)),
                "done": True,
                "tool_result": final_obs.get("tool_result", ""),
            }
        )

    state = _env_state(episode_id)
    final_reward = float(final_obs.get("reward", 0.0))
    return {
        "task_id": task_id,
        "difficulty": final_obs.get("difficulty", ""),
        "model_name": model_name,
        "final_reward": round(final_reward, 4),
        "success": final_reward >= SUCCESS_THRESHOLD,
        "steps_used": int(state.get("step_count", step_num)),
        "violation_count": int(state.get("violation_count", 0)),
        "reward_breakdown": final_obs.get("reward_breakdown", {}),
        "feedback": final_obs.get("feedback", ""),
        "trace": trace,
    }


def run_model(client: OpenAI, model_name: str, task_ids: list[str]) -> list[dict[str, Any]]:
    return [run_task(client, model_name=model_name, task_id=task_id) for task_id in task_ids]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_artifacts(baseline: list[dict[str, Any]], trained: list[dict[str, Any]]) -> None:
    by_task = {row["task_id"]: row for row in trained}
    comparison_rows: list[dict[str, Any]] = []
    for base in baseline:
        tr = by_task[base["task_id"]]
        comparison_rows.append(
            {
                "task_id": base["task_id"],
                "difficulty": base["difficulty"],
                "baseline_reward": base["final_reward"],
                "trained_reward": tr["final_reward"],
                "reward_delta": round(tr["final_reward"] - base["final_reward"], 4),
                "baseline_steps": base["steps_used"],
                "trained_steps": tr["steps_used"],
                "step_delta": tr["steps_used"] - base["steps_used"],
                "baseline_violations": base["violation_count"],
                "trained_violations": tr["violation_count"],
                "violation_delta": tr["violation_count"] - base["violation_count"],
                "baseline_success": int(base["success"]),
                "trained_success": int(tr["success"]),
            }
        )

    _write_json(ARTIFACT_DIR / "baseline_llm_eval.json", baseline)
    _write_json(ARTIFACT_DIR / "trained_llm_eval.json", trained)
    _write_json(
        ARTIFACT_DIR / "llm_eval_protocol.json",
        {
            "task_set": "easy_001..hard_015",
            "seed": EVAL_SEED,
            "max_steps": MAX_STEPS,
            "decode_config": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            "env_base_url": ENV_BASE_URL,
            "baseline_model_name": BASELINE_MODEL,
            "trained_model_name": TRAINED_MODEL,
            "success_threshold": SUCCESS_THRESHOLD,
        },
    )

    with (ARTIFACT_DIR / "llm_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    baseline_rewards = [r["baseline_reward"] for r in comparison_rows]
    trained_rewards = [r["trained_reward"] for r in comparison_rows]
    reward_deltas = [r["reward_delta"] for r in comparison_rows]
    baseline_steps = [r["baseline_steps"] for r in comparison_rows]
    trained_steps = [r["trained_steps"] for r in comparison_rows]
    baseline_violations = [r["baseline_violations"] for r in comparison_rows]
    trained_violations = [r["trained_violations"] for r in comparison_rows]
    baseline_success = [r["baseline_success"] for r in comparison_rows]
    trained_success = [r["trained_success"] for r in comparison_rows]

    summary = {
        "task_count": len(comparison_rows),
        "baseline_mean_reward": round(mean(baseline_rewards), 4),
        "trained_mean_reward": round(mean(trained_rewards), 4),
        "mean_reward_delta": round(mean(trained_rewards) - mean(baseline_rewards), 4),
        "median_reward_delta": round(median(reward_deltas), 4),
        "baseline_success_rate": round(mean(baseline_success), 4),
        "trained_success_rate": round(mean(trained_success), 4),
        "success_rate_delta": round(mean(trained_success) - mean(baseline_success), 4),
        "baseline_mean_steps": round(mean(baseline_steps), 4),
        "trained_mean_steps": round(mean(trained_steps), 4),
        "step_delta": round(mean(trained_steps) - mean(baseline_steps), 4),
        "baseline_mean_violations": round(mean(baseline_violations), 4),
        "trained_mean_violations": round(mean(trained_violations), 4),
        "violation_delta": round(mean(trained_violations) - mean(baseline_violations), 4),
        "tasks_with_positive_reward_delta": sum(1 for x in reward_deltas if x > 0),
        "tasks_with_no_reward_delta": sum(1 for x in reward_deltas if x == 0),
        "per_difficulty": {},
    }

    for difficulty in ("easy", "medium", "hard"):
        subset = [r for r in comparison_rows if r["difficulty"] == difficulty]
        if not subset:
            continue
        summary["per_difficulty"][difficulty] = {
            "count": len(subset),
            "baseline_mean_reward": round(mean([r["baseline_reward"] for r in subset]), 4),
            "trained_mean_reward": round(mean([r["trained_reward"] for r in subset]), 4),
            "reward_delta": round(
                mean([r["trained_reward"] for r in subset]) - mean([r["baseline_reward"] for r in subset]),
                4,
            ),
            "baseline_mean_steps": round(mean([r["baseline_steps"] for r in subset]), 4),
            "trained_mean_steps": round(mean([r["trained_steps"] for r in subset]), 4),
            "step_delta": round(
                mean([r["trained_steps"] for r in subset]) - mean([r["baseline_steps"] for r in subset]),
                4,
            ),
        }

    _write_json(ARTIFACT_DIR / "llm_summary.json", summary)

    target_task = "hard_015"
    base_case = next((r for r in baseline if r["task_id"] == target_task), None)
    tr_case = next((r for r in trained if r["task_id"] == target_task), None)
    if base_case and tr_case:
        case_study = f"""# LLM Case Study: {target_task}

## Baseline model ({BASELINE_MODEL})
- Reward: {base_case['final_reward']:.4f}
- Steps: {base_case['steps_used']}
- Violations: {base_case['violation_count']}
- Feedback: {base_case['feedback']}

## Trained model ({TRAINED_MODEL})
- Reward: {tr_case['final_reward']:.4f}
- Steps: {tr_case['steps_used']}
- Violations: {tr_case['violation_count']}
- Feedback: {tr_case['feedback']}
"""
        (ARTIFACT_DIR / "llm_case_study_hard_015.md").write_text(case_study)


def main() -> None:
    _require_env()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_ids = _get_task_ids()
    baseline_results = run_model(client, BASELINE_MODEL, task_ids)
    trained_results = run_model(client, TRAINED_MODEL, task_ids)
    write_artifacts(baseline_results, trained_results)
    print("Wrote LLM checkpoint artifacts to", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
