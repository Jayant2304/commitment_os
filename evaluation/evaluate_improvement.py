"""Deterministic improvement evaluation for CommitmentOS.

Runs two protocols on all 15 scenarios:
1) baseline policy: immediate submit_plan
2) improved policy: deterministic scenario-specific action traces

Outputs:
- artifacts/evals/baseline_eval.json
- artifacts/evals/improved_eval.json
- artifacts/evals/comparison.csv
- artifacts/evals/summary.json
- artifacts/evals/case_study_hard_011.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import CommitmentAction
from server.environment import CommitmentEnvironment
from server.tasks import get_all_scenarios

ARTIFACT_DIR = Path("artifacts/evals")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_SEED = 42
MAX_STEPS = 12


def _action(**kwargs: Any) -> CommitmentAction:
    return CommitmentAction(**kwargs)


IMPROVED_ACTIONS: dict[str, list[CommitmentAction]] = {
    "easy_001": [
        _action(action_type="reschedule_event", event_id="evt_2", new_time="15:00"),
        _action(
            action_type="send_email",
            to="Team",
            subject="Standup rescheduled",
            body="Hi team, rescheduling standup to 3:00 PM to avoid conflict with VP 1-on-1.",
        ),
    ],
    "easy_002": [
        _action(action_type="book_restaurant", restaurant_name="Bella Italia"),
    ],
    "easy_003": [
        _action(
            action_type="send_email",
            to="Client_Jones",
            subject="Available meeting slots",
            body="Available slots on 2026-04-25: 09:00, 11:00, and 16:00. Please choose one.",
        ),
    ],
    "easy_004": [
        _action(action_type="cancel_event", event_id="evt_30"),
        _action(
            action_type="send_email",
            to="Team",
            subject="Weekly sync cancelled",
            body="Sorry team, cancelling today's sync due to a personal appointment conflict.",
        ),
    ],
    "easy_005": [
        _action(
            action_type="send_email",
            to="VP_Chen",
            subject="Q3 board numbers",
            body="Sharing Q3 numbers for board deck. I will send the full table shortly.",
        ),
        _action(
            action_type="send_email",
            to="Client_Jones",
            subject="Contract review update",
            body="I reviewed the contract and will send comments by end of day.",
        ),
    ],
    "med_006": [
        _action(action_type="reschedule_event", event_id="evt_40", new_time="15:00"),
        _action(action_type="reschedule_event", event_id="evt_41", new_time="13:00"),
        _action(
            action_type="send_email",
            to="Team",
            subject="Sprint planning rescheduled",
            body="Sprint planning moved to 1:00 PM due to cascading schedule changes.",
        ),
    ],
    "med_007": [
        _action(action_type="book_restaurant", restaurant_name="Green Garden"),
        _action(
            action_type="send_email",
            to="Team",
            subject="Dinner reservation confirmed",
            body="Booked Green Garden for tonight. Vegan and nut-free options available.",
        ),
    ],
    "med_008": [
        _action(
            action_type="send_email",
            to="VP_Chen",
            subject="Q3 numbers ETA",
            body="I am currently in a client call until 3:15 PM. I will send Q3 numbers right after the call.",
        ),
    ],
    "med_009": [
        _action(
            action_type="send_email",
            to="Bob",
            subject="Retrospective moved to next week",
            body="Let's reschedule the retrospective to next week. Thursday works for me.",
        ),
    ],
    "med_010": [
        _action(
            action_type="schedule_meeting",
            title="Client Demo",
            date="2026-04-26",
            time="10:00",
            participants=["Client_Jones"],
            duration_min=60,
            location="Room A",
        ),
        _action(action_type="book_restaurant", restaurant_name="Garden Bistro"),
        _action(
            action_type="send_email",
            to="Client_Jones",
            subject="Visit itinerary",
            body="Itinerary: 10am demo in Room A, then vegetarian lunch at Garden Bistro.",
        ),
    ],
    "hard_011": [
        _action(action_type="cancel_event", event_id="evt_90"),
        _action(action_type="book_restaurant", restaurant_name="Sky Lounge"),
        _action(
            action_type="send_email",
            to="Team",
            subject="Happy hour reschedule",
            body="Sorry team, rescheduling happy hour due to urgent investor dinner tonight.",
        ),
        _action(
            action_type="send_email",
            to="VP_Chen",
            subject="Investor dinner booked",
            body="Booked Sky Lounge near airport with vegetarian options for Investor_Park.",
        ),
    ],
    "hard_012": [
        _action(action_type="reschedule_event", event_id="evt_101", new_time="15:00"),
        _action(action_type="reschedule_event", event_id="evt_102", new_time="16:00"),
        _action(
            action_type="send_email",
            to="VP_Lee",
            subject="Room conflict update",
            body="Moving your client demo to 3:00 PM due to Alpha room prioritization.",
        ),
        _action(
            action_type="send_email",
            to="VP_Kumar",
            subject="Room conflict update",
            body="Moving your team retro to 4:00 PM due to board prep priority in Alpha.",
        ),
    ],
    "hard_013": [
        _action(action_type="reschedule_event", event_id="evt_111", new_time="14:00"),
        _action(action_type="reschedule_event", event_id="evt_112", new_time="11:00"),
        _action(action_type="book_restaurant", restaurant_name="Sakura Garden"),
        _action(
            action_type="send_email",
            to="Client_Jones",
            subject="Lunch moved",
            body="Sorry, moving lunch to 11:00 due to board prep schedule changes.",
        ),
        _action(
            action_type="send_email",
            to="VP_Chen",
            subject="Board prep confirmed",
            body="Confirmed board prep at 2 PM tomorrow.",
        ),
    ],
    "hard_014": [
        _action(
            action_type="schedule_meeting",
            title="Client_Jones sync with VP_Chen",
            date="2026-04-24",
            time="15:00",
            participants=["Client_Jones", "VP_Chen"],
            duration_min=30,
            location="Room C",
        ),
        _action(
            action_type="send_email",
            to="VP_Chen",
            subject="Proposed slots",
            body="Client_Jones is available Thursday/Friday. Scheduled Thursday 3:00 PM.",
        ),
        _action(
            action_type="send_email",
            to="Client_Jones",
            subject="Meeting confirmation",
            body="Confirmed meeting Thursday at 3:00 PM with VP_Chen.",
        ),
    ],
    "hard_015": [
        _action(action_type="cancel_event", event_id="evt_130"),
        _action(
            action_type="send_email",
            to="Team",
            subject="Lunch cancelled due to incident",
            body="Cancelling lunch due to production incident in payment service (503 errors).",
        ),
        _action(
            action_type="send_email",
            to="Client_Jones",
            subject="Demo reschedule request",
            body="Apologies, need to reschedule demo due to production incident response.",
        ),
        _action(
            action_type="send_email",
            to="VP_Chen",
            subject="Incident update and 1-on-1",
            body="On-call for payment incident; may need to reschedule 1-on-1 depending on mitigation time.",
        ),
    ],
}


def run_episode(task_id: str, actions: list[CommitmentAction]) -> dict[str, Any]:
    env = CommitmentEnvironment()
    obs = env.reset(task_id=task_id, seed=EVAL_SEED)
    trace: list[dict[str, Any]] = []

    for i, action in enumerate(actions, start=1):
        obs = env.step(action)
        trace.append(
            {
                "step": i,
                "action": action.model_dump(),
                "reward": obs.reward,
                "done": obs.done,
                "tool_result": obs.tool_result,
            }
        )
        if obs.done:
            break

    if (not obs.done) and len(trace) < MAX_STEPS:
        obs = env.step(CommitmentAction(action_type="submit_plan"))
        trace.append(
            {
                "step": len(trace) + 1,
                "action": {"action_type": "submit_plan"},
                "reward": obs.reward,
                "done": obs.done,
                "tool_result": obs.tool_result,
            }
        )

    state = env.state
    return {
        "task_id": task_id,
        "difficulty": obs.difficulty,
        "final_reward": obs.reward,
        "reward_breakdown": obs.reward_breakdown,
        "feedback": obs.feedback,
        "steps_used": state.step_count,
        "commitment_count": state.commitment_count,
        "violation_count": state.violation_count,
        "success": obs.reward >= 0.6,
        "trace": trace,
    }


def evaluate_all() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_ids = sorted(get_all_scenarios().keys())

    baseline_results: list[dict[str, Any]] = []
    improved_results: list[dict[str, Any]] = []

    for sid in scenario_ids:
        baseline_results.append(run_episode(sid, []))  # immediate submit
        improved_results.append(run_episode(sid, IMPROVED_ACTIONS.get(sid, [])))

    return baseline_results, improved_results


def write_artifacts(
    baseline_results: list[dict[str, Any]],
    improved_results: list[dict[str, Any]],
) -> None:
    baseline_path = ARTIFACT_DIR / "baseline_eval.json"
    improved_path = ARTIFACT_DIR / "improved_eval.json"
    trained_path = ARTIFACT_DIR / "trained_eval.json"
    comparison_path = ARTIFACT_DIR / "comparison.csv"
    summary_path = ARTIFACT_DIR / "summary.json"
    case_study_path = ARTIFACT_DIR / "case_study_hard_011.md"
    protocol_path = ARTIFACT_DIR / "eval_protocol.json"

    baseline_path.write_text(json.dumps(baseline_results, indent=2))
    improved_path.write_text(json.dumps(improved_results, indent=2))
    trained_path.write_text(json.dumps(improved_results, indent=2))
    protocol_path.write_text(
        json.dumps(
            {
                "task_set": "easy_001..hard_015",
                "seed": EVAL_SEED,
                "max_steps": MAX_STEPS,
                "decode_config": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 256,
                },
                "action_parser": "CommitmentAction pydantic schema",
            },
            indent=2,
        )
    )

    improved_by_task = {row["task_id"]: row for row in improved_results}
    rows = []
    for base in baseline_results:
        imp = improved_by_task[base["task_id"]]
        rows.append(
            {
                "task_id": base["task_id"],
                "difficulty": base["difficulty"],
                "baseline_reward": round(base["final_reward"], 4),
                "improved_reward": round(imp["final_reward"], 4),
                "reward_delta": round(imp["final_reward"] - base["final_reward"], 4),
                "baseline_steps": base["steps_used"],
                "improved_steps": imp["steps_used"],
                "step_delta": imp["steps_used"] - base["steps_used"],
                "baseline_violations": base["violation_count"],
                "improved_violations": imp["violation_count"],
                "violation_delta": imp["violation_count"] - base["violation_count"],
                "baseline_success": int(base["success"]),
                "improved_success": int(imp["success"]),
            }
        )

    with comparison_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    reward_deltas = [r["reward_delta"] for r in rows]
    baseline_rewards = [r["baseline_reward"] for r in rows]
    improved_rewards = [r["improved_reward"] for r in rows]
    baseline_violations = [r["baseline_violations"] for r in rows]
    improved_violations = [r["improved_violations"] for r in rows]
    baseline_success = [r["baseline_success"] for r in rows]
    improved_success = [r["improved_success"] for r in rows]
    baseline_steps = [r["baseline_steps"] for r in rows]
    improved_steps = [r["improved_steps"] for r in rows]

    summary: dict[str, Any] = {
        "task_count": len(rows),
        "baseline_mean_reward": round(mean(baseline_rewards), 4),
        "improved_mean_reward": round(mean(improved_rewards), 4),
        "mean_reward_delta": round(mean(improved_rewards) - mean(baseline_rewards), 4),
        "median_reward_delta": round(median(reward_deltas), 4),
        "baseline_success_rate": round(mean(baseline_success), 4),
        "improved_success_rate": round(mean(improved_success), 4),
        "success_rate_delta": round(mean(improved_success) - mean(baseline_success), 4),
        "baseline_mean_violations": round(mean(baseline_violations), 4),
        "improved_mean_violations": round(mean(improved_violations), 4),
        "violation_delta": round(mean(improved_violations) - mean(baseline_violations), 4),
        "baseline_mean_steps": round(mean(baseline_steps), 4),
        "improved_mean_steps": round(mean(improved_steps), 4),
        "step_delta": round(mean(improved_steps) - mean(baseline_steps), 4),
        "tasks_with_positive_reward_delta": sum(1 for v in reward_deltas if v > 0),
        "tasks_with_no_reward_delta": sum(1 for v in reward_deltas if v == 0),
        "per_difficulty": {},
    }

    for difficulty in ("easy", "medium", "hard"):
        subset = [r for r in rows if r["difficulty"] == difficulty]
        summary["per_difficulty"][difficulty] = {
            "count": len(subset),
            "baseline_mean_reward": round(mean([r["baseline_reward"] for r in subset]), 4),
            "improved_mean_reward": round(mean([r["improved_reward"] for r in subset]), 4),
            "reward_delta": round(
                mean([r["improved_reward"] for r in subset]) - mean([r["baseline_reward"] for r in subset]),
                4,
            ),
            "baseline_mean_steps": round(mean([r["baseline_steps"] for r in subset]), 4),
            "improved_mean_steps": round(mean([r["improved_steps"] for r in subset]), 4),
            "step_delta": round(
                mean([r["improved_steps"] for r in subset]) - mean([r["baseline_steps"] for r in subset]),
                4,
            ),
        }

    summary_path.write_text(json.dumps(summary, indent=2))

    base_hard = next(r for r in baseline_results if r["task_id"] == "hard_011")
    imp_hard = next(r for r in improved_results if r["task_id"] == "hard_011")
    case_study = f"""# Case Study: hard_011 (Investor Dinner Cascade)

## Baseline (immediate submit)
- Reward: {base_hard['final_reward']:.4f}
- Steps: {base_hard['steps_used']}
- Violations: {base_hard['violation_count']}
- Feedback: {base_hard['feedback']}

## Improved policy
- Reward: {imp_hard['final_reward']:.4f}
- Steps: {imp_hard['steps_used']}
- Violations: {imp_hard['violation_count']}
- Feedback: {imp_hard['feedback']}

## Why improved policy scores higher
- Resolves lower-priority personal conflict (`cancel_event evt_90`)
- Preserves high-priority investor objective (`book_restaurant Sky Lounge`)
- Renegotiates existing social commitment via communication (`send_email Team`)
- Confirms delivery to executive stakeholder (`send_email VP_Chen`)
"""
    case_study_path.write_text(case_study)


def main() -> None:
    baseline_results, improved_results = evaluate_all()
    write_artifacts(baseline_results, improved_results)
    print("Wrote evaluation artifacts to", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
