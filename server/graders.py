"""Deterministic grading — 5-component reward for CommitmentOS.

Components:
  constraint_satisfaction (0.35) — binary per scenario constraint
  conflict_resolution     (0.20) — final calendar free of overlaps
  commitment_coherence    (0.20) — ledger violations penalised
  communication_quality   (0.15) — keyword matching on sent emails
  step_efficiency         (0.10) — fewer steps = higher score
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from server.domain import ScenarioDef
from server.world import WorldState, _time_to_min

WEIGHTS: Dict[str, float] = {
    "constraint_satisfaction": 0.35,
    "conflict_resolution": 0.20,
    "commitment_coherence": 0.20,
    "communication_quality": 0.15,
    "step_efficiency": 0.10,
}


def _keyword_score(text: str, keywords: List[str], min_matches: int = 2) -> Tuple[float, List[str]]:
    """0 hits -> 0.0, < min_matches -> 0.5 (partial), >= min_matches -> 1.0."""
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in text_lower]
    if len(matched) == 0:
        return 0.0, matched
    if len(matched) < min_matches:
        return 0.5, matched
    return 1.0, matched


def _check_constraint(constraint, world: WorldState) -> bool:
    """Evaluate a single ConstraintDef against the world state."""
    ct = constraint.check_type
    params = constraint.check_params

    if ct == "calendar_no_conflict":
        return _calendar_has_no_overlaps(world)

    elif ct == "event_exists":
        eid = params.get("event_id", "")
        return eid in world.calendar

    elif ct == "event_cancelled":
        eid = params.get("event_id", "")
        return eid not in world.calendar

    elif ct == "email_sent":
        to = params.get("to", "").lower()
        keywords = params.get("keywords", [])
        for em in world.emails_sent:
            if to in em.get("to", "").lower():
                if keywords:
                    score, _ = _keyword_score(em.get("body", ""), keywords, min_matches=1)
                    if score > 0:
                        return True
                else:
                    return True
        return False

    elif ct == "restaurant_match":
        name = params.get("name", "")
        if name:
            return world.booked_restaurant.lower() == name.lower()
        criteria = params.get("criteria", {})
        if not world.booked_restaurant:
            return False
        r = world.restaurants.get(world.booked_restaurant)
        if r is None:
            return False
        if "dietary" in criteria and criteria["dietary"].lower() not in [d.lower() for d in r.dietary_options]:
            return False
        if "max_price" in criteria and r.price_per_person > criteria["max_price"]:
            return False
        if "max_distance" in criteria and r.distance_miles > criteria["max_distance"]:
            return False
        if "near_airport" in criteria and criteria["near_airport"] and not r.near_airport:
            return False
        return True

    elif ct == "priority_order":
        higher = params.get("higher", "").lower()
        lower = params.get("lower", "").lower()
        higher_kept = any(
            ev.title.lower() == higher or higher in ev.title.lower()
            for ev in world.calendar.values()
        )
        lower_moved = not any(
            ev.title.lower() == lower or lower in ev.title.lower()
            for ev in world.calendar.values()
        ) or any(
            em.get("to", "").lower() == lower or lower in em.get("body", "").lower()
            for em in world.emails_sent
        )
        return higher_kept

    return False


def _calendar_has_no_overlaps(world: WorldState) -> bool:
    events = list(world.calendar.values())
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            if a.date != b.date:
                continue
            a_start = _time_to_min(a.time)
            a_end = a_start + a.duration_min
            b_start = _time_to_min(b.time)
            b_end = b_start + b.duration_min
            if a_start < b_end and b_start < a_end:
                return False
    return True


def _score_constraint_satisfaction(scenario: ScenarioDef, world: WorldState) -> Tuple[float, str]:
    if not scenario.constraints:
        return 1.0, "No constraints defined"
    met = sum(1 for c in scenario.constraints if _check_constraint(c, world))
    total = len(scenario.constraints)
    score = met / total
    return score, f"{met}/{total} constraints met"


def _score_conflict_resolution(world: WorldState) -> Tuple[float, str]:
    clean = _calendar_has_no_overlaps(world)
    return (1.0 if clean else 0.0), ("No calendar conflicts" if clean else "Calendar has overlapping events")


def _score_commitment_coherence(world: WorldState) -> Tuple[float, str]:
    total = len(world.commitment_ledger)
    if total == 0:
        return 1.0, "No commitments created"
    violations = world.get_silent_violations()
    silent_count = len(violations)

    renegotiated = sum(1 for c in world.commitment_ledger if c.renegotiated_at is not None)
    honored = total - silent_count - renegotiated

    score = (total - silent_count) / total
    parts = []
    if honored > 0:
        parts.append(f"{honored} honored")
    if renegotiated > 0:
        parts.append(f"{renegotiated} renegotiated")
    if silent_count > 0:
        parts.append(f"{silent_count} SILENTLY BROKEN")
    return score, " | ".join(parts) if parts else "OK"


def _score_communication(scenario: ScenarioDef, world: WorldState) -> Tuple[float, str]:
    reqs = scenario.communication_requirements
    if not reqs:
        return 1.0, "No communication requirements"

    total_score = 0.0
    feedback_parts: List[str] = []
    for req in reqs:
        to_lower = req.to.lower()
        matching_emails = [
            em for em in world.emails_sent
            if to_lower in em.get("to", "").lower()
        ]
        if not matching_emails:
            feedback_parts.append(f"MISSING email to {req.to}")
            continue

        best_score = 0.0
        for em in matching_emails:
            body = em.get("body", "") + " " + em.get("subject", "")
            if req.required_keywords:
                ks, matched = _keyword_score(body, req.required_keywords, min_matches=1)
                best_score = max(best_score, ks)
            else:
                best_score = 1.0

        total_score += best_score
        if best_score >= 1.0:
            feedback_parts.append(f"Email to {req.to}: full credit")
        elif best_score > 0:
            feedback_parts.append(f"Email to {req.to}: partial ({best_score:.1f})")
        else:
            feedback_parts.append(f"Email to {req.to}: missing keywords")

    score = total_score / len(reqs) if reqs else 1.0
    return score, " | ".join(feedback_parts)


def _score_step_efficiency(scenario: ScenarioDef, world: WorldState) -> Tuple[float, str]:
    optimal = scenario.optimal_steps
    actual = world.step_count
    if actual <= optimal:
        return 1.0, f"{actual} steps (optimal: {optimal})"
    penalty = (actual - optimal) * 0.1
    score = max(0.0, 1.0 - penalty)
    return score, f"{actual} steps (optimal: {optimal}, penalty: -{penalty:.1f})"


def grade_scenario(
    scenario: ScenarioDef,
    world: WorldState,
) -> Tuple[float, Dict[str, float], str]:
    """Returns ``(total_reward, breakdown, feedback)``."""
    breakdown: Dict[str, float] = {}
    feedback_parts: List[str] = []

    cs_score, cs_fb = _score_constraint_satisfaction(scenario, world)
    breakdown["constraint_satisfaction"] = round(cs_score * WEIGHTS["constraint_satisfaction"], 4)
    feedback_parts.append(f"[constraints] {cs_fb}")

    cr_score, cr_fb = _score_conflict_resolution(world)
    breakdown["conflict_resolution"] = round(cr_score * WEIGHTS["conflict_resolution"], 4)
    feedback_parts.append(f"[conflicts] {cr_fb}")

    cc_score, cc_fb = _score_commitment_coherence(world)
    breakdown["commitment_coherence"] = round(cc_score * WEIGHTS["commitment_coherence"], 4)
    feedback_parts.append(f"[commitments] {cc_fb}")

    cq_score, cq_fb = _score_communication(scenario, world)
    breakdown["communication_quality"] = round(cq_score * WEIGHTS["communication_quality"], 4)
    feedback_parts.append(f"[communication] {cq_fb}")

    se_score, se_fb = _score_step_efficiency(scenario, world)
    breakdown["step_efficiency"] = round(se_score * WEIGHTS["step_efficiency"], 4)
    feedback_parts.append(f"[efficiency] {se_fb}")

    total_reward = round(sum(breakdown.values()), 4)
    total_reward = max(0.01, min(0.99, total_reward))

    feedback = " | ".join(feedback_parts)
    return total_reward, breakdown, feedback
