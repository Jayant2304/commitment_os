"""API-facing Pydantic models — the public contract of CommitmentOS."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class CommitmentAction(Action):
    """Agent's tool call submitted via POST /step.

    Each step is one tool invocation. The agent fills ``action_type`` and
    the relevant subset of optional parameters for that tool.
    """

    action_type: str = Field(
        ...,
        description=(
            "Tool to invoke: 'view_calendar' | 'check_availability' | "
            "'search_restaurants' | 'schedule_meeting' | 'reschedule_event' | "
            "'cancel_event' | 'send_email' | 'submit_plan'"
        ),
    )

    # calendar operations
    date: str = Field("", description="ISO date for calendar queries (yyyy-mm-dd)")
    event_id: str = Field("", description="Event ID for reschedule / cancel")
    new_time: str = Field("", description="New start time HH:MM for reschedule")
    title: str = Field("", description="Title for new meetings")
    participants: List[str] = Field(default_factory=list, description="Attendee names")
    time: str = Field("", description="Start time HH:MM for new meetings")
    duration_min: int = Field(60, description="Meeting duration in minutes")
    location: str = Field("", description="Room or location")

    # contact queries
    person: str = Field("", description="Contact name for availability check")

    # restaurant search
    cuisine: str = Field("", description="Cuisine filter")
    max_price: int = Field(0, description="Max price per person (0 = no limit)")
    dietary: str = Field("", description="Dietary requirement filter")
    max_distance_miles: float = Field(0.0, description="Max distance (0 = no limit)")
    near_airport: bool = Field(False, description="Filter for airport proximity")
    restaurant_name: str = Field("", description="Specific restaurant to book")

    # email
    to: str = Field("", description="Recipient name for send_email")
    subject: str = Field("", description="Email subject line")
    body: str = Field("", description="Email body text")


class CommitmentObservation(Observation):
    """Observation from reset() and step(). Inherits ``done``, ``reward``."""

    scenario_id: str = Field(default="", description="Current scenario identifier")
    difficulty: str = Field(default="", description="easy | medium | hard")
    briefing: str = Field(default="", description="Scenario description shown on reset")
    tool_result: str = Field(default="", description="Output of the last tool call")
    calendar_snapshot: List[Dict[str, Any]] = Field(
        default_factory=list, description="Current calendar events",
    )
    inbox: List[Dict[str, Any]] = Field(
        default_factory=list, description="Unread inbox emails",
    )
    pending_commitments: int = Field(0, description="Number of active commitments in ledger")
    step_number: int = Field(0, description="Current step within this episode")
    max_steps: int = Field(15, description="Maximum steps before forced submission")
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Per-component reward scores",
    )
    feedback: str = Field(default="", description="Human-readable grader feedback")


class CommitmentState(State):
    """Episode metadata from GET /state."""

    scenario_id: str = Field(default="", description="Current scenario identifier")
    difficulty: str = Field(default="", description="Current difficulty level")
    completed: bool = Field(default=False, description="Whether episode is finished")
    cumulative_reward: float = Field(default=0.0, description="Sum of rewards this episode")
    commitment_count: int = Field(default=0, description="Total commitments created")
    violation_count: int = Field(default=0, description="Silent commitment violations")
    available_tasks: List[str] = Field(
        default_factory=list, description="All scenario IDs in the dataset",
    )
