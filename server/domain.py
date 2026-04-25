"""Internal domain types — not exposed via the HTTP API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Commitment ledger entry
# ---------------------------------------------------------------------------

@dataclass
class Commitment:
    """A binding constraint the agent created via its own actions."""

    turn_created: int
    commitment_type: str       # "meeting_scheduled" | "email_promise" | "reservation_made"
    description: str           # human-readable: "3pm meeting with Client X"
    constraint: str            # machine key: "2026-04-25T15:00"
    to_whom: str               # who was promised
    active: bool = True
    renegotiated_at: Optional[int] = None


# ---------------------------------------------------------------------------
# Scenario / task definition
# ---------------------------------------------------------------------------

class CalendarEvent(BaseModel):
    """A single calendar entry."""

    event_id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Event title")
    date: str = Field(..., description="ISO date yyyy-mm-dd")
    time: str = Field(..., description="Start time HH:MM")
    duration_min: int = Field(60, description="Duration in minutes")
    participants: List[str] = Field(default_factory=list)
    location: str = Field("", description="Room or location name")
    priority: str = Field("normal", description="low | normal | high | critical")
    is_personal: bool = Field(False, description="Personal vs work event")


class Contact(BaseModel):
    """A person the agent can interact with."""

    name: str
    role: str = ""
    email: str = ""
    priority_level: int = Field(1, description="1 (lowest) to 5 (highest)")
    availability: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="date -> list of free time slots e.g. {'2026-04-25': ['09:00','10:00','14:00']}",
    )
    dietary: str = Field("", description="Dietary restrictions if any")


class Restaurant(BaseModel):
    """A restaurant option the agent can search/book."""

    name: str
    cuisine: str
    price_per_person: int
    distance_miles: float
    dietary_options: List[str] = Field(default_factory=list)
    capacity: int = 20
    hours: str = "11:00-22:00"
    has_private_room: bool = False
    near_airport: bool = False


class InboxEmail(BaseModel):
    """An email in the agent's inbox."""

    email_id: str
    sender: str
    subject: str
    body: str
    urgency: str = Field("normal", description="low | normal | high | critical")
    received_at: str = Field("", description="ISO datetime")
    requires_response: bool = True
    context_hint: str = Field("", description="Hidden hint for grader about what the correct action is")


class ConstraintDef(BaseModel):
    """A single verifiable constraint for grading."""

    description: str = Field(..., description="Human-readable: 'Restaurant must have vegan options'")
    check_type: str = Field(..., description="'calendar_no_conflict' | 'restaurant_match' | 'email_sent' | 'event_exists' | 'event_cancelled' | 'priority_order'")
    check_params: Dict[str, Any] = Field(default_factory=dict)


class CommunicationReq(BaseModel):
    """A required outgoing communication for grading."""

    to: str = Field(..., description="Recipient name")
    required_keywords: List[str] = Field(default_factory=list, description="Keywords that should appear")
    purpose: str = Field("", description="'notify_reschedule' | 'propose_alternative' | 'acknowledge' | 'renegotiate'")


class ScenarioDef(BaseModel):
    """Complete definition of a single task scenario."""

    scenario_id: str
    difficulty: str = Field(..., description="easy | medium | hard")
    briefing: str = Field(..., description="The scenario description the agent sees on reset")
    initial_calendar: List[CalendarEvent] = Field(default_factory=list)
    initial_inbox: List[InboxEmail] = Field(default_factory=list)
    available_restaurants: List[Restaurant] = Field(default_factory=list)
    contacts: List[Contact] = Field(default_factory=list)
    constraints: List[ConstraintDef] = Field(default_factory=list)
    priority_ordering: List[str] = Field(
        default_factory=list,
        description="Ordered list from highest to lowest priority contact/event",
    )
    communication_requirements: List[CommunicationReq] = Field(default_factory=list)
    optimal_steps: int = Field(3, description="Minimum steps to solve perfectly")
    max_steps: int = Field(15, description="Maximum allowed steps before timeout")

    # ground-truth for grading
    expected_final_events: List[str] = Field(
        default_factory=list,
        description="Event IDs that should exist in final calendar",
    )
    expected_cancelled_events: List[str] = Field(
        default_factory=list,
        description="Event IDs that should be cancelled",
    )
    expected_restaurant: str = Field("", description="Name of the correct restaurant pick")
