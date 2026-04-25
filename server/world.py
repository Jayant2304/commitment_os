"""Simulated personal world — calendar, contacts, restaurants, email state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from server.domain import (
    CalendarEvent,
    Commitment,
    Contact,
    InboxEmail,
    Restaurant,
    ScenarioDef,
)


class WorldState:
    """Mutable in-memory state for a single episode."""

    def __init__(self, scenario: ScenarioDef) -> None:
        self.scenario = scenario
        self.calendar: Dict[str, CalendarEvent] = {
            e.event_id: deepcopy(e) for e in scenario.initial_calendar
        }
        self.contacts: Dict[str, Contact] = {
            c.name: deepcopy(c) for c in scenario.contacts
        }
        self.restaurants: Dict[str, Restaurant] = {
            r.name: deepcopy(r) for r in scenario.available_restaurants
        }
        self.inbox: List[InboxEmail] = deepcopy(scenario.initial_inbox)
        self.emails_sent: List[Dict[str, str]] = []
        self.commitment_ledger: List[Commitment] = []
        self.step_count: int = 0
        self.booked_restaurant: str = ""
        self._next_event_id: int = 100

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def view_calendar(self, date: str) -> str:
        events = [
            e for e in self.calendar.values()
            if e.date == date
        ]
        if not events:
            return f"No events on {date}."
        events.sort(key=lambda e: e.time)
        lines = [f"Calendar for {date}:"]
        for ev in events:
            parts = ev.participants
            part_str = f" with {', '.join(parts)}" if parts else ""
            loc_str = f" at {ev.location}" if ev.location else ""
            lines.append(
                f"  [{ev.event_id}] {ev.time} ({ev.duration_min}min) "
                f"{ev.title}{part_str}{loc_str} "
                f"[priority={ev.priority}]"
            )
        return "\n".join(lines)

    def check_availability(self, person: str) -> str:
        contact = self.contacts.get(person)
        if contact is None:
            return f"Contact '{person}' not found."
        if not contact.availability:
            return f"{person} has no availability information on file."
        lines = [f"Availability for {person} (role: {contact.role}):"]
        for date, slots in sorted(contact.availability.items()):
            lines.append(f"  {date}: {', '.join(slots)}")
        if contact.dietary:
            lines.append(f"  Dietary: {contact.dietary}")
        return "\n".join(lines)

    def search_restaurants(
        self,
        cuisine: str = "",
        max_price: int = 0,
        dietary: str = "",
        max_distance_miles: float = 0.0,
        near_airport: bool = False,
    ) -> str:
        matches: List[Restaurant] = []
        for r in self.restaurants.values():
            if cuisine and cuisine.lower() not in r.cuisine.lower():
                continue
            if max_price > 0 and r.price_per_person > max_price:
                continue
            if dietary and dietary.lower() not in [d.lower() for d in r.dietary_options]:
                continue
            if max_distance_miles > 0 and r.distance_miles > max_distance_miles:
                continue
            if near_airport and not r.near_airport:
                continue
            matches.append(r)

        if not matches:
            return "No restaurants match your criteria."
        lines = ["Matching restaurants:"]
        for r in matches:
            lines.append(
                f"  {r.name} — {r.cuisine}, ${r.price_per_person}/pp, "
                f"{r.distance_miles}mi, dietary: {', '.join(r.dietary_options)}, "
                f"capacity: {r.capacity}, hours: {r.hours}"
                f"{', near airport' if r.near_airport else ''}"
                f"{', private room' if r.has_private_room else ''}"
            )
        return "\n".join(lines)

    def schedule_meeting(
        self,
        title: str,
        date: str,
        time: str,
        duration_min: int = 60,
        participants: Optional[List[str]] = None,
        location: str = "",
        turn: int = 0,
    ) -> str:
        conflict = self._find_conflict(date, time, duration_min)
        if conflict is not None:
            return (
                f"CONFLICT: '{title}' at {time} overlaps with "
                f"'{conflict.title}' at {conflict.time}. "
                f"Resolve the conflict first."
            )

        eid = f"evt_{self._next_event_id}"
        self._next_event_id += 1
        event = CalendarEvent(
            event_id=eid,
            title=title,
            date=date,
            time=time,
            duration_min=duration_min,
            participants=participants or [],
            location=location,
        )
        self.calendar[eid] = event

        self.commitment_ledger.append(Commitment(
            turn_created=turn,
            commitment_type="meeting_scheduled",
            description=f"{time} {title} on {date}",
            constraint=f"{date}T{time}",
            to_whom=", ".join(participants or ["self"]),
        ))

        return f"Meeting scheduled: [{eid}] {date} {time} — {title}"

    def reschedule_event(self, event_id: str, new_time: str, turn: int = 0) -> str:
        event = self.calendar.get(event_id)
        if event is None:
            return f"Event '{event_id}' not found."

        conflict = self._find_conflict(event.date, new_time, event.duration_min, exclude=event_id)
        if conflict is not None:
            return (
                f"CONFLICT: moving '{event.title}' to {new_time} would overlap "
                f"with '{conflict.title}' at {conflict.time}."
            )

        old_time = event.time
        event.time = new_time

        for c in self.commitment_ledger:
            if c.active and c.constraint == f"{event.date}T{old_time}":
                c.active = False
                c.renegotiated_at = turn

        self.commitment_ledger.append(Commitment(
            turn_created=turn,
            commitment_type="meeting_scheduled",
            description=f"{new_time} {event.title} on {event.date} (rescheduled from {old_time})",
            constraint=f"{event.date}T{new_time}",
            to_whom=", ".join(event.participants) if event.participants else "self",
        ))

        return f"Rescheduled [{event_id}] '{event.title}' from {old_time} to {new_time}."

    def cancel_event(self, event_id: str, turn: int = 0) -> str:
        event = self.calendar.pop(event_id, None)
        if event is None:
            return f"Event '{event_id}' not found."

        for c in self.commitment_ledger:
            if c.active and c.constraint == f"{event.date}T{event.time}":
                if event.is_personal:
                    c.active = False
                    c.renegotiated_at = turn
                # non-personal cancellations remain active until email is sent

        return f"Cancelled [{event_id}] '{event.title}' at {event.time} on {event.date}."

    def send_email(self, to: str, subject: str, body: str, turn: int = 0) -> str:
        self.emails_sent.append({
            "to": to,
            "subject": subject,
            "body": body,
            "turn": turn,
        })

        body_lower = body.lower()
        renegotiation_keywords = ["reschedule", "move", "cancel", "change", "instead", "alternative", "postpone"]
        is_renegotiation = any(kw in body_lower for kw in renegotiation_keywords)

        if is_renegotiation:
            for c in self.commitment_ledger:
                if c.active and to.lower() in c.to_whom.lower():
                    c.renegotiated_at = turn

        return f"Email sent to {to}: '{subject}'"

    def book_restaurant(self, restaurant_name: str, turn: int = 0) -> str:
        r = self.restaurants.get(restaurant_name)
        if r is None:
            return f"Restaurant '{restaurant_name}' not found."
        self.booked_restaurant = restaurant_name

        self.commitment_ledger.append(Commitment(
            turn_created=turn,
            commitment_type="reservation_made",
            description=f"Reservation at {restaurant_name}",
            constraint=restaurant_name,
            to_whom="group",
        ))

        return f"Reservation confirmed at {restaurant_name}."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_conflict(
        self, date: str, time: str, duration_min: int, exclude: str = "",
    ) -> Optional[CalendarEvent]:
        new_start = _time_to_min(time)
        new_end = new_start + duration_min
        for eid, ev in self.calendar.items():
            if eid == exclude:
                continue
            if ev.date != date:
                continue
            ev_start = _time_to_min(ev.time)
            ev_end = ev_start + ev.duration_min
            if new_start < ev_end and new_end > ev_start:
                return ev
        return None

    def get_calendar_snapshot(self) -> List[Dict[str, Any]]:
        return [ev.model_dump() for ev in sorted(self.calendar.values(), key=lambda e: (e.date, e.time))]

    def get_inbox_snapshot(self) -> List[Dict[str, Any]]:
        return [e.model_dump(exclude={"context_hint"}) for e in self.inbox]

    def get_active_commitments(self) -> List[Commitment]:
        return [c for c in self.commitment_ledger if c.active]

    def get_silent_violations(self) -> List[Commitment]:
        """Commitments that are still active but whose constraint no longer holds."""
        violations: List[Commitment] = []
        for c in self.commitment_ledger:
            if not c.active:
                continue
            if c.renegotiated_at is not None:
                continue
            if c.commitment_type == "meeting_scheduled":
                time_key = c.constraint
                parts = time_key.split("T")
                if len(parts) == 2:
                    date_str, time_str = parts
                    found = any(
                        ev.date == date_str and ev.time == time_str
                        for ev in self.calendar.values()
                    )
                    if not found:
                        has_email = any(
                            c.to_whom.lower() in em.get("to", "").lower()
                            for em in self.emails_sent
                        )
                        if not has_email:
                            violations.append(c)
        return violations


def _time_to_min(t: str) -> int:
    """Convert 'HH:MM' to minutes since midnight."""
    parts = t.split(":")
    return int(parts[0]) * 60 + int(parts[1])
