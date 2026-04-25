"""Scenario dataset — 15 tasks across 3 difficulty tiers.

Each scenario is a validated ``ScenarioDef`` Pydantic model containing the
initial world state and deterministic grader keys.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from server.domain import (
    CalendarEvent,
    CommunicationReq,
    ConstraintDef,
    Contact,
    InboxEmail,
    Restaurant,
    ScenarioDef,
)

# ===================================================================
# EASY — 2-4 tool calls, single constraint domain
# ===================================================================

_EASY_001 = ScenarioDef(
    scenario_id="easy_001",
    difficulty="easy",
    briefing=(
        "You have two meetings at 2:00 PM today (2026-04-25): a 1-on-1 with your boss "
        "VP_Chen and a team standup with 6 people. Both are in different rooms. "
        "VP_Chen's meeting is higher priority. Reschedule the standup to a free slot "
        "and notify the team."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_1", title="1-on-1 with VP_Chen", date="2026-04-25", time="14:00", duration_min=30, participants=["VP_Chen"], location="Room A", priority="high"),
        CalendarEvent(event_id="evt_2", title="Team Standup", date="2026-04-25", time="14:00", duration_min=30, participants=["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"], location="Room B", priority="normal"),
        CalendarEvent(event_id="evt_3", title="Lunch", date="2026-04-25", time="12:00", duration_min=60, participants=[], priority="low", is_personal=True),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_1", sender="VP_Chen", subject="Our 1-on-1 today", body="Looking forward to our 2pm chat. I have some feedback on the Q3 roadmap.", urgency="high"),
    ],
    contacts=[
        Contact(name="VP_Chen", role="VP Engineering", priority_level=5),
        Contact(name="Alice", role="Engineer", priority_level=2),
        Contact(name="Team", role="Engineering Team", priority_level=2, email="team@company.com"),
    ],
    constraints=[
        ConstraintDef(description="1-on-1 with VP_Chen must remain at 14:00", check_type="event_exists", check_params={"event_id": "evt_1"}),
        ConstraintDef(description="Team standup must not conflict with 1-on-1", check_type="calendar_no_conflict", check_params={}),
        ConstraintDef(description="Team must be notified of reschedule", check_type="email_sent", check_params={"to": "Team", "keywords": ["reschedule", "standup", "move"]}),
    ],
    priority_ordering=["VP_Chen", "Team"],
    communication_requirements=[
        CommunicationReq(to="Team", required_keywords=["reschedule", "standup"], purpose="notify_reschedule"),
    ],
    optimal_steps=3,
    max_steps=8,
    expected_cancelled_events=[],
    expected_final_events=["evt_1"],
)

_EASY_002 = ScenarioDef(
    scenario_id="easy_002",
    difficulty="easy",
    briefing=(
        "Book a dinner tonight (2026-04-25) for 4 people. Requirements: "
        "Italian cuisine, under $50 per person, within 3 miles. "
        "Search restaurants and book the best match."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_10", title="Morning Standup", date="2026-04-25", time="09:00", duration_min=30, participants=["Team"]),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_10", sender="Alice", subject="Dinner tonight?", body="Can you find us a nice Italian place? Budget is $50/person max. Needs to be close to the office.", urgency="normal"),
    ],
    available_restaurants=[
        Restaurant(name="Bella Italia", cuisine="Italian", price_per_person=40, distance_miles=2.0, dietary_options=["vegetarian", "gluten-free"], capacity=30),
        Restaurant(name="Chez Pierre", cuisine="French", price_per_person=80, distance_miles=1.5, dietary_options=["vegetarian"], capacity=40),
        Restaurant(name="Pasta Palace", cuisine="Italian", price_per_person=55, distance_miles=1.0, dietary_options=["vegan", "vegetarian"], capacity=20),
        Restaurant(name="Dragon Wok", cuisine="Chinese", price_per_person=25, distance_miles=4.0, dietary_options=["vegan", "vegetarian"], capacity=50),
    ],
    contacts=[
        Contact(name="Alice", role="Friend", priority_level=2),
    ],
    constraints=[
        ConstraintDef(description="Restaurant must be Italian", check_type="restaurant_match", check_params={"criteria": {"dietary": ""}}),
        ConstraintDef(description="Restaurant must be under $50/pp", check_type="restaurant_match", check_params={"criteria": {"max_price": 50}}),
        ConstraintDef(description="Restaurant must be within 3 miles", check_type="restaurant_match", check_params={"criteria": {"max_distance": 3.0}}),
    ],
    optimal_steps=2,
    max_steps=6,
    expected_restaurant="Bella Italia",
)

_EASY_003 = ScenarioDef(
    scenario_id="easy_003",
    difficulty="easy",
    briefing=(
        "Client_Jones has emailed asking for a meeting this week. Check your "
        "calendar for 2026-04-25 and Client_Jones's availability, then propose "
        "3 available slots via email."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_20", title="Team Sync", date="2026-04-25", time="10:00", duration_min=60, participants=["Team"]),
        CalendarEvent(event_id="evt_21", title="Lunch", date="2026-04-25", time="12:00", duration_min=60, is_personal=True),
        CalendarEvent(event_id="evt_22", title="Design Review", date="2026-04-25", time="15:00", duration_min=60, participants=["Bob", "Carol"]),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_20", sender="Client_Jones", subject="Meeting this week?", body="Hi, I'd love to catch up this week. Do you have any openings? Need about 30 minutes.", urgency="high"),
    ],
    contacts=[
        Contact(name="Client_Jones", role="Client", priority_level=4, availability={"2026-04-25": ["09:00", "11:00", "14:00", "16:00"]}),
    ],
    constraints=[
        ConstraintDef(description="Email must be sent to Client_Jones", check_type="email_sent", check_params={"to": "Client_Jones", "keywords": ["slot", "available", "meet"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="Client_Jones", required_keywords=["available", "slot", "time"], purpose="propose_slots"),
    ],
    optimal_steps=3,
    max_steps=8,
)

_EASY_004 = ScenarioDef(
    scenario_id="easy_004",
    difficulty="easy",
    briefing=(
        "Your personal doctor appointment at 3:00 PM today (2026-04-25) conflicts "
        "with the weekly team sync. The doctor appointment was booked first and is "
        "important. Cancel the team sync and notify the team."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_30", title="Weekly Team Sync", date="2026-04-25", time="15:00", duration_min=60, participants=["Team"], priority="normal"),
        CalendarEvent(event_id="evt_31", title="Doctor Appointment", date="2026-04-25", time="15:00", duration_min=60, priority="high", is_personal=True),
    ],
    initial_inbox=[],
    contacts=[
        Contact(name="Team", role="Engineering Team", priority_level=2),
    ],
    constraints=[
        ConstraintDef(description="Doctor appointment must remain", check_type="event_exists", check_params={"event_id": "evt_31"}),
        ConstraintDef(description="Team sync must be cancelled", check_type="event_cancelled", check_params={"event_id": "evt_30"}),
        ConstraintDef(description="Team must be notified", check_type="email_sent", check_params={"to": "Team", "keywords": ["cancel", "sync"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="Team", required_keywords=["cancel", "sync", "apologi"], purpose="notify_reschedule"),
    ],
    optimal_steps=2,
    max_steps=6,
    expected_cancelled_events=["evt_30"],
    expected_final_events=["evt_31"],
)

_EASY_005 = ScenarioDef(
    scenario_id="easy_005",
    difficulty="easy",
    briefing=(
        "You have 3 unread emails. Triage them by urgency and respond to the most "
        "urgent one first. VP_Chen's email is critical, Client_Jones is high, "
        "and Alice is normal priority."
    ),
    initial_calendar=[],
    initial_inbox=[
        InboxEmail(email_id="em_50", sender="Alice", subject="Lunch tomorrow?", body="Want to grab lunch tomorrow?", urgency="low"),
        InboxEmail(email_id="em_51", sender="Client_Jones", subject="Contract review", body="Please review the attached contract by end of day.", urgency="high"),
        InboxEmail(email_id="em_52", sender="VP_Chen", subject="URGENT: Board deck", body="I need the Q3 numbers for the board deck. Can you send them in the next hour?", urgency="critical"),
    ],
    contacts=[
        Contact(name="VP_Chen", role="VP Engineering", priority_level=5),
        Contact(name="Client_Jones", role="Client", priority_level=4),
        Contact(name="Alice", role="Engineer", priority_level=2),
    ],
    constraints=[
        ConstraintDef(description="VP_Chen must be responded to", check_type="email_sent", check_params={"to": "VP_Chen", "keywords": ["Q3", "number", "board"]}),
        ConstraintDef(description="Client_Jones must be responded to", check_type="email_sent", check_params={"to": "Client_Jones", "keywords": ["contract", "review"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="VP_Chen", required_keywords=["Q3", "numbers", "send"], purpose="acknowledge"),
        CommunicationReq(to="Client_Jones", required_keywords=["contract", "review"], purpose="acknowledge"),
    ],
    optimal_steps=2,
    max_steps=6,
)

# ===================================================================
# MEDIUM — 5-8 tool calls, cross-domain with commitment tracking
# ===================================================================

_MED_006 = ScenarioDef(
    scenario_id="med_006",
    difficulty="medium",
    briefing=(
        "Meeting A ('Design Review') has been moved from 2:00 PM to 3:00 PM today "
        "(2026-04-25). But you have Meeting B ('Sprint Planning') at 3:00 PM, and "
        "Meeting C ('Demo Prep') at 4:00 PM depends on Sprint Planning's output. "
        "Resolve the cascade: reschedule B without conflicting with C, and notify "
        "all affected parties."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_40", title="Design Review", date="2026-04-25", time="14:00", duration_min=60, participants=["Bob", "Carol"], priority="high"),
        CalendarEvent(event_id="evt_41", title="Sprint Planning", date="2026-04-25", time="15:00", duration_min=60, participants=["Team"], priority="normal"),
        CalendarEvent(event_id="evt_42", title="Demo Prep", date="2026-04-25", time="16:00", duration_min=60, participants=["Alice", "Dave"], priority="normal"),
        CalendarEvent(event_id="evt_43", title="Morning Standup", date="2026-04-25", time="09:00", duration_min=30, participants=["Team"]),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_40", sender="Bob", subject="Design Review moved", body="Hey, I need to push our 2pm design review to 3pm. Apologies for the late change.", urgency="high"),
    ],
    contacts=[
        Contact(name="Bob", role="Lead Designer", priority_level=3),
        Contact(name="Team", role="Engineering Team", priority_level=2),
        Contact(name="Alice", role="Engineer", priority_level=2),
    ],
    constraints=[
        ConstraintDef(description="Design Review must be at 15:00", check_type="calendar_no_conflict", check_params={}),
        ConstraintDef(description="Sprint Planning must not conflict", check_type="calendar_no_conflict", check_params={}),
        ConstraintDef(description="Demo Prep must remain after Sprint Planning", check_type="event_exists", check_params={"event_id": "evt_42"}),
        ConstraintDef(description="Team notified about Sprint Planning change", check_type="email_sent", check_params={"to": "Team", "keywords": ["sprint", "reschedule", "move"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="Team", required_keywords=["sprint", "planning", "reschedule"], purpose="notify_reschedule"),
    ],
    optimal_steps=4,
    max_steps=10,
)

_MED_007 = ScenarioDef(
    scenario_id="med_007",
    difficulty="medium",
    briefing=(
        "Plan a team dinner for 6 people tonight (2026-04-25). Constraints: "
        "Alice is vegan, Bob has a nut allergy, must be within 3 miles, "
        "under $45 per person, and needs a private room for 6+. "
        "Search restaurants, book the right one, and email the team with details."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_50", title="Afternoon Focus", date="2026-04-25", time="14:00", duration_min=120),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_50", sender="Alice", subject="Dinner tonight", body="Can you book a place? Remember I'm vegan. Bob has a nut allergy. We need a private room.", urgency="normal"),
    ],
    available_restaurants=[
        Restaurant(name="Green Garden", cuisine="Mediterranean", price_per_person=38, distance_miles=2.5, dietary_options=["vegan", "nut-free", "vegetarian"], capacity=30, has_private_room=True),
        Restaurant(name="Steak House Prime", cuisine="American", price_per_person=55, distance_miles=1.0, dietary_options=["gluten-free"], capacity=50, has_private_room=True),
        Restaurant(name="Lotus Thai", cuisine="Thai", price_per_person=30, distance_miles=3.5, dietary_options=["vegan", "vegetarian"], capacity=25, has_private_room=False),
        Restaurant(name="Cafe Novo", cuisine="Fusion", price_per_person=42, distance_miles=2.0, dietary_options=["vegan", "nut-free", "gluten-free", "vegetarian"], capacity=15, has_private_room=True),
        Restaurant(name="Burgers & Brew", cuisine="American", price_per_person=20, distance_miles=0.5, dietary_options=["vegetarian"], capacity=40, has_private_room=False),
    ],
    contacts=[
        Contact(name="Alice", role="Engineer", priority_level=2, dietary="vegan"),
        Contact(name="Bob", role="Engineer", priority_level=2, dietary="nut-free"),
        Contact(name="Team", role="Engineering Team", priority_level=2),
    ],
    constraints=[
        ConstraintDef(description="Restaurant has vegan options", check_type="restaurant_match", check_params={"criteria": {"dietary": "vegan"}}),
        ConstraintDef(description="Restaurant under $45/pp", check_type="restaurant_match", check_params={"criteria": {"max_price": 45}}),
        ConstraintDef(description="Restaurant within 3 miles", check_type="restaurant_match", check_params={"criteria": {"max_distance": 3.0}}),
        ConstraintDef(description="Team notified of dinner details", check_type="email_sent", check_params={"to": "Team", "keywords": ["dinner", "restaurant"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="Team", required_keywords=["dinner", "tonight", "restaurant"], purpose="notify_reschedule"),
    ],
    optimal_steps=3,
    max_steps=8,
    expected_restaurant="Green Garden",
)

_MED_008 = ScenarioDef(
    scenario_id="med_008",
    difficulty="medium",
    briefing=(
        "You are currently in a client call (Client_Jones) that ends at 3:15 PM. "
        "Your boss VP_Chen just emailed saying 'Need Q3 numbers in 30 minutes — "
        "board meeting moved up.' It's currently 2:45 PM on 2026-04-25. "
        "You cannot leave the client call early. Acknowledge VP_Chen with a "
        "realistic ETA and do NOT cancel the client meeting."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_60", title="Client Call with Jones", date="2026-04-25", time="14:30", duration_min=45, participants=["Client_Jones"], priority="high"),
        CalendarEvent(event_id="evt_61", title="Focus Time", date="2026-04-25", time="16:00", duration_min=60, priority="low"),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_60", sender="VP_Chen", subject="URGENT: Q3 numbers NOW", body="Board meeting moved up. I need the Q3 revenue numbers in the next 30 minutes. This is critical.", urgency="critical"),
    ],
    contacts=[
        Contact(name="VP_Chen", role="VP Engineering", priority_level=5),
        Contact(name="Client_Jones", role="Client", priority_level=4),
    ],
    constraints=[
        ConstraintDef(description="Client call must NOT be cancelled", check_type="event_exists", check_params={"event_id": "evt_60"}),
        ConstraintDef(description="VP_Chen must be acknowledged", check_type="email_sent", check_params={"to": "VP_Chen", "keywords": ["Q3", "numbers"]}),
        ConstraintDef(description="Realistic ETA communicated", check_type="email_sent", check_params={"to": "VP_Chen", "keywords": ["after", "3:15", "call", "send"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="VP_Chen", required_keywords=["Q3", "numbers", "after", "client"], purpose="acknowledge"),
    ],
    optimal_steps=2,
    max_steps=6,
    expected_final_events=["evt_60"],
)

_MED_009 = ScenarioDef(
    scenario_id="med_009",
    difficulty="medium",
    briefing=(
        "You received an email from Bob saying 'Can we push our thing to next week?' "
        "You have 3 recurring meetings with Bob: Monday Design Review (evt_70), "
        "Wednesday Code Review (evt_71), and Friday Retrospective (evt_72) — all on "
        "different days this week (2026-04-25 is Friday). Check the context and "
        "determine which meeting Bob means, then confirm via email."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_70", title="Design Review with Bob", date="2026-04-21", time="10:00", duration_min=60, participants=["Bob"]),
        CalendarEvent(event_id="evt_71", title="Code Review with Bob", date="2026-04-23", time="14:00", duration_min=60, participants=["Bob"]),
        CalendarEvent(event_id="evt_72", title="Retrospective with Bob", date="2026-04-25", time="11:00", duration_min=60, participants=["Bob"]),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_70", sender="Bob", subject="Push our thing?", body="Hey, can we push our thing to next week? I'm swamped with the release today.", urgency="normal", context_hint="Bob means the Retrospective (today, Friday) since he says 'today'"),
    ],
    contacts=[
        Contact(name="Bob", role="Lead Designer", priority_level=3, availability={"2026-05-02": ["11:00", "14:00"]}),
    ],
    constraints=[
        ConstraintDef(description="Bob must be responded to", check_type="email_sent", check_params={"to": "Bob", "keywords": ["retrospective", "next week"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="Bob", required_keywords=["retrospective", "next week", "reschedule"], purpose="renegotiate"),
    ],
    optimal_steps=4,
    max_steps=10,
)

_MED_010 = ScenarioDef(
    scenario_id="med_010",
    difficulty="medium",
    briefing=(
        "Client_Jones is visiting your office tomorrow (2026-04-26). You need to: "
        "(1) book a conference room for a 10 AM demo, "
        "(2) arrange lunch at a restaurant with vegetarian options, "
        "and (3) send Client_Jones an itinerary email with all details."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_80", title="Team Standup", date="2026-04-26", time="09:00", duration_min=30, participants=["Team"]),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_80", sender="Client_Jones", subject="Visit tomorrow", body="Looking forward to the demo tomorrow. Is 10am still good? I'm vegetarian by the way.", urgency="high"),
    ],
    available_restaurants=[
        Restaurant(name="Garden Bistro", cuisine="Mediterranean", price_per_person=35, distance_miles=0.5, dietary_options=["vegetarian", "vegan"], capacity=20),
        Restaurant(name="BBQ Pit", cuisine="American BBQ", price_per_person=30, distance_miles=1.0, dietary_options=[], capacity=40),
    ],
    contacts=[
        Contact(name="Client_Jones", role="Client", priority_level=4, availability={"2026-04-26": ["10:00", "11:00", "12:00", "13:00"]}, dietary="vegetarian"),
    ],
    constraints=[
        ConstraintDef(description="Demo meeting scheduled at 10:00", check_type="calendar_no_conflict", check_params={}),
        ConstraintDef(description="Restaurant has vegetarian options", check_type="restaurant_match", check_params={"criteria": {"dietary": "vegetarian"}}),
        ConstraintDef(description="Client_Jones receives itinerary", check_type="email_sent", check_params={"to": "Client_Jones", "keywords": ["itinerary", "10", "demo", "lunch"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="Client_Jones", required_keywords=["itinerary", "demo", "lunch", "10"], purpose="notify_reschedule"),
    ],
    optimal_steps=4,
    max_steps=10,
    expected_restaurant="Garden Bistro",
)

# ===================================================================
# HARD — 8-15 tool calls, full cross-task cascade + SRE crisis
# ===================================================================

_HARD_011 = ScenarioDef(
    scenario_id="hard_011",
    difficulty="hard",
    briefing=(
        "VP_Chen just emailed: an important investor (Investor_Park) is in town tonight "
        "(2026-04-25) and needs a dinner meeting. Investor_Park has a 9:00 PM flight "
        "so dinner must end by 8:00 PM. Investor_Park is vegetarian. Your calendar: "
        "6:00 PM Yoga (personal), 7:00 PM Team Happy Hour (you organised it and "
        "promised the team last week). You must: find a restaurant near the airport "
        "with vegetarian options under $60/pp, handle the calendar conflicts by "
        "priority (investor > happy hour > yoga), and email everyone affected."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_90", title="Yoga", date="2026-04-25", time="18:00", duration_min=60, priority="low", is_personal=True),
        CalendarEvent(event_id="evt_91", title="Team Happy Hour", date="2026-04-25", time="19:00", duration_min=120, participants=["Team"], priority="normal"),
        CalendarEvent(event_id="evt_92", title="Afternoon Focus", date="2026-04-25", time="14:00", duration_min=120),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_90", sender="VP_Chen", subject="Investor dinner TONIGHT", body="Investor_Park is in town tonight only. We need dinner before their 9pm flight. They're vegetarian. Book something near the airport. This is top priority.", urgency="critical"),
    ],
    available_restaurants=[
        Restaurant(name="Sky Lounge", cuisine="International", price_per_person=55, distance_miles=1.0, dietary_options=["vegetarian", "vegan", "gluten-free"], capacity=30, near_airport=True, has_private_room=True),
        Restaurant(name="Terminal Grill", cuisine="American", price_per_person=35, distance_miles=0.5, dietary_options=["vegetarian"], capacity=50, near_airport=True),
        Restaurant(name="Downtown Sushi", cuisine="Japanese", price_per_person=45, distance_miles=8.0, dietary_options=["vegetarian"], capacity=20),
        Restaurant(name="Fancy Steak", cuisine="Steakhouse", price_per_person=70, distance_miles=0.8, dietary_options=[], capacity=40, near_airport=True),
    ],
    contacts=[
        Contact(name="VP_Chen", role="VP Engineering", priority_level=5),
        Contact(name="Investor_Park", role="Investor", priority_level=5, dietary="vegetarian"),
        Contact(name="Team", role="Engineering Team", priority_level=2),
    ],
    constraints=[
        ConstraintDef(description="Restaurant near airport", check_type="restaurant_match", check_params={"criteria": {"near_airport": True}}),
        ConstraintDef(description="Restaurant has vegetarian options", check_type="restaurant_match", check_params={"criteria": {"dietary": "vegetarian"}}),
        ConstraintDef(description="Restaurant under $60/pp", check_type="restaurant_match", check_params={"criteria": {"max_price": 60}}),
        ConstraintDef(description="Yoga cancelled (lower priority)", check_type="event_cancelled", check_params={"event_id": "evt_90"}),
        ConstraintDef(description="Team notified about Happy Hour change", check_type="email_sent", check_params={"to": "Team", "keywords": ["happy hour", "reschedule", "sorry"]}),
        ConstraintDef(description="VP_Chen sent dinner plan", check_type="email_sent", check_params={"to": "VP_Chen", "keywords": ["dinner", "restaurant", "investor"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="Team", required_keywords=["happy hour", "reschedule", "sorry", "apologi"], purpose="renegotiate"),
        CommunicationReq(to="VP_Chen", required_keywords=["dinner", "restaurant", "investor", "vegetarian"], purpose="acknowledge"),
    ],
    optimal_steps=7,
    max_steps=15,
    expected_restaurant="Sky Lounge",
    expected_cancelled_events=["evt_90"],
)

_HARD_012 = ScenarioDef(
    scenario_id="hard_012",
    difficulty="hard",
    briefing=(
        "Three VPs all want Conference Room Alpha at 2:00 PM today (2026-04-25) for "
        "different meetings. VP_Chen: Board Prep (critical). VP_Lee: Client Demo "
        "(high). VP_Kumar: Team Retro (normal). You must assess priority, keep the "
        "highest-priority meeting in Alpha, propose alternative rooms/times for the "
        "other two, and send diplomatic emails to all three VPs."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_100", title="Board Prep", date="2026-04-25", time="14:00", duration_min=60, participants=["VP_Chen"], location="Alpha", priority="critical"),
        CalendarEvent(event_id="evt_101", title="Client Demo", date="2026-04-25", time="14:00", duration_min=60, participants=["VP_Lee", "Client_Jones"], location="Alpha", priority="high"),
        CalendarEvent(event_id="evt_102", title="Team Retro", date="2026-04-25", time="14:00", duration_min=60, participants=["VP_Kumar", "Team"], location="Alpha", priority="normal"),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_100", sender="Admin", subject="Room conflict alert", body="Conference Room Alpha has 3 bookings at 2pm. Please resolve.", urgency="critical"),
    ],
    contacts=[
        Contact(name="VP_Chen", role="VP Engineering", priority_level=5),
        Contact(name="VP_Lee", role="VP Sales", priority_level=4),
        Contact(name="VP_Kumar", role="VP Product", priority_level=3),
    ],
    constraints=[
        ConstraintDef(description="Board Prep stays in Alpha at 14:00", check_type="event_exists", check_params={"event_id": "evt_100"}),
        ConstraintDef(description="No calendar conflicts after resolution", check_type="calendar_no_conflict", check_params={}),
        ConstraintDef(description="VP_Lee notified of room change", check_type="email_sent", check_params={"to": "VP_Lee", "keywords": ["room", "move", "demo"]}),
        ConstraintDef(description="VP_Kumar notified of room change", check_type="email_sent", check_params={"to": "VP_Kumar", "keywords": ["room", "move", "retro"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="VP_Lee", required_keywords=["room", "move", "alternative", "apologi"], purpose="renegotiate"),
        CommunicationReq(to="VP_Kumar", required_keywords=["room", "move", "alternative", "apologi"], purpose="renegotiate"),
    ],
    optimal_steps=6,
    max_steps=15,
)

_HARD_013 = ScenarioDef(
    scenario_id="hard_013",
    difficulty="hard",
    briefing=(
        "Triple crisis on 2026-04-25: (1) Your 4:00 PM flight (evt_110) was cancelled — "
        "you need to rebook before the 6:00 PM board prep (evt_111) tomorrow. "
        "(2) Board prep moved from 4:00 PM to 2:00 PM tomorrow (2026-04-26), "
        "conflicting with your lunch with Client_Jones (evt_112). "
        "(3) Your dinner reservation at Downtown Sushi was lost. "
        "Handle all three crises: rebook flight constraints, reschedule lunch "
        "with Client_Jones, find a new dinner restaurant, email all affected parties."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_110", title="Flight to NYC", date="2026-04-25", time="16:00", duration_min=180, priority="high"),
        CalendarEvent(event_id="evt_111", title="Board Prep", date="2026-04-26", time="16:00", duration_min=120, participants=["VP_Chen"], priority="critical"),
        CalendarEvent(event_id="evt_112", title="Lunch with Client_Jones", date="2026-04-26", time="12:00", duration_min=90, participants=["Client_Jones"], priority="high"),
        CalendarEvent(event_id="evt_113", title="Morning Standup", date="2026-04-26", time="09:00", duration_min=30, participants=["Team"]),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_110", sender="Airline", subject="Flight cancelled", body="Your flight at 4:00 PM today has been cancelled. Next available flight: 6:00 PM or 8:00 PM.", urgency="critical"),
        InboxEmail(email_id="em_111", sender="VP_Chen", subject="Board prep moved up", body="Board prep is now at 2pm tomorrow instead of 4pm. Non-negotiable.", urgency="critical"),
        InboxEmail(email_id="em_112", sender="Downtown Sushi", subject="Reservation cancelled", body="We regret to inform you that we had to cancel your reservation due to a private event.", urgency="high"),
    ],
    available_restaurants=[
        Restaurant(name="Sakura Garden", cuisine="Japanese", price_per_person=40, distance_miles=2.0, dietary_options=["vegetarian", "vegan"], capacity=25),
        Restaurant(name="Pizza Corner", cuisine="Italian", price_per_person=25, distance_miles=1.0, dietary_options=["vegetarian"], capacity=30),
    ],
    contacts=[
        Contact(name="VP_Chen", role="VP Engineering", priority_level=5),
        Contact(name="Client_Jones", role="Client", priority_level=4, availability={"2026-04-26": ["09:30", "10:00", "11:00"]}),
    ],
    constraints=[
        ConstraintDef(description="Board Prep rescheduled to 14:00", check_type="calendar_no_conflict", check_params={}),
        ConstraintDef(description="Client_Jones notified of lunch reschedule", check_type="email_sent", check_params={"to": "Client_Jones", "keywords": ["lunch", "reschedule", "move"]}),
        ConstraintDef(description="New dinner restaurant booked", check_type="restaurant_match", check_params={"criteria": {}}),
        ConstraintDef(description="VP_Chen acknowledged board prep change", check_type="email_sent", check_params={"to": "VP_Chen", "keywords": ["board", "prep", "2pm", "confirmed"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="Client_Jones", required_keywords=["lunch", "reschedule", "sorry", "alternative"], purpose="renegotiate"),
        CommunicationReq(to="VP_Chen", required_keywords=["board", "prep", "confirmed"], purpose="acknowledge"),
    ],
    optimal_steps=8,
    max_steps=15,
)

_HARD_014 = ScenarioDef(
    scenario_id="hard_014",
    difficulty="hard",
    briefing=(
        "VP_Chen asks you to schedule a meeting with Client_Jones 'sometime this week' "
        "(2026-04-21 to 2026-04-25). Client_Jones privately told you they're unavailable "
        "Mon-Wed due to a family emergency — this is confidential. VP_Chen doesn't know. "
        "You must propose Thu/Fri slots without revealing Client_Jones's private reason. "
        "Navigate the information asymmetry diplomatically."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_120", title="Team Sync", date="2026-04-24", time="10:00", duration_min=60, participants=["Team"]),
        CalendarEvent(event_id="evt_121", title="1-on-1 with VP_Chen", date="2026-04-25", time="14:00", duration_min=30, participants=["VP_Chen"]),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_120", sender="VP_Chen", subject="Meeting with Jones", body="Can you set up a meeting with Client_Jones this week? 30 minutes. Any day works for me.", urgency="high"),
        InboxEmail(email_id="em_121", sender="Client_Jones", subject="Availability - confidential", body="I'm dealing with a family emergency Mon-Wed. I'd prefer to keep this private. I'm free Thu after 2pm and all day Friday.", urgency="normal", context_hint="CONFIDENTIAL: do not share reason with VP_Chen"),
    ],
    contacts=[
        Contact(name="VP_Chen", role="VP Engineering", priority_level=5, availability={"2026-04-24": ["09:00", "10:00", "14:00", "15:00"], "2026-04-25": ["09:00", "10:00", "15:00", "16:00"]}),
        Contact(name="Client_Jones", role="Client", priority_level=4, availability={"2026-04-24": ["14:00", "15:00", "16:00"], "2026-04-25": ["09:00", "10:00", "11:00", "14:00", "15:00"]}),
    ],
    constraints=[
        ConstraintDef(description="Meeting scheduled Thu or Fri only", check_type="calendar_no_conflict", check_params={}),
        ConstraintDef(description="VP_Chen notified of proposed time", check_type="email_sent", check_params={"to": "VP_Chen", "keywords": ["Thursday", "Friday", "Client_Jones", "slot"]}),
        ConstraintDef(description="Client_Jones notified", check_type="email_sent", check_params={"to": "Client_Jones", "keywords": ["meeting", "VP", "time"]}),
    ],
    communication_requirements=[
        CommunicationReq(to="VP_Chen", required_keywords=["Thursday", "Friday", "Client_Jones", "available"], purpose="propose_slots"),
        CommunicationReq(to="Client_Jones", required_keywords=["meeting", "time", "VP_Chen"], purpose="propose_slots"),
    ],
    optimal_steps=5,
    max_steps=12,
)

_HARD_015 = ScenarioDef(
    scenario_id="hard_015",
    difficulty="hard",
    briefing=(
        "PRODUCTION INCIDENT: At 11:45 AM on 2026-04-25, PagerDuty fires — "
        "payment-service is returning 503s with 94%% error rate. HikariPool connection "
        "pool exhausted. You're the on-call engineer.\n\n"
        "Your existing commitments today:\n"
        "- 12:00 PM: Team lunch at Garden Bistro (you organised, 6 people attending)\n"
        "- 2:00 PM: Client demo with Client_Jones (promised last week)\n"
        "- 3:30 PM: 1-on-1 with VP_Chen\n"
        "- 6:00 PM: Personal dinner reservation\n\n"
        "You must triage the incident (acknowledge, page backup), handle your "
        "commitments (which ones to keep, which to reschedule), and properly "
        "notify everyone affected. The incident is highest priority."
    ),
    initial_calendar=[
        CalendarEvent(event_id="evt_130", title="Team Lunch", date="2026-04-25", time="12:00", duration_min=90, participants=["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"], location="Garden Bistro", priority="normal"),
        CalendarEvent(event_id="evt_131", title="Client Demo", date="2026-04-25", time="14:00", duration_min=60, participants=["Client_Jones"], priority="high"),
        CalendarEvent(event_id="evt_132", title="1-on-1 with VP_Chen", date="2026-04-25", time="15:30", duration_min=30, participants=["VP_Chen"], priority="high"),
        CalendarEvent(event_id="evt_133", title="Dinner", date="2026-04-25", time="18:00", duration_min=120, priority="low", is_personal=True),
    ],
    initial_inbox=[
        InboxEmail(email_id="em_130", sender="PagerDuty", subject="[CRITICAL] payment-service 503 — 94% error rate", body="payment-service ERROR HikariPool-1 Connection not available, timed out after 30000ms. Active: 10, Idle: 0, Waiting: 47. Circuit breaker OPEN.", urgency="critical"),
    ],
    contacts=[
        Contact(name="VP_Chen", role="VP Engineering", priority_level=5),
        Contact(name="Client_Jones", role="Client", priority_level=4),
        Contact(name="Team", role="Engineering Team", priority_level=2),
        Contact(name="Alice", role="Engineer (Backup On-Call)", priority_level=3),
    ],
    constraints=[
        ConstraintDef(description="Incident acknowledged via email", check_type="email_sent", check_params={"to": "Team", "keywords": ["incident", "payment", "503"]}),
        ConstraintDef(description="Team lunch cancelled or rescheduled", check_type="event_cancelled", check_params={"event_id": "evt_130"}),
        ConstraintDef(description="Client_Jones notified of demo reschedule", check_type="email_sent", check_params={"to": "Client_Jones", "keywords": ["reschedule", "demo", "apologi"]}),
        ConstraintDef(description="VP_Chen informed of incident", check_type="email_sent", check_params={"to": "VP_Chen", "keywords": ["incident", "payment", "on-call"]}),
        ConstraintDef(description="No unresolved calendar conflicts", check_type="calendar_no_conflict", check_params={}),
    ],
    communication_requirements=[
        CommunicationReq(to="Team", required_keywords=["incident", "payment", "cancel", "lunch"], purpose="notify_reschedule"),
        CommunicationReq(to="Client_Jones", required_keywords=["reschedule", "demo", "sorry", "apologi", "production"], purpose="renegotiate"),
        CommunicationReq(to="VP_Chen", required_keywords=["incident", "payment", "1-on-1", "reschedule"], purpose="renegotiate"),
    ],
    optimal_steps=8,
    max_steps=15,
    expected_cancelled_events=["evt_130"],
)


# ===================================================================
# Registry helpers
# ===================================================================

_ALL_SCENARIOS: Dict[str, ScenarioDef] = {
    s.scenario_id: s
    for s in [
        _EASY_001, _EASY_002, _EASY_003, _EASY_004, _EASY_005,
        _MED_006, _MED_007, _MED_008, _MED_009, _MED_010,
        _HARD_011, _HARD_012, _HARD_013, _HARD_014, _HARD_015,
    ]
}


def get_all_scenarios() -> Dict[str, ScenarioDef]:
    return _ALL_SCENARIOS


def get_scenario(scenario_id: str) -> Optional[ScenarioDef]:
    return _ALL_SCENARIOS.get(scenario_id)


def get_scenarios_by_difficulty(difficulty: str) -> List[ScenarioDef]:
    return [s for s in _ALL_SCENARIOS.values() if s.difficulty == difficulty]


def get_scenario_ids_grouped() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {"easy": [], "medium": [], "hard": []}
    for s in _ALL_SCENARIOS.values():
        grouped.setdefault(s.difficulty, []).append(s.scenario_id)
    return grouped
