---
title: CommitmentOS
emoji: 📋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - commitment-coherence
  - personal-task-management
  - multi-turn
---

# CommitmentOS: Training Temporal Commitment Coherence in LLMs

**The first RL environment that trains LLMs to keep their promises.**

CommitmentOS is a multi-turn personal task management environment where
agents manage calendars, emails, and dining reservations across realistic
scenarios. The key innovation: the agent's own prior decisions create
binding future constraints tracked via a **commitment ledger**, and
violations are penalised regardless of how many turns have elapsed.

## Quick Start

```bash
# Reset to a scenario
curl -X POST "https://jayant2304-commitment-os.hf.space/reset?task_id=easy_001"

# Make a tool call
curl -X POST "https://jayant2304-commitment-os.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "view_calendar", "date": "2026-04-25"}'

# Get state
curl "https://jayant2304-commitment-os.hf.space/state"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode (optional: `task_id`, `difficulty`) |
| `/step` | POST | Execute one tool call |
| `/state` | GET | Current episode state |
| `/health` | GET | Health check |
| `/tasks` | GET | List all available scenarios |
| `/mcp` | POST | MCP JSON-RPC 2.0 |

## 15 Scenarios (5 Easy / 5 Medium / 5 Hard)

Scenarios range from simple calendar reschedules to multi-crisis cascades
with information asymmetry and production incidents interrupting a full day
of commitments.

## Reward Function (5 components)

| Component | Weight | Signal |
|-----------|--------|--------|
| Constraint Satisfaction | 35% | Binary per-constraint checks |
| Conflict Resolution | 20% | Calendar free of overlaps |
| **Commitment Coherence** | **20%** | **Violations tracked via ledger** |
| Communication Quality | 15% | Keyword matching on emails |
| Step Efficiency | 10% | Fewer steps = higher score |

## What Makes This Novel

Existing constraint-satisfaction environments compute dependency graphs
upfront. CommitmentOS is different: constraints **emerge from the agent's
own decisions** as the episode unfolds. A meeting scheduled in turn 2
becomes a binding constraint in turn 7. Breaking it without communication
is a tracked, penalised violation.

This is **temporal commitment coherence** — a capability no existing RL
environment trains.
