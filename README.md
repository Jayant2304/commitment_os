# CommitmentOS: Training Temporal Commitment Coherence in LLMs

> *The first RL environment that trains LLMs to keep their promises.*

**Innovation claim**: The first RL environment for training temporal commitment coherence — where the agent's own prior decisions create binding future constraints, tracked and penalised across multi-turn episodes.

**Theme**: Primary 3.2 (Personal Tasks) + Secondary Theme 2 (Long-Horizon Planning)

## Links

- GitHub Repository: [https://github.com/Jayant2304/commitment_os](https://github.com/Jayant2304/commitment_os)
- Hugging Face Space (live environment): [https://huggingface.co/spaces/Jayant2304/commitment-os](https://huggingface.co/spaces/Jayant2304/commitment-os)
- Colab Training Notebook: `Added in final submission form`
- Demo Video / Blog / Slides: `Added in final submission form`

---

## Architecture

```
┌──────────────── Client ────────────────┐     ┌────────────── CommitmentOS Server ──────────────┐
│                                        │     │                                                 │
│  inference.py ──HTTP──▶ POST /reset    │────▶│  FastAPI App                                    │
│  (LLM agent)    HTTP──▶ POST /step     │     │    │                                            │
│                  HTTP──▶ GET  /state    │     │    ▼                                            │
│                                        │     │  CommitmentEnvironment                          │
│  train_grpo.py                         │     │    ├── WorldState (calendar, contacts,           │
│  (GRPO+TRL)                            │     │    │   restaurants, inbox)                       │
│                                        │     │    ├── CommitmentLedger (tracks promises)        │
│                                        │     │    └── Grader (5-component reward)               │
└────────────────────────────────────────┘     └─────────────────────────────────────────────────┘
```

## Why CommitmentOS is Novel

Existing constraint-satisfaction environments (GAP, LGC-MARL, NeMo Gym, PEARL) compute dependency graphs **upfront**. CommitmentOS is fundamentally different:

- **Constraints emerge from the agent's own decisions** as the episode unfolds
- A meeting scheduled in turn 2 becomes a **binding constraint** in turn 7
- Breaking it without communication is a **tracked, penalised violation**
- The commitment ledger persists across the full episode — the agent must remember what it promised

This is **temporal commitment coherence** — a capability no existing RL environment trains.

---

## Quick Start

### Local Development

```bash
cd commitment_os

# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Run tests
pip install pytest httpx
pytest tests/ -v
```

### Docker

```bash
docker build -t commitment-os .
docker run -p 7860:7860 commitment-os
```

### API Usage

```bash
# Reset to a scenario
curl -X POST "http://localhost:7860/reset?task_id=easy_001"

# Make a tool call (multi-turn — one per step)
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "view_calendar", "date": "2026-04-25"}}'

# Get state
curl "http://localhost:7860/state"

# List all scenarios
curl "http://localhost:7860/tasks"
```

---

## Reward Function (5 Components)

| Component | Weight | How it's Measured |
|-----------|--------|-------------------|
| **Constraint Satisfaction** | 35% | Binary per-constraint checks |
| **Conflict Resolution** | 20% | Final calendar free of overlapping events |
| **Commitment Coherence** | 20% | `(total - silent_violations) / total` from ledger |
| **Communication Quality** | 15% | Keyword matching on sent emails |
| **Step Efficiency** | 10% | `max(0, 1 - (steps - optimal) × 0.1)` |

**Example** (easy_001 — perfect run):
```
constraints: 3/3 met         → 0.35 × 1.0 = 0.350
conflicts:   0 overlaps      → 0.20 × 1.0 = 0.200
commitments: 1 honored       → 0.20 × 1.0 = 0.200
emails:      Team notified   → 0.15 × 1.0 = 0.150
efficiency:  3 steps (opt 3) → 0.10 × 1.0 = 0.100
─────────────────────────────────────────────
total = 0.99 (clamped to [0.01, 0.99])
```

---

## 15 Scenarios

### Easy (2-4 steps)
| ID | Description |
|----|-------------|
| easy_001 | Double-booked meetings — reschedule by priority |
| easy_002 | Book dinner with cuisine/price/distance constraints |
| easy_003 | Check availability and propose meeting slots |
| easy_004 | Cancel conflicting work meeting for personal appointment |
| easy_005 | Triage inbox by urgency priority |

### Medium (5-8 steps)
| ID | Description |
|----|-------------|
| med_006 | Cascading reschedule chain (A→B→C dependency) |
| med_007 | Team dinner with 3 dietary + distance + budget constraints |
| med_008 | Boss's urgent request during client call (commitment conflict) |
| med_009 | Disambiguate vague "push our thing" across 3 recurring meetings |
| med_010 | Client visit: conference room + lunch + itinerary |

### Hard (8-15 steps)
| ID | Description |
|----|-------------|
| hard_011 | VP investor dinner: cascade, restaurant, multi-party notification |
| hard_012 | Triple conference room conflict with diplomatic resolution |
| hard_013 | Triple crisis: cancelled flight + moved board prep + lost reservation |
| hard_014 | Information asymmetry — schedule without revealing confidential reasons |
| hard_015 | **SRE Crisis** — production incident interrupts day of commitments |

---

## Training

### GRPO + TRL + LoRA

```bash
pip install trl transformers peft datasets torch

python training/train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 2 \
  --lr 5e-6 \
  --lora_rank 16 \
  --batch_size 4
```

**What improves with training:**
- Constraint satisfaction score ↑
- Commitment violation rate ↓
- Steps per episode ↓
- Communication quality ↑

### Real Training Run (Colab)

The following metrics are from an actual GRPO run on `Qwen/Qwen2.5-1.5B-Instruct`:

- Runtime: **507.6 seconds** (~8.46 min)
- Steps: **30**
- Epochs: **2**
- Final train loss: **-0.02182**
- Reward range during training: **0.4021 -> 0.6896**
- Final reward: **0.5437**

Artifacts saved in `artifacts/`:
- `artifacts/loss_curve.png`
- `artifacts/reward_curve.png`
- `artifacts/training_summary.csv`
- `artifacts/training_metrics.json`

#### Loss Curve

![CommitmentOS GRPO Loss vs Step](artifacts/loss_curve.png)

#### Reward Curve

![CommitmentOS GRPO Reward vs Step](artifacts/reward_curve.png)

These curves show non-trivial reward improvement peaks (~0.69) and confirm
that the training loop runs end-to-end against the environment (not a static
dataset baseline).

---

## Submission Compliance

| Requirement | Status |
|-------------|--------|
| reset() / step() / state() | ✅ |
| openenv.yaml with 15 tasks | ✅ |
| Programmatic graders, scores ∈ (0, 1) | ✅ |
| inference.py at root using openai client | ✅ |
| [START]/[STEP]/[END] log format | ✅ |
| API_BASE_URL / MODEL_NAME / HF_TOKEN from env | ✅ |
| Dockerfile builds and responds to /reset | ✅ |
| pyproject.toml with [project.scripts] | ✅ |
| uv.lock generated | ✅ |
| server/app.py main() with if __name__ | ✅ |

---

## Story Hook

> "Every AI assistant today can schedule one meeting. But your real life is never one meeting. CommitmentOS trains AI to juggle the chaos — and penalises it when it breaks its own promises."

**Connection to Round 1**: In Round 1, we trained agents to diagnose production incidents. In Round 2, we asked: *what happens when that incident interrupts a day full of commitments?* CommitmentOS was born. Hard scenario `hard_015` directly reuses SRE incident data from Round 1.

---

## License

MIT
