# True LLM Learning Evaluation (Pre-RL vs Post-RL)

This folder is for checkpoint-vs-checkpoint evidence:

- pre-RL base model
- post-RL trained checkpoint

Both are evaluated with an identical protocol.

## Required environment variables

- `API_BASE_URL`
- `HF_TOKEN` (or `OPENAI_API_KEY`)
- `ENV_BASE_URL`
- `BASELINE_MODEL_NAME`
- `TRAINED_MODEL_NAME`

Optional protocol overrides:

- `EVAL_SEED` (default: `42`)
- `EVAL_MAX_STEPS` (default: `12`)
- `EVAL_TEMPERATURE` (default: `0.0`)
- `EVAL_TOP_P` (default: `1.0`)
- `EVAL_MAX_NEW_TOKENS` (default: `256`)
- `EVAL_SUCCESS_THRESHOLD` (default: `0.6`)

## Run

```bash
cd commitment_os
python3 evaluation/evaluate_llm_checkpoints.py
python3 evaluation/plot_llm_checkpoints.py
```

## Expected outputs

- `llm_eval_protocol.json`
- `baseline_llm_eval.json`
- `trained_llm_eval.json`
- `llm_comparison.csv`
- `llm_summary.json`
- `llm_case_study_hard_015.md`
- `llm_reward_by_task.svg`
- `llm_violations_before_after.svg`
