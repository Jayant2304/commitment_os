# Improvement Evaluation Artifacts

This folder contains deterministic baseline-vs-trained-style evaluation outputs for all 15 CommitmentOS tasks.

This is **not** the same as the real LLM checkpoint comparison; see root **README** section **B) True LLM Learning Eval** and `artifacts/evals_llm/`.

## Files

- `eval_protocol.json`: fixed protocol (task set, seed, max steps, decode config)
- `baseline_eval.json`: per-task baseline rollouts
- `trained_eval.json`: per-task improved/trained-style rollouts (same protocol)
- `improved_eval.json`: alias of trained outputs for backward compatibility
- `comparison.csv`: task-by-task delta table
- `summary.json`: aggregate metrics (mean/median deltas, difficulty splits, steps, success)
- `case_study_hard_011.md`: concise before/after narrative for one hard scenario
- `reward_by_task.svg`: visual comparison of final reward by task
- `violations_before_after.svg`: visual comparison of commitment violations

## Reproduce

```bash
cd commitment_os
python3 evaluation/evaluate_improvement.py
python3 evaluation/plot_improvement.py
```
