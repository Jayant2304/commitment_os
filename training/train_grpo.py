"""GRPO training script for CommitmentOS.

Uses TRL's GRPOTrainer with LoRA to train Qwen2.5-1.5B-Instruct on
temporal commitment coherence tasks.

Designed for Google Colab A100 or similar GPU environments.

Usage:
  python training/train_grpo.py [--model MODEL] [--epochs N] [--lr LR]

Environment variables:
  HF_TOKEN — HuggingFace token for model upload (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for CommitmentOS")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--output_dir", default="./training_output", help="Output directory")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", default="jayant2304/commitmentos-qwen-grpo", help="HF Hub model ID")
    parser.add_argument("--num_scenarios", type=int, default=15, help="Number of scenarios to use")
    parser.add_argument("--max_turns", type=int, default=8, help="Max turns per episode")
    parser.add_argument("--group_size", type=int, default=4, help="GRPO group size (completions per prompt)")
    return parser.parse_args()


def build_dataset(num_scenarios: int = 15) -> List[Dict[str, Any]]:
    """Build training dataset from CommitmentOS scenarios."""
    from server.tasks import get_all_scenarios
    from training.env_factory import build_initial_prompt, build_system_prompt

    scenarios = list(get_all_scenarios().values())[:num_scenarios]
    system_prompt = build_system_prompt()
    dataset: List[Dict[str, Any]] = []

    for scenario in scenarios:
        user_msg = build_initial_prompt(scenario)
        dataset.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "scenario_id": scenario.scenario_id,
            "difficulty": scenario.difficulty,
        })

    return dataset


def reward_function(completions: List[Any], **kwargs: Any) -> List[float]:
    """Reward function for GRPO — evaluates completions against CommitmentOS."""
    from training.env_factory import CommitmentOSEnvFactory

    def _completion_to_text(completion: Any) -> str:
        """Normalize TRL completion payloads across versions.

        Depending on TRL/Transformers version, completions can arrive as
        strings, dicts, or nested lists of chat/message objects.
        """
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            content = completion.get("content", completion.get("text", ""))
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)
            return str(content)
        if isinstance(completion, list):
            parts: List[str] = []
            for item in completion:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    content = item.get("content", item.get("text", ""))
                    if isinstance(content, list):
                        content = " ".join(
                            block.get("text", str(block)) if isinstance(block, dict) else str(block)
                            for block in content
                        )
                    parts.append(str(content))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(completion)

    factory = CommitmentOSEnvFactory(max_turns=8)
    normalized = [_completion_to_text(completion) for completion in completions]
    return factory(normalized)


def main() -> None:
    args = parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"Missing training dependency: {e}")
        print("Install with: pip install trl transformers peft datasets torch")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    print("Building dataset...")
    raw_data = build_dataset(args.num_scenarios)
    dataset = Dataset.from_list(raw_data)

    training_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=50,
        bf16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_completion_length=512,
        num_generations=args.group_size,
        report_to="none",
    )

    print("Initialising GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        hf_token = os.getenv("HF_TOKEN", "")
        if hf_token:
            print(f"Pushing to hub: {args.hub_model_id}")
            trainer.push_to_hub(args.hub_model_id, token=hf_token)
        else:
            print("HF_TOKEN not set — skipping hub push")

    print("Training complete!")

    save_training_metrics(trainer, args.output_dir)


def save_training_metrics(trainer: Any, output_dir: str) -> None:
    """Save training metrics to JSON for plotting training curves."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history = trainer.state.log_history if hasattr(trainer.state, "log_history") else []
    metrics_file = output_path / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
