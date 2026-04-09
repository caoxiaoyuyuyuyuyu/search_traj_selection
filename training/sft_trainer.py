"""SFT training script based on TRL for search-reasoning distillation.

Trains student models (Qwen3-8B/4B) on selected search-reasoning trajectories
from the teacher (Qwen3-32B).
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def load_trajectories(path: str | Path, max_samples: int | None = None) -> list[dict]:
    """Load selected trajectories from JSONL."""
    trajectories = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            trajectories.append(json.loads(line.strip()))
            if max_samples and len(trajectories) >= max_samples:
                break
    logger.info("Loaded %d trajectories from %s", len(trajectories), path)
    return trajectories


def format_trajectory_for_sft(
    trajectory: dict,
    tokenizer: AutoTokenizer,
) -> str:
    """Format a trajectory into a chat-format string for SFT.

    The trajectory becomes a single-turn conversation:
    User: question → Assistant: search-reasoning trajectory
    """
    messages = [
        {"role": "user", "content": f"Answer the following question by searching for information step by step.\n\nQuestion: {trajectory['question']}"},
        {"role": "assistant", "content": trajectory["text"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_dataset(
    trajectories: list[dict],
    tokenizer: AutoTokenizer,
) -> Dataset:
    """Build HuggingFace Dataset from trajectories."""
    formatted = [
        {"text": format_trajectory_for_sft(t, tokenizer)}
        for t in trajectories
    ]
    return Dataset.from_list(formatted)


def train(
    model_name: str,
    train_path: str,
    output_dir: str,
    eval_path: str | None = None,
    max_seq_len: int = 4096,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = 42,
    bf16: bool = True,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
):
    """Run SFT training.

    Args:
        model_name: Student model name/path.
        train_path: Path to training trajectories JSONL.
        output_dir: Output directory for checkpoints.
        eval_path: Optional evaluation trajectories JSONL.
        max_seq_len: Maximum sequence length.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate.
        num_epochs: Number of training epochs.
        warmup_ratio: Warmup ratio.
        weight_decay: Weight decay.
        seed: Random seed.
        bf16: Use bfloat16.
        logging_steps: Log every N steps.
        save_strategy: Save strategy.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
    """
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Build datasets
    train_trajs = load_trajectories(train_path)
    train_dataset = build_dataset(train_trajs, tokenizer)

    eval_dataset = None
    if eval_path:
        eval_trajs = load_trajectories(eval_path)
        eval_dataset = build_dataset(eval_trajs, tokenizer)

    # Configure training
    report_to = "wandb" if wandb_project else "none"
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=bf16,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        eval_strategy="epoch" if eval_dataset else "no",
        seed=seed,
        max_seq_length=max_seq_len,
        report_to=report_to,
        run_name=wandb_run_name,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training: %d examples, %d epochs", len(train_dataset), num_epochs)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete. Model saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="SFT training for search-reasoning distillation")
    parser.add_argument("--model", type=str, required=True, help="Student model name")
    parser.add_argument("--train_data", type=str, required=True, help="Training trajectories JSONL")
    parser.add_argument("--eval_data", type=str, default=None, help="Eval trajectories JSONL")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    train(
        model_name=args.model,
        train_path=args.train_data,
        output_dir=args.output_dir,
        eval_path=args.eval_data,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
