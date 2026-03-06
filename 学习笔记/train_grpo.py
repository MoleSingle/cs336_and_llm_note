#!/usr/bin/env python3
"""
Minimal GRPO training script (TRL).

Usage:
  python train_grpo.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_path ./train.jsonl \
    --output_dir ./grpo_ckpt
"""

import argparse
import re
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Think in <think>...</think> and output final answer in <answer>...</answer>."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with GRPO (TRL).")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model checkpoint.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Optional local json/jsonl path. Must contain `question` and `answer` columns.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Optional HF dataset name. Ignored if --dataset_path is given.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="HF dataset split.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=512,
        help="Max training samples (for quick experiments).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./grpo_ckpt",
        help="Output checkpoint directory.",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    return parser.parse_args()


def build_toy_dataset() -> Dataset:
    data = {
        "question": [
            "What is 3 + 5?",
            "What is 12 - 7?",
            "What is 6 * 4?",
            "What is 9 + 8?",
        ],
        "answer": ["8", "5", "24", "17"],
    }
    return Dataset.from_dict(data)


def load_train_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset_path:
        ds = load_dataset("json", data_files=args.dataset_path, split="train")
    elif args.dataset_name:
        ds = load_dataset(args.dataset_name, split=args.dataset_split)
    else:
        ds = build_toy_dataset()

    if "question" not in ds.column_names or "answer" not in ds.column_names:
        raise ValueError(
            f"Dataset must include `question` and `answer` columns, got: {ds.column_names}"
        )

    ds = ds.select(range(min(len(ds), args.max_samples)))

    def _to_prompt(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(example["question"])},
            ],
            "answer": str(example["answer"]),
        }

    return ds.map(_to_prompt)


def completion_to_text(completion: Any) -> str:
    # TRL may return either plain strings or chat-format structures.
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(completion)


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def format_reward(completions, **kwargs):
    scores = []
    for c in completions:
        text = completion_to_text(c)
        ok = bool(
            re.search(
                r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
                text,
                flags=re.DOTALL,
            )
        )
        scores.append(1.0 if ok else 0.0)
    return scores


def accuracy_reward(completions, **kwargs):
    answers = kwargs.get("answer", None)
    if answers is None:
        return [0.0] * len(completions)

    scores = []
    for c, gold in zip(completions, answers):
        pred = extract_answer(completion_to_text(c))
        scores.append(1.0 if pred == str(gold).strip() else 0.0)
    return scores


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )

    train_dataset = load_train_dataset(args)

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, accuracy_reward],
        args=grpo_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
