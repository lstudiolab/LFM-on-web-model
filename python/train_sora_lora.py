#!/usr/bin/env python3
"""QLoRA fine-tuning for Sora/LFM using a pretrained instruction model.

Accepts JSONL records in one of these forms:
  {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
  {"prompt":"...", "response":"..."}
  {"instruction":"...", "output":"...", "input":"optional context"}

The script is intended for a GPU notebook (e.g. Kaggle). It saves adapter-only
checkpoints and trainer state to an output directory; it never overwrites the
base model. Start with --max-steps 20 to smoke-test before a longer run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default="data/sora_chat.jsonl", help="Training JSONL file")
    p.add_argument("--output", default="runs/sora-qwen-lora")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="Optional maximum number of records; 0 means all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-4bit", action="store_true", help="Use bf16/fp16 base weights instead of 4-bit")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def record_to_text(row: dict[str, Any], tokenizer: Any) -> str | None:
    messages = row.get("messages")
    if isinstance(messages, list):
        clean = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            text = msg.get("content", msg.get("text", ""))
            if role in {"system", "user", "assistant"} and isinstance(text, str) and text.strip():
                clean.append({"role": role, "content": text.strip()})
        if not any(m["role"] == "assistant" for m in clean):
            return None
        return tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)

    prompt = row.get("prompt") or row.get("instruction") or row.get("question")
    answer = row.get("response") or row.get("output") or row.get("answer")
    context = row.get("input")
    if isinstance(prompt, str) and isinstance(answer, str) and prompt.strip() and answer.strip():
        user_text = prompt.strip()
        if isinstance(context, str) and context.strip():
            user_text += "\n\nContext:\n" + context.strip()
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text},
             {"role": "assistant", "content": answer.strip()}],
            tokenize=False,
            add_generation_prompt=False,
        )
    return None


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required. Open a GPU notebook runtime; CPU-only training is not supported.")
    if args.max_steps < 1 or args.max_length < 64:
        raise SystemExit("--max-steps must be positive and --max-length must be at least 64.")
    set_seed(args.seed)
    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Training file not found: {data_path}. Provide --data with a JSONL chat dataset.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON on line {line_no}: {e}")
                continue
            if isinstance(row, dict):
                text = record_to_text(row, tokenizer)
                if text:
                    rows.append({"text": text})
            if args.limit and len(rows) >= args.limit:
                break
    if len(rows) < 2:
        raise SystemExit(f"Only {len(rows)} usable examples found in {data_path}; need at least 2.")
    print(f"Loaded {len(rows):,} formatted examples from {data_path}")

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["text"], truncation=True, max_length=args.max_length,
            padding=False, add_special_tokens=False,
        )
        encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
        return encoded

    dataset = Dataset.from_list(rows).map(
        tokenize_batch, batched=True, remove_columns=["text"], desc="Tokenizing"
    )
    # Dynamic padding; labels on padding positions are ignored.
    from dataclasses import dataclass

    @dataclass
    class CausalCollator:
        tokenizer: Any
        def __call__(self, features):
            batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            batch["labels"][batch["attention_mask"] == 0] = -100
            return batch

    use_4bit = not args.no_4bit
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = None
    if use_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=compute_dtype if not use_4bit else None,
        device_map="auto",
        trust_remote_code=False,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    from peft import get_peft_model
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        save_safetensors=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalCollator(tokenizer),
    )
    resume_from = True if args.resume else None
    trainer.train(resume_from_checkpoint=resume_from)
    final_dir = output / "adapter-final"
    trainer.model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    print(f"Finished. LoRA adapter saved to: {final_dir}")
    print("Keep the base model ID with this adapter; the adapter alone is not a complete model.")


if __name__ == "__main__":
    main()
