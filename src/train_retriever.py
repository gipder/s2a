#!/usr/bin/env python3
"""
Contrastive dual-encoder LoRA fine-tuning for the Audio2Tool retriever,
trained from scratch (base Qwen3-0.6B, no STOP warm-start) on the synthetic
(utterance, tool) corpus from generate_synthetic_utterances.py +
filter_synthetic_utterances.py.

Recipe ported from noise_aware_slu/retriever/src/train_retriever.py:
symmetric in-batch-negative InfoNCE (CLIP/SimCSE style) on a single
LoRA-adapted encoder shared between utterance (X) and tool target_text (Y),
PromptEOL-wrapped + last-token pooling (embedding_utils.py). Deliberately
simplified relative to the STOP version: no DDP, no depth/repel auxiliary
losses, no mid-training checkpoint/resume -- this corpus is 152 tools x ~19
utterances (2,911 pairs total), small enough that none of that machinery
is needed for a first pass.

Batching: each step samples --batch-size DISTINCT tools (never two
utterances of the same tool in one batch -- otherwise in-batch-negative
InfoNCE would wrongly push them apart despite sharing identical target
text), one random utterance per sampled tool. --epochs full passes are
approximated by ceil(n_utterances / batch_size) steps per epoch, matching
the STOP script's "one utterance per distinct label per batch, repeated
over steps" approach at this corpus's scale.

Usage:
    python src/train_retriever.py --epochs 1 --output experiment/retriever_train/scratch_1ep
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from embedding_utils import encode_texts

BASE_DIR = Path(__file__).resolve().parent.parent


def load_corpus(path: Path) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    corpus = [
        {
            "tool_name": t["tool_name"],
            "target_text": f"{t['signature']}: {t['description']}",
            "utterances": t["utterances"],
        }
        for t in raw["tools"] if t["utterances"]
    ]
    dropped = len(raw["tools"]) - len(corpus)
    if dropped:
        print(f"NOTE: {dropped} tool(s) had zero surviving utterances after filtering, excluded from training")
    return corpus


def sample_batch(corpus: list[dict], batch_size: int, rng: random.Random) -> tuple[list[str], list[str]]:
    tools = rng.sample(corpus, min(batch_size, len(corpus)))
    xs = [rng.choice(t["utterances"]) for t in tools]
    ys = [t["target_text"] for t in tools]
    return xs, ys


def info_nce_loss(x_emb: torch.Tensor, y_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = x_emb @ y_emb.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="FEATURE_EXTRACTION", bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.train()

    corpus = load_corpus(Path(args.corpus))
    n_utterances = sum(len(t["utterances"]) for t in corpus)
    steps_per_epoch = max(1, -(-n_utterances // args.batch_size))  # ceil
    total_steps = steps_per_epoch * args.epochs
    print(
        f"corpus: {len(corpus)} tools, {n_utterances} utterances -> "
        f"{steps_per_epoch} steps/epoch x {args.epochs} epoch(s) = {total_steps} steps "
        f"(batch_size={args.batch_size})"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)

    for step in range(total_steps):
        xs, ys = sample_batch(corpus, args.batch_size, rng)
        x_emb = encode_texts(xs, tokenizer, model, len(xs), args.max_length, no_grad=False)
        y_emb = encode_texts(ys, tokenizer, model, len(ys), args.max_length, no_grad=False)
        loss = info_nce_loss(x_emb, y_emb, args.temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % args.log_every == 0 or step == total_steps - 1:
            print(f"step {step + 1}/{total_steps}  loss={loss.item():.4f}")

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"Saved LoRA adapter -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/Qwen3-0.6B")
    p.add_argument("--corpus", default="experiment/synthetic_utterances/raw_Qwen3-32B_k20_filtered.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--output", default="experiment/retriever_train/scratch_1ep")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
