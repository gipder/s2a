#!/usr/bin/env python3
"""
Retriever baseline for Audio2Tool Tier-1 -- zero-shot, or a STOP-trained
LoRA adapter transferred cross-domain with zero Audio2Tool-specific training.

Audio2Tool ships no train split -- every tier is HF-config'd as `split: test`
only, so a from-scratch STOP-style contrastively-fine-tuned retriever needs
its own train/dev/test carve-out first (see README "다음 단계"). This script
instead asks two cheaper questions first: (1) how far does an OFF-THE-SHELF
embedding model get with zero training, and (2) does a retriever already
trained on STOP (a different dataset, different tool taxonomy) transfer at
all -- via --lora_adapter pointing at one of noise_aware_slu's checkpoints.
Same PromptEOL + last-token-pooling recipe noise_aware_slu/retriever
validated as a strong zero-shot baseline on STOP (domain_acc 48.3% -> 82.0%,
intent_acc 14.3% -> 49.3% with Qwen3-0.6B) either way -- the LoRA adapter is
just extra weights merged into the same encoder, sharing X (utterance) and Y
(target_text) as noise_aware_slu/retriever/src/train_retriever.py does.

Corpus: 152 tool texts (signature + description) from tools_registry.csv.
Queries: Tier-1's 2,146 unique utterances.
Metric: Recall@k -- is the gold tool_name within the top-k most similar
tools by cosine similarity (embeddings are L2-normalized, so cosine sim ==
dot product)? Directly comparable to tier1_oracle.py's --topk oracle ceiling
(GT + same-domain random distractors): if recall@5 here approaches that
ceiling's downstream accuracy (96.9%), a real retriever could plausibly
replace the oracle's guaranteed-GT-inclusion assumption.

Usage:
    python src/eval_zeroshot_retriever.py --model model/Qwen3-0.6B
    python src/eval_zeroshot_retriever.py --model model/Qwen3-0.6B --n_queries 200  # pilot

    # STOP-trained retriever, no Audio2Tool training at all:
    python src/eval_zeroshot_retriever.py --model model/Qwen3-0.6B \\
        --lora_adapter ../noise_aware_slu/experiments/retriever_train/qwen3-0.6b_depth0.1_seed44_5ep/epoch5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from embedding_utils import encode_texts
from tier1_oracle import DATA_PATH, load_unique_queries
from tools_registry import load_tools

BASE_DIR = Path(__file__).resolve().parent.parent
TOOLS_CSV = BASE_DIR / "data/Audio2Tool_mod/tools_registry.csv"


def main(args: argparse.Namespace) -> None:
    tools = load_tools(TOOLS_CSV)
    if not (1 <= args.save_topk <= len(tools)):
        raise ValueError(f"--save_topk must be in [1, {len(tools)}] (corpus size), got {args.save_topk}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    if args.lora_adapter:
        model = PeftModel.from_pretrained(model, args.lora_adapter)
    model.eval()

    corpus_texts = [f"{t['signature']}: {t['description']}" for t in tools]
    tool_names = [t["tool_name"] for t in tools]
    tool_domains = [t["domain"] for t in tools]

    queries = load_unique_queries(DATA_PATH)
    if args.n_queries is not None:
        queries = queries[:: max(1, len(queries) // args.n_queries)][: args.n_queries]

    print(f"Embedding {len(corpus_texts)} tool descriptions...")
    corpus_emb = encode_texts(corpus_texts, tokenizer, model, args.batch_size, args.max_length)

    print(f"Embedding {len(queries)} queries...")
    query_texts = [q["query"] for q in queries]
    query_emb = encode_texts(query_texts, tokenizer, model, args.batch_size, args.max_length)

    sims = query_emb @ corpus_emb.T  # [n_queries, n_tools], cosine sim (both L2-normalized)
    ranked_idx_all = sims.argsort(dim=-1, descending=True).tolist()

    topk_list = [1, 3, 5, 10]
    hits = {k: 0 for k in topk_list}
    domain_hits = {k: 0 for k in topk_list}
    results = []
    for i, q in enumerate(queries):
        gt_name, gt_domain = q["tool_name"], q["domain"]
        ranked_idx = ranked_idx_all[i]
        ranked_names = [tool_names[j] for j in ranked_idx]
        ranked_domains = [tool_domains[j] for j in ranked_idx]
        gold_rank = ranked_names.index(gt_name) + 1 if gt_name in ranked_names else None
        for k in topk_list:
            if gold_rank is not None and gold_rank <= k:
                hits[k] += 1
            if gt_domain in ranked_domains[:k]:
                domain_hits[k] += 1
        save_n = args.save_topk
        results.append({
            "query_idx": q["query_idx"],
            "query": q["query"],
            "domain": gt_domain,
            "gold_tool": gt_name,
            "gold_rank": gold_rank,
            "retrieved_tools": ranked_names[:save_n],
            "retrieved_scores": [round(sims[i, j].item(), 4) for j in ranked_idx[:save_n]],
        })

    n = len(queries)
    label = f"STOP-trained retriever (LoRA={args.lora_adapter})" if args.lora_adapter else "Zero-shot retriever (no training)"
    print(f"\n{label} | model={args.model} | n={n} queries, {len(tools)} tools")
    for k in topk_list:
        print(
            f"  recall@{k}: {hits[k]}/{n} = {hits[k] / n * 100:.1f}%   "
            f"(domain-only recall@{k}: {domain_hits[k] / n * 100:.1f}%)"
        )

    model_name = Path(args.model).name
    # Adapter name folded into the default filename so a STOP-trained-retriever
    # run doesn't collide with (or get mistaken for) the plain zero-shot run.
    if args.lora_adapter:
        adapter_path = Path(args.lora_adapter.rstrip("/"))
        tag = f"{model_name}_lora-{adapter_path.parent.name}-{adapter_path.name}"
    else:
        tag = model_name
    out_path = Path(args.output) if args.output else BASE_DIR / f"experiment/zeroshot_retriever/{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "lora_adapter": args.lora_adapter,
                "n_queries": n,
                "n_tools": len(tools),
                "recall_at_k": {k: hits[k] / n * 100 for k in topk_list},
                "domain_recall_at_k": {k: domain_hits[k] / n * 100 for k in topk_list},
                "samples": results,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/Qwen3-0.6B")
    p.add_argument("--lora_adapter", default=None,
                    help="Path to a trained LoRA adapter (e.g. one of noise_aware_slu's STOP-trained "
                         "retriever checkpoints) to test cross-domain transfer with zero Audio2Tool "
                         "training. Must match --model as its base_model_name_or_path. Default: none "
                         "(plain zero-shot base model)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--n_queries", type=int, default=None,
                    help="Subsample for a quick pilot run (default: full 2,146 queries)")
    p.add_argument("--save_topk", type=int, default=30,
                    help="How many ranked tools to save per query (for downstream real-retriever "
                         "pipeline tests, e.g. tier1_oracle.py --retrieved_from)")
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
