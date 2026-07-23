#!/usr/bin/env python3
"""
Audio2Tool Tier-1 (Direct) reproduction -- oracle text pipeline.

Reproduces the paper's "Qwen 8B" row in Table 3 (Whisper replaced by the
ground-truth query text, isolating tool-selection ability from ASR error).
Target: Tier-1 Acc = EM = 85.6%.

Text-only: feeds the ground-truth `query` string (not audio) to the LLM served
via vLLM's OpenAI-compatible API, alongside the full 152-tool taxonomy.

Usage:
    # Start the server separately, or use script/run_with_vllm.sh which wraps this.
    #   CUDA_VISIBLE_DEVICES=0 vllm serve model/Qwen3-8B --port 8000 --max-model-len 8192
    python src/tier1_oracle.py --model model/Qwen3-8B

    # Quick pilot on a subsample before the full 2,146-query run:
    python src/tier1_oracle.py --model model/Qwen3-8B --n_queries 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

from action_metrics import extract_action_call, extract_tool_name
from tools_registry import format_tools_by_domain, format_tools_by_domain_category, load_tools

TOOL_FORMATS = {
    "domain": format_tools_by_domain,
    "domain_category": format_tools_by_domain_category,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data/Audio2Tool_mod/public/tier1_direct_data/tier1_direct.json"
TOOLS_CSV = BASE_DIR / "data/Audio2Tool_mod/tools_registry.csv"


def load_unique_queries(data_path: Path) -> list[dict]:
    """tier1_direct.json has 2 rows per query (2 speaker renditions each) --
    oracle mode is text-only, so dedupe to one row per query_idx."""
    with open(data_path) as f:
        rows = json.load(f)
    seen: dict[int, dict] = {}
    for row in rows:
        seen.setdefault(row["query_idx"], row)
    return [seen[k] for k in sorted(seen)]


def sample_topk_candidates(all_tools: list[dict], gt_tool_name: str, k: int, rng: random.Random) -> list[dict]:
    """GT tool + (k-1) random distractors from the SAME domain -- matches the
    convention used in reasoning_for_asr's Tier4 top-k experiments
    (docs/audio2tool_tier4_experiments.md: "GT 포함 + 같은 도메인 랜덤 (k-1)개").
    Not a real retriever (no ranking signal) -- this is the retriever-shaped
    upper bound: "if a retriever nailed top-k recall, how well would the LLM
    do downstream", same role it played in the Tier4 ablations.

    Deliberately does NOT take the item's own `domain` field as an input: for
    64/2146 Tier-1 queries (setLockState/controlPlayback/getLockState/setVolume)
    that field disagrees with the domain tools_registry.csv actually registers
    the gold tool_name under -- e.g. item domain=smart_home, tool_name=
    setLockState, but setLockState is registered as smart_car (the
    smart_home-specific tool is the separate setLockState_home entry). A
    likely dataset labeling bug, not a model error -- see README. Looking the
    GT tool up by name across the full registry instead avoids crashing on
    those rows, and picking distractors from the tool's OWN registered domain
    keeps the candidate set internally coherent regardless."""
    by_name = {t["tool_name"]: t for t in all_tools}
    gt_tool = by_name[gt_tool_name]
    real_domain = gt_tool["domain"]
    pool = [t for t in all_tools if t["domain"] == real_domain and t["tool_name"] != gt_tool_name]
    distractors = rng.sample(pool, min(k - 1, len(pool)))
    candidates = [gt_tool] + distractors
    rng.shuffle(candidates)  # don't leak the answer via list position
    return candidates


def build_user_prompt(query: str, tools_str: str, n_tools: int) -> str:
    return (
        f"Available tools ({n_tools} total), grouped by domain:\n{tools_str}\n\n"
        f'User utterance: "{query}"\n\n'
        f"Respond with a single tool call in the form tool_name(arg=\"value\", ...), "
        f"or tool_name() if it takes no arguments.\n\n"
        f"Tool call:"
    )


def shorten_prompt(prompt: str) -> str:
    """Replace the tool-listing block with a pointer to tools_registry.csv.
    Only used for the full-152-tool case (--topk unset): logging the same
    ~27k-char tool block in every one of 2,146 sample records would bloat the
    output JSON for no benefit, since it's identical across samples and fully
    reconstructible from tools_registry.csv + --tool_format. --topk runs keep
    the full prompt since it's already short (just the k sampled candidates,
    which DO vary per sample)."""
    return re.sub(
        r"(Available tools \(\d+ total\), grouped by domain:\n).*?(?=\n\nUser utterance:)",
        r"\1<all 152 tools from data/Audio2Tool_mod/tools_registry.csv -- "
        r"formatting per this run's tool_format, see out_data['tool_format']>",
        prompt,
        flags=re.DOTALL,
    )


async def query_one(
    client: AsyncOpenAI,
    model: str,
    sample: dict,
    system_prompt: str,
    user_prompt: str,
    enable_thinking: bool,
    max_tokens: int,
    store_full_prompt: bool,
    idx: int,
    total: int,
) -> dict:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    output = response.choices[0].message.content or ""
    output = output.strip()

    pred_call = extract_action_call(output)
    pred_tool = extract_tool_name(pred_call) if pred_call else None
    ref_tool = extract_tool_name(sample["expected_tool_call"])
    correct = bool(pred_tool and ref_tool and pred_tool.upper() == ref_tool.upper())

    log.info(
        "[%d/%d] [%s] expected=%s | predicted=%s",
        idx + 1, total, "O" if correct else "X",
        sample["tool_name"], pred_tool or "<none>",
    )
    return {
        "query_idx": sample["query_idx"],
        "query": sample["query"],
        "domain": sample["domain"],
        "expected_tool_call": sample["expected_tool_call"],
        "prompt": user_prompt if store_full_prompt else shorten_prompt(user_prompt),
        "raw_output": output,
        "predicted_call": pred_call,
        "predicted_tool": pred_tool,
        "correct": correct,
    }


async def run_async(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    tools = load_tools(TOOLS_CSV)

    queries = load_unique_queries(DATA_PATH)
    if args.n_queries is not None:
        queries = queries[:: max(1, len(queries) // args.n_queries)][: args.n_queries]

    system_prompt = queries[0]["instruction"]
    max_tokens = 1024 if args.enable_thinking else 64

    log.info(
        "Model: %s | Queries: %d | Tools: %s | Thinking: %s | Concurrency: %d",
        args.model, len(queries), (f"top-{args.topk}" if args.topk else len(tools)),
        args.enable_thinking, args.concurrency,
    )

    if args.topk is not None:
        # Retriever-shaped upper bound: GT + (k-1) same-domain random distractors,
        # instead of the full 152-tool taxonomy. Not a real retriever -- see
        # sample_topk_candidates docstring.
        rng = random.Random(args.seed)
        prompts = []
        for q in queries:
            candidates = sample_topk_candidates(tools, q["tool_name"], args.topk, rng)
            prompts.append(build_user_prompt(q["query"], TOOL_FORMATS[args.tool_format](candidates), len(candidates)))
    else:
        tools_str = TOOL_FORMATS[args.tool_format](tools)
        prompts = [build_user_prompt(q["query"], tools_str, len(tools)) for q in queries]

    semaphore = asyncio.Semaphore(args.concurrency)
    store_full_prompt = args.topk is not None

    async def bounded_query(i: int) -> dict:
        async with semaphore:
            return await query_one(
                client, args.model, queries[i], system_prompt, prompts[i],
                args.enable_thinking, max_tokens, store_full_prompt, i, len(queries),
            )

    t0 = time.perf_counter()
    results = await asyncio.gather(*[bounded_query(i) for i in range(len(queries))])
    elapsed = time.perf_counter() - t0

    results = sorted(results, key=lambda r: r["query_idx"])
    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / len(results) * 100
    parse_failures = sum(1 for r in results if r["predicted_call"] is None)

    log.info(
        "Tier-1 oracle | Acc=EM: %d/%d = %.1f%%  (paper target: 85.6%%)  |  "
        "parse failures: %d  |  elapsed: %.1fs",
        correct_count, len(results), accuracy, parse_failures, elapsed,
    )

    model_name = Path(args.model).name
    out_data = {
        "model": args.model,
        "pipeline": "tier1_oracle",
        "enable_thinking": args.enable_thinking,
        "tool_format": args.tool_format,
        "topk": args.topk,
        "system_prompt": system_prompt,  # identical for every sample -- stored once here, not per-sample
        "paper_target_acc": 85.6,
        "summary": {
            "correct": correct_count,
            "total": len(results),
            "accuracy": accuracy,
            "parse_failures": parse_failures,
            "elapsed_sec": round(elapsed, 1),
        },
        "samples": results,
    }

    think_tag = "think" if args.enable_thinking else "nothink"
    topk_tag = f"top{args.topk}" if args.topk else "all152"
    out_path = (
        Path(args.output) if args.output
        else BASE_DIR / f"experiment/tier1_oracle/{model_name}_{args.tool_format}_{topk_tag}_{think_tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    log.info("Saved -> %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/Qwen3-8B",
                    help="vLLM model path (must match what the server was started with)")
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--n_queries", type=int, default=None,
                    help="Subsample for a quick pilot run (default: full 2,146 queries)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--enable_thinking", action="store_true",
                    help="Qwen3 thinking mode (off by default; sweep if accuracy doesn't match paper)")
    p.add_argument("--tool_format", default="domain", choices=list(TOOL_FORMATS),
                    help="Tool-list prompt format (default: flat per-domain list)")
    p.add_argument("--topk", type=int, default=None,
                    help="Retriever-shaped upper bound: GT + (k-1) same-domain random "
                         "distractors instead of the full 152-tool list (default: full list)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --topk distractor sampling")
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_async(parse_args()))
