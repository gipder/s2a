#!/usr/bin/env python3
"""
Audio2Tool Tier-1 (Direct) reproduction -- oracle text pipeline.

Reproduces the paper's "Qwen 8B" row in Table 3 (Whisper replaced by the
ground-truth query text, isolating tool-selection ability from ASR error).
Target: Tier-1 Acc = EM = 85.6%.

Text-only: feeds the ground-truth `query` string (not audio) to the LLM served
via vLLM's OpenAI-compatible API, alongside the full 152-tool taxonomy.

Grading policy is tier-specific, confirmed against the worked examples at
https://audio2tool.github.io/: Tier-1's ground truth is shown as a bare
"Tool: setZoneTemperature" (no argument notation), while Tier-2+ ground truth
shows "Tool: X: {args...}" -- arguments only enter grading from Tier-2 onward.
So here, EM is tool-name match only, same as Tool-Acc (see query_one) --
`expected_tool_call`'s trailing "()" is call-syntax formatting, not "must
match zero args". This does NOT generalize to later tiers: tier2_oracle.py
etc. grade EM with action_metrics.em() (full name+argument tree match)
instead of reusing this file's name-only logic.

Tool-candidate selection (--topk/--domain_filtered/--retrieved_from) and
prompt building are tier-agnostic and live in oracle_shared.py, shared with
tier2_oracle.py -- only the grading in query_one is Tier-1-specific here.

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
import time
from pathlib import Path

from openai import AsyncOpenAI

from action_metrics import ActionParseError, extract_action_call, extract_tool_name, parse_canonical_action
from oracle_shared import add_candidate_selection_args, build_candidate_prompts, load_unique_queries, shorten_prompt
from tools_registry import load_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data/Audio2Tool_mod/public/tier1_direct_data/tier1_direct.json"
TOOLS_CSV = BASE_DIR / "data/Audio2Tool_mod/tools_registry.csv"


def model_included_args(pred_call: str | None) -> bool:
    """Whether the model's predicted call carries any arguments at all --
    purely diagnostic (see query_one), not part of Tier-1 grading."""
    if not pred_call:
        return False
    try:
        pred_tree = parse_canonical_action(pred_call)
    except ActionParseError:
        return False
    return bool(pred_tree["slots"])


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
    # Tier-1's EM == Tool-Acc by the benchmark's own definition, confirmed
    # against https://audio2tool.github.io/'s worked examples: Tier-1 ground
    # truth is shown as bare "Tool: setZoneTemperature" (no argument notation
    # at all), while Tier-2+ ground truth shows "Tool: X: {args...}" -- i.e.
    # arguments only enter grading from Tier-2 onward. `expected_tool_call`'s
    # trailing "()" is just call-syntax formatting, not "must have zero args"
    # (this also explains why Table 3 has Acc==EM for every single model on
    # Tier-1: that identity is only possible if EM there ignores arguments).
    # Whether the model tacks on an argument anyway is tracked separately
    # below purely as a behavioral diagnostic, not an error.
    em_score = correct
    included_args = model_included_args(pred_call)

    log.info(
        "[%d/%d] [%s] expected=%s | predicted=%s%s",
        idx + 1, total, "O" if correct else "X",
        sample["tool_name"], pred_tool or "<none>",
        " (+unrequested args)" if correct and included_args else "",
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
        "em": em_score,
        "included_args": included_args,
    }


async def run_async(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    tools = load_tools(TOOLS_CSV)

    queries = load_unique_queries(DATA_PATH)
    if args.n_queries is not None:
        queries = queries[:: max(1, len(queries) // args.n_queries)][: args.n_queries]

    system_prompt = queries[0]["instruction"]
    max_tokens = 1024 if args.enable_thinking else 64

    prompts, store_full_prompt, tools_desc = build_candidate_prompts(args, tools, queries, TOOLS_CSV)
    log.info(
        "Model: %s | Queries: %d | Tools: %s | Thinking: %s | Concurrency: %d",
        args.model, len(queries), tools_desc, args.enable_thinking, args.concurrency,
    )

    semaphore = asyncio.Semaphore(args.concurrency)

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
    em_count = sum(r["em"] for r in results)  # == correct_count on Tier-1, see query_one
    em_accuracy = em_count / len(results) * 100
    parse_failures = sum(1 for r in results if r["predicted_call"] is None)
    # Purely behavioral diagnostic, not an error: how often the model attaches
    # an argument to a correct call even though the gold call has none.
    unrequested_args = sum(1 for r in results if r["correct"] and r["included_args"])

    log.info(
        "Tier-1 oracle | Acc=EM: %d/%d = %.1f%%  (paper target: 85.6%%)  |  "
        "correct calls with unrequested args: %d  |  parse failures: %d  |  elapsed: %.1fs",
        correct_count, len(results), accuracy,
        unrequested_args, parse_failures, elapsed,
    )

    model_name = Path(args.model).name
    out_data = {
        "model": args.model,
        "pipeline": "tier1_oracle",
        "enable_thinking": args.enable_thinking,
        "tool_format": args.tool_format,
        "topk": args.topk,
        "domain_filtered": args.domain_filtered,
        "retrieved_from": args.retrieved_from,
        "retrieved_topk": args.retrieved_topk if args.retrieved_from else None,
        "retrieved_domain_from": args.retrieved_domain_from,
        "system_prompt": system_prompt,  # identical for every sample -- stored once here, not per-sample
        "paper_target_acc": 85.6,
        "summary": {
            "correct": correct_count,
            "total": len(results),
            "accuracy": accuracy,
            "em_correct": em_count,
            "em_accuracy": em_accuracy,
            "unrequested_args": unrequested_args,
            "parse_failures": parse_failures,
            "elapsed_sec": round(elapsed, 1),
        },
        "samples": results,
    }

    think_tag = "think" if args.enable_thinking else "nothink"
    topk_tag = (
        f"top{args.topk}" if args.topk
        else "domainfiltered" if args.domain_filtered
        # Includes the source retriever's own filename stem -- otherwise two
        # different retrievers (e.g. Qwen3-0.6B.json vs Qwen3-1.7B.json) run
        # at the same --retrieved_topk would collide on the same output path
        # and silently overwrite each other, with no way to tell from the
        # filename which retriever produced a given result.
        else f"retrieved{args.retrieved_topk}-{Path(args.retrieved_from).stem}" if args.retrieved_from
        else f"reddomain-{Path(args.retrieved_domain_from).stem}" if args.retrieved_domain_from
        else "all152"
    )
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
    add_candidate_selection_args(p)
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_async(parse_args()))
