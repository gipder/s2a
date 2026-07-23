#!/usr/bin/env python3
"""
Audio2Tool Tier-2 (Parametric) reproduction -- oracle text pipeline.

Reproduces the paper's "Qwen 8B" row in Table 3 for Tier-2 (Acc=77.1%,
EM=10.1%, F1=19.3%). Unlike Tier-1, Tier-2's ground truth carries real
argument values (e.g. getAirQuality(deviceId='living_room_sensor')) and its
grading DOES cover arguments (confirmed against https://audio2tool.github.io/'s
worked examples: Tier-2 ground truth is shown as "Tool: X: {args...}", unlike
Tier-1's bare "Tool: X") -- so EM here is a real
action_metrics.parse_canonical_action + em() full name+argument-tree match,
not the name-only shortcut tier1_oracle.py correctly uses for Tier-1 only.

Candidate-selection modes (--topk / --domain_filtered / --retrieved_from) and
prompt building are tier-agnostic and live in oracle_shared.py, shared with
tier1_oracle.py -- only the grading here (score/query_one) is Tier-2-specific.

Usage:
    python src/tier2_oracle.py --model model/Qwen3-8B
    python src/tier2_oracle.py --model model/Qwen3-8B --n_queries 50  # pilot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from openai import AsyncOpenAI

from action_metrics import (
    ActionParseError,
    em,
    extract_action_call,
    extract_tool_name,
    parse_canonical_action,
    slot_f1,
)
from oracle_shared import add_candidate_selection_args, build_candidate_prompts, load_unique_queries, shorten_prompt
from tools_registry import load_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data/Audio2Tool_mod/public/tier2_parametric_data/tier2_parametric.json"
TOOLS_CSV = BASE_DIR / "data/Audio2Tool_mod/tools_registry.csv"


def score(pred_call: str | None, ref_tree: dict) -> tuple[bool, bool, float, bool]:
    """Returns (tool_acc, em, slot_f1, full_parse_ok) against an
    already-parsed gold tree (see run_async -- unlike Tier-1, 2/2,041 unique
    Tier-2 gold strings are themselves malformed dataset bugs, e.g. an
    unescaped apostrophe or a stray non-ASCII character mid-identifier, so
    parsing gold happens once up front with those rows dropped, not per-query
    here).

    Tool-Acc is graded via extract_tool_name's lenient regex extraction, NOT
    a full parse_canonical_action -- same reasoning as tier1_oracle.py/that
    function's docstring: a malformed or hallucinated argument (e.g. a
    dangling comma, an unquoted string) shouldn't zero out an otherwise
    correct tool-name prediction. Requiring a clean full parse for Acc too
    was tier2_oracle.py's own bug, not something compute_action_metrics'
    convention calls for -- that convention is specifically about EM/F1,
    which genuinely can't be scored without a structured argument tree.
    full_parse_ok tells the caller whether pred_call was well-formed enough
    for EM/F1 to mean anything (used for an accurate parse_failures count --
    counting only `pred_call is None` misses "extracted a call, but its
    arguments didn't parse")."""
    if not pred_call:
        return False, False, 0.0, False
    pred_tool_name = extract_tool_name(pred_call)
    ref_tool_name = ref_tree.get("intent")
    acc = bool(pred_tool_name and ref_tool_name and pred_tool_name.upper() == ref_tool_name.upper())
    try:
        pred_tree = parse_canonical_action(pred_call)
    except ActionParseError:
        return acc, False, 0.0, False
    return acc, bool(em(pred_tree, ref_tree)), slot_f1(pred_tree, ref_tree), True


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
    output = (response.choices[0].message.content or "").strip()

    pred_call = extract_action_call(output)
    pred_tool = extract_tool_name(pred_call) if pred_call else None
    acc, em_score, f1, full_parse_ok = score(pred_call, sample["_ref_tree"])

    log.info(
        "[%d/%d] [acc=%s em=%s f1=%.2f] expected=%s | predicted=%s",
        idx + 1, total, "O" if acc else "X", "O" if em_score else "X", f1,
        sample["expected_tool_call"], pred_call or "<none>",
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
        "full_parse_ok": full_parse_ok,
        "tool_acc": acc,
        "em": em_score,
        "slot_f1": f1,
    }


async def run_async(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    tools = load_tools(TOOLS_CSV)

    queries = load_unique_queries(DATA_PATH)
    if args.n_queries is not None:
        queries = queries[:: max(1, len(queries) // args.n_queries)][: args.n_queries]

    # 2/2,041 unique Tier-2 gold strings are themselves malformed dataset bugs
    # (an unescaped apostrophe mid-string, a stray non-ASCII character injected
    # into a tool name) -- parse gold once up front and drop those rows rather
    # than crashing mid-run or silently scoring them wrong against a broken
    # reference. Attaches the parsed tree so query_one/score don't re-parse it.
    clean_queries = []
    bad_gold = []
    for q in queries:
        try:
            q["_ref_tree"] = parse_canonical_action(q["expected_tool_call"])
            clean_queries.append(q)
        except ActionParseError as e:
            bad_gold.append((q["query_idx"], q["expected_tool_call"], str(e)))
    if bad_gold:
        log.warning(
            "Dropping %d/%d quer(y/ies) with unparseable gold expected_tool_call "
            "(dataset bug, not a model error): %s",
            len(bad_gold), len(queries), bad_gold[:5],
        )
    queries = clean_queries

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
    n = len(results)
    acc_count = sum(r["tool_acc"] for r in results)
    em_count = sum(r["em"] for r in results)
    avg_f1 = sum(r["slot_f1"] for r in results) / n
    # Not just `predicted_call is None` -- that only catches extract_action_call
    # finding nothing at all. A call can be extracted but still fail the full
    # parse_canonical_action (e.g. malformed argument syntax), which counts as
    # a parse failure for EM/F1 purposes too (see score()'s full_parse_ok).
    parse_failures = sum(1 for r in results if not r["full_parse_ok"])

    log.info(
        "Tier-2 oracle | Acc: %d/%d = %.1f%%  EM: %d/%d = %.1f%%  F1: %.1f%%  "
        "(paper target: Acc=77.1%% EM=10.1%% F1=19.3%%)  |  parse failures: %d  |  elapsed: %.1fs",
        acc_count, n, acc_count / n * 100, em_count, n, em_count / n * 100,
        avg_f1 * 100, parse_failures, elapsed,
    )

    model_name = Path(args.model).name
    out_data = {
        "model": args.model,
        "pipeline": "tier2_oracle",
        "enable_thinking": args.enable_thinking,
        "tool_format": args.tool_format,
        "topk": args.topk,
        "domain_filtered": args.domain_filtered,
        "retrieved_from": args.retrieved_from,
        "retrieved_topk": args.retrieved_topk if args.retrieved_from else None,
        "retrieved_domain_from": args.retrieved_domain_from,
        "system_prompt": system_prompt,
        "dropped_unparseable_gold": bad_gold,
        "paper_target": {"acc": 77.1, "em": 10.1, "f1": 19.3},
        "summary": {
            "total": n,
            "tool_acc": acc_count / n * 100,
            "em_accuracy": em_count / n * 100,
            "avg_slot_f1": avg_f1 * 100,
            "parse_failures": parse_failures,
            "elapsed_sec": round(elapsed, 1),
        },
        "samples": results,
    }

    think_tag = "think" if args.enable_thinking else "nothink"
    topk_tag = (
        f"top{args.topk}" if args.topk
        else "domainfiltered" if args.domain_filtered
        else f"retrieved{args.retrieved_topk}-{Path(args.retrieved_from).stem}" if args.retrieved_from
        else f"reddomain-{Path(args.retrieved_domain_from).stem}" if args.retrieved_domain_from
        else "all152"
    )
    out_path = (
        Path(args.output) if args.output
        else BASE_DIR / f"experiment/tier2_oracle/{model_name}_{args.tool_format}_{topk_tag}_{think_tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    log.info("Saved -> %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/Qwen3-8B")
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--n_queries", type=int, default=None,
                    help="Subsample for a quick pilot run (default: full 3,160 queries)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--enable_thinking", action="store_true")
    add_candidate_selection_args(p)
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_async(parse_args()))
