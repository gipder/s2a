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
prompt building are unchanged from Tier-1 -- imported directly from
tier1_oracle.py rather than duplicated, since none of that logic is
Tier-1-specific (see that module for the domain-label-mismatch robustness
notes on sample_topk_candidates/filter_tools_by_domain).

Usage:
    python src/tier2_oracle.py --model model/Qwen3-8B
    python src/tier2_oracle.py --model model/Qwen3-8B --n_queries 50  # pilot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
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
    tool_acc,
)
from tier1_oracle import (
    TOOL_FORMATS,
    build_user_prompt,
    filter_tools_by_domain,
    load_unique_queries,
    sample_topk_candidates,
    shorten_prompt,
)
from tools_registry import load_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data/Audio2Tool_mod/public/tier2_parametric_data/tier2_parametric.json"
TOOLS_CSV = BASE_DIR / "data/Audio2Tool_mod/tools_registry.csv"


def score(pred_call: str | None, ref_tree: dict) -> tuple[bool, bool, float]:
    """Returns (tool_acc, em, slot_f1) against an already-parsed gold tree
    (see filter_unparseable_gold -- unlike Tier-1, 2/2,041 unique Tier-2 gold
    strings are themselves malformed dataset bugs, e.g. an unescaped
    apostrophe or a stray non-ASCII character mid-identifier, so parsing gold
    happens once up front with those rows dropped, not per-query here).
    pred_call may still fail to parse (malformed/hallucinated model output),
    which -- matching compute_action_metrics' convention -- scores 0 on every
    metric rather than raising."""
    if not pred_call:
        return False, False, 0.0
    try:
        pred_tree = parse_canonical_action(pred_call)
    except ActionParseError:
        return False, False, 0.0
    return bool(tool_acc(pred_tree, ref_tree)), bool(em(pred_tree, ref_tree)), slot_f1(pred_tree, ref_tree)


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
    acc, em_score, f1 = score(pred_call, sample["_ref_tree"])

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

    if sum(x is not None for x in [args.topk, args.retrieved_from]) + int(args.domain_filtered) > 1:
        raise ValueError("--topk, --domain_filtered, --retrieved_from are mutually exclusive")

    tools_desc = (
        f"top-{args.topk} (oracle: GT+random distractors)" if args.topk
        else "domain-filtered (~53/86/13 per domain)" if args.domain_filtered
        else f"top-{args.retrieved_topk} (REAL retriever, no GT guarantee)" if args.retrieved_from
        else str(len(tools))
    )
    log.info(
        "Model: %s | Queries: %d | Tools: %s | Thinking: %s | Concurrency: %d",
        args.model, len(queries), tools_desc, args.enable_thinking, args.concurrency,
    )

    if args.topk is not None:
        rng = random.Random(args.seed)
        prompts = []
        for q in queries:
            candidates = sample_topk_candidates(tools, q["tool_name"], args.topk, rng)
            prompts.append(build_user_prompt(q["query"], TOOL_FORMATS[args.tool_format](candidates), len(candidates)))
    elif args.domain_filtered:
        prompts = []
        for q in queries:
            candidates = filter_tools_by_domain(tools, q["tool_name"])
            prompts.append(build_user_prompt(q["query"], TOOL_FORMATS[args.tool_format](candidates), len(candidates)))
    elif args.retrieved_from is not None:
        if args.retrieved_topk < 1:
            raise ValueError(f"--retrieved_topk must be >= 1, got {args.retrieved_topk}")
        by_name = {t["tool_name"]: t for t in tools}
        with open(args.retrieved_from) as f:
            retrieved_raw = json.load(f)
        if "samples" not in retrieved_raw:
            raise ValueError(f"{args.retrieved_from} has no 'samples' key -- not an eval_zeroshot_retriever.py output?")
        missing_field = [r["query_idx"] for r in retrieved_raw["samples"] if "retrieved_tools" not in r]
        if missing_field:
            raise ValueError(
                f"{args.retrieved_from}: {len(missing_field)} sample(s) have no 'retrieved_tools' field "
                f"(old schema used 'top5_tools' -- re-run eval_zeroshot_retriever.py to regenerate)"
            )
        saved_topk = len(retrieved_raw["samples"][0]["retrieved_tools"])
        if args.retrieved_topk > saved_topk:
            raise ValueError(
                f"--retrieved_topk={args.retrieved_topk} exceeds the {saved_topk} candidates saved per "
                f"query in {args.retrieved_from} -- re-run eval_zeroshot_retriever.py with a larger --save_topk"
            )
        retrieved = {r["query_idx"]: r["retrieved_tools"] for r in retrieved_raw["samples"]}
        missing_queries = [q["query_idx"] for q in queries if q["query_idx"] not in retrieved]
        if missing_queries:
            raise ValueError(
                f"{len(missing_queries)}/{len(queries)} query_idx from this run are not present in "
                f"{args.retrieved_from} (e.g. {missing_queries[:5]}) -- was this retriever run against "
                f"Tier-2 queries, not Tier-1? eval_zeroshot_retriever.py needs --data_path pointed at "
                f"tier2_parametric.json to produce a compatible file"
            )
        unknown_names = {n for names in retrieved.values() for n in names[: args.retrieved_topk]} - set(by_name)
        if unknown_names:
            raise ValueError(
                f"{args.retrieved_from} references tool name(s) not in {TOOLS_CSV}: {sorted(unknown_names)[:5]}"
            )
        prompts = []
        for q in queries:
            names = retrieved[q["query_idx"]][: args.retrieved_topk]
            candidates = [by_name[n] for n in names]
            prompts.append(build_user_prompt(q["query"], TOOL_FORMATS[args.tool_format](candidates), len(candidates)))
    else:
        tools_str = TOOL_FORMATS[args.tool_format](tools)
        prompts = [build_user_prompt(q["query"], tools_str, len(tools)) for q in queries]

    semaphore = asyncio.Semaphore(args.concurrency)
    # Only genuinely per-sample candidate sets are worth logging in full --
    # --domain_filtered's tool list is one of 3 fixed per-domain blocks
    # repeated across many samples, same shorten_prompt treatment as all-152
    # (see that function's docstring in tier1_oracle.py for the motivation).
    store_full_prompt = args.topk is not None or args.retrieved_from is not None

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
    parse_failures = sum(1 for r in results if r["predicted_call"] is None)

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
    p.add_argument("--tool_format", default="domain", choices=list(TOOL_FORMATS))
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--domain_filtered", action="store_true")
    p.add_argument("--retrieved_from", default=None)
    p.add_argument("--retrieved_topk", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_async(parse_args()))
