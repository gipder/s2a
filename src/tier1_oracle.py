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
match zero args". This does NOT generalize to later tiers: a future
tier2_oracle.py etc. must grade EM with action_metrics.em() (full
name+argument tree match) instead of reusing this file's name-only logic.

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

from action_metrics import ActionParseError, extract_action_call, extract_tool_name, parse_canonical_action
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


def filter_tools_by_domain(all_tools: list[dict], gt_tool_name: str) -> list[dict]:
    """All tools from the GT tool's own domain -- the middle ground between
    full-152 and the --topk retriever-shaped upper bound: a realistic
    "we know the assistant's domain (car/home/wearable), not which specific
    tool" scenario, same category reasoning_for_asr's Tier4 experiments tested
    (~53-tool domain-filtered vs 152 all). Domain sizes here are uneven
    (smart_car=86, smart_home=53, wearables=13 -- see tools_registry.csv), so
    this isn't a fixed-size cut like --topk, just "not all 3 domains at once".

    Same robustness note as sample_topk_candidates: uses the GT tool's own
    registered domain, not the item's `domain` field, since those disagree on
    64/2146 Tier-1 rows (see that function's docstring / README)."""
    by_name = {t["tool_name"]: t for t in all_tools}
    real_domain = by_name[gt_tool_name]["domain"]
    return [t for t in all_tools if t["domain"] == real_domain]


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
    Used whenever the candidate set is constant across many samples --
    full-152 (--topk/--domain_filtered/--retrieved_from all unset) and
    --domain_filtered alike: the latter's tool list is one of only 3 fixed
    per-domain blocks (up to 86 tools, smart_car), repeated verbatim across
    every sample of that domain, so logging it in full blew up a Tier-2
    domain_filtered run to 29.7MB (vs. 2.7MB for all152) for zero benefit --
    both are fully reconstructible from tools_registry.csv + this run's
    --tool_format + (for domain_filtered) the sample's own `domain` field.
    NOT used for --topk/--retrieved_from, where the candidate set is
    genuinely per-sample (random distractors / actual retrieval), so there's
    real information to keep. The placeholder deliberately doesn't restate
    the tool count itself -- it's already visible in the kept
    "(N total)" prefix, so it can't drift out of sync with the actual N
    (a hardcoded "152" here would be silently wrong for a 53-tool domain
    block)."""
    return re.sub(
        r"(Available tools \(\d+ total\), grouped by domain:\n).*?(?=\n\nUser utterance:)",
        r"\1<tool list omitted -- reconstructible from data/Audio2Tool_mod/tools_registry.csv "
        r"+ this run's tool_format + (if domain_filtered) this sample's domain field>",
        prompt,
        flags=re.DOTALL,
    )


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
        # Retriever-shaped upper bound: GT + (k-1) same-domain random distractors,
        # instead of the full 152-tool taxonomy. Not a real retriever -- see
        # sample_topk_candidates docstring.
        rng = random.Random(args.seed)
        prompts = []
        for q in queries:
            candidates = sample_topk_candidates(tools, q["tool_name"], args.topk, rng)
            prompts.append(build_user_prompt(q["query"], TOOL_FORMATS[args.tool_format](candidates), len(candidates)))
    elif args.domain_filtered:
        # Middle ground between full-152 and --topk -- see filter_tools_by_domain docstring.
        prompts = []
        for q in queries:
            candidates = filter_tools_by_domain(tools, q["tool_name"])
            prompts.append(build_user_prompt(q["query"], TOOL_FORMATS[args.tool_format](candidates), len(candidates)))
    elif args.retrieved_from is not None:
        # Real (non-oracle) retriever pipeline: candidates are whatever
        # eval_zeroshot_retriever.py actually retrieved, GT may or may not be
        # among them -- unlike --topk/--domain_filtered above, this can fail
        # outright when the retriever's top-k misses the gold tool entirely.
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
                f"query in {args.retrieved_from} (its --save_topk at eval_zeroshot_retriever.py run time) "
                f"-- silently truncating would make the top-{args.retrieved_topk} log/filename tag lie "
                f"about the actual candidate count; re-run eval_zeroshot_retriever.py with a larger --save_topk instead"
            )
        retrieved = {r["query_idx"]: r["retrieved_tools"] for r in retrieved_raw["samples"]}
        missing_queries = [q["query_idx"] for q in queries if q["query_idx"] not in retrieved]
        if missing_queries:
            raise ValueError(
                f"{len(missing_queries)}/{len(queries)} query_idx from this run are not present in "
                f"{args.retrieved_from} (e.g. {missing_queries[:5]}) -- likely a --n_queries subsample "
                f"mismatch between this run and the retriever run; use the same --n_queries (or none) for both"
            )
        unknown_names = {n for names in retrieved.values() for n in names[: args.retrieved_topk]} - set(by_name)
        if unknown_names:
            raise ValueError(
                f"{args.retrieved_from} references tool name(s) not in {TOOLS_CSV}: {sorted(unknown_names)[:5]} "
                f"-- registry may have changed since the retriever was run"
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
    # (see that function's docstring for the 29.7MB-vs-2.7MB motivation).
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
    p.add_argument("--tool_format", default="domain", choices=list(TOOL_FORMATS),
                    help="Tool-list prompt format (default: flat per-domain list)")
    p.add_argument("--topk", type=int, default=None,
                    help="Retriever-shaped upper bound: GT + (k-1) same-domain random "
                         "distractors instead of the full 152-tool list (default: full list)")
    p.add_argument("--domain_filtered", action="store_true",
                    help="All tools from the GT tool's own domain (~53/86/13 depending on "
                         "domain) instead of the full 152-tool list. Mutually exclusive with --topk")
    p.add_argument("--retrieved_from", default=None,
                    help="Path to eval_zeroshot_retriever.py output JSON -- use its actual "
                         "retrieved candidates (real retriever, GT not guaranteed present) "
                         "instead of the full 152-tool list. Mutually exclusive with --topk/--domain_filtered")
    p.add_argument("--retrieved_topk", type=int, default=10,
                    help="How many of --retrieved_from's ranked candidates to actually use")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --topk distractor sampling")
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_async(parse_args()))
