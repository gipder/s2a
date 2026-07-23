"""
Tier-agnostic building blocks shared by tier1_oracle.py, tier2_oracle.py
(and any future tierN_oracle.py): query loading, tool-candidate selection
(--topk / --domain_filtered / --retrieved_from), and prompt construction.
None of this depends on how a tier grades its results (Tier-1 is
tool-name-only, Tier-2+ grades arguments too -- that logic stays in each
tierN_oracle.py's own query_one/score).

Originally lived in tier1_oracle.py; pulled out once tier2_oracle.py started
importing from it too, since "a tier-2 script depends on the tier-1 script"
was the wrong shape (tier1_oracle.py should be free to change its own
Tier-1-specific grading without those changes rippling into tier2_oracle.py,
and vice versa).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tools_registry import format_tools_by_domain, format_tools_by_domain_category

TOOL_FORMATS = {
    "domain": format_tools_by_domain,
    "domain_category": format_tools_by_domain_category,
}


def load_unique_queries(data_path: Path) -> list[dict]:
    """Audio2Tool tier data files have 2 rows per query (2 speaker renditions
    each) -- oracle mode is text-only, so dedupe to one row per query_idx."""
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
    keeps the candidate set internally coherent regardless.

    Silently returns fewer than k candidates if the domain doesn't have k-1
    distractors to offer (wearables has only 13 tools total, so --topk above
    13 always clamps there) -- the caller's "top-{k}" log line / output
    filename still says the requested k, not the actual count actually used
    for that particular query."""
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
        f"or tool_name() if it takes no arguments. Extract every argument value from "
        f"the user utterance itself -- do not copy the \"defaults\" shown above, and do "
        f"not omit an argument the utterance specifies a value for.\n\n"
        f"Tool call:"
    )


def shorten_prompt(prompt: str) -> str:
    """Replace the tool-listing block with a pointer to tools_registry.csv.
    Used whenever the candidate set is constant across many samples --
    full-152 (--topk/--domain_filtered/--retrieved_from/--retrieved_domain_from
    all unset), --domain_filtered, and --retrieved_domain_from alike: each of
    the latter two's tool list is one of only 3 fixed per-domain blocks (up
    to 86 tools, smart_car), repeated verbatim across many samples, so
    logging it in full blew up a Tier-2 domain_filtered run to 29.7MB (vs.
    2.7MB for all152) for zero benefit -- all three are fully reconstructible
    from tools_registry.csv + this run's --tool_format, given which domain
    was actually used.

    That "which domain" is NOT simply the sample's own `domain` field, despite
    an earlier version of this placeholder claiming so -- --domain_filtered
    uses the GT tool's own registered domain in tools_registry.csv (looked up
    by tool_name via filter_tools_by_domain), which disagrees with the item's
    `domain` field on 64 known Tier-1 rows (see that function's docstring).
    --retrieved_domain_from is a different data source entirely: the
    domain PREDICTED by the retriever's top-1 tool, which can differ from
    the true domain whenever the retriever guessed wrong -- reconstructing
    it needs the actual --retrieved_domain_from JSON file, not anything on
    this sample. The placeholder below points at the responsible code path
    (and doesn't restate the tool count -- already visible in the kept
    "(N total)" prefix, so it can't drift out of sync with the actual N the
    way a hardcoded "152" would for a 53-tool domain block)."""
    import re
    return re.sub(
        r"(Available tools \(\d+ total\), grouped by domain:\n).*?(?=\n\nUser utterance:)",
        r"\1<tool list omitted -- reconstructible from data/Audio2Tool_mod/tools_registry.csv "
        r"+ this run's tool_format + this run's actual candidate-selection mode (see "
        r"oracle_shared.filter_tools_by_domain / --retrieved_domain_from, NOT simply this "
        r"sample's own domain field)>",
        prompt,
        flags=re.DOTALL,
    )


def add_candidate_selection_args(p: argparse.ArgumentParser) -> None:
    """--topk / --domain_filtered / --retrieved_from / --retrieved_topk /
    --retrieved_domain_from / --seed / --tool_format -- identical across
    every tierN_oracle.py."""
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
                         "instead of the full 152-tool list. Mutually exclusive with the other modes")
    p.add_argument("--retrieved_topk", type=int, default=10,
                    help="How many of --retrieved_from's ranked candidates to actually use")
    p.add_argument("--retrieved_domain_from", default=None,
                    help="Path to eval_zeroshot_retriever.py output JSON -- non-oracle version of "
                         "--domain_filtered: predict the query's domain from the retriever's own "
                         "top-1 tool (not the ground truth), then give the LLM every tool in that "
                         "PREDICTED domain. Mutually exclusive with the other modes")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --topk distractor sampling")


def _load_retrieval_file(path: str, queries: list[dict], retrieved_topk: int, tools_csv_path: Path) -> dict[int, list[str]]:
    """Load + validate an eval_zeroshot_retriever.py output JSON, shared by
    --retrieved_from and --retrieved_domain_from (same file format, same
    failure modes to guard against). Returns {query_idx: ranked tool names}."""
    with open(path) as f:
        retrieved_raw = json.load(f)
    if "samples" not in retrieved_raw:
        raise ValueError(f"{path} has no 'samples' key -- not an eval_zeroshot_retriever.py output?")
    missing_field = [r["query_idx"] for r in retrieved_raw["samples"] if "retrieved_tools" not in r]
    if missing_field:
        raise ValueError(
            f"{path}: {len(missing_field)} sample(s) have no 'retrieved_tools' field "
            f"(old schema used 'top5_tools' -- re-run eval_zeroshot_retriever.py to regenerate)"
        )
    saved_topk = len(retrieved_raw["samples"][0]["retrieved_tools"])
    if retrieved_topk > saved_topk:
        raise ValueError(
            f"requested top-{retrieved_topk} exceeds the {saved_topk} candidates saved per query in "
            f"{path} (its --save_topk at eval_zeroshot_retriever.py run time) -- silently truncating "
            f"would make the top-{retrieved_topk} log/filename tag lie about the actual candidate "
            f"count; re-run eval_zeroshot_retriever.py with a larger --save_topk instead"
        )
    retrieved = {r["query_idx"]: r["retrieved_tools"] for r in retrieved_raw["samples"]}
    missing_queries = [q["query_idx"] for q in queries if q["query_idx"] not in retrieved]
    if missing_queries:
        raise ValueError(
            f"{len(missing_queries)}/{len(queries)} query_idx from this run are not present in "
            f"{path} (e.g. {missing_queries[:5]}) -- likely a --n_queries subsample mismatch "
            f"between this run and the retriever run (or a wrong --tier at retrieval time); "
            f"use the same --n_queries/--tier (or none) for both"
        )
    return retrieved


def build_candidate_prompts(
    args: argparse.Namespace, tools: list[dict], queries: list[dict], tools_csv_path: Path,
) -> tuple[list[str], bool, str]:
    """Applies whichever of --topk/--domain_filtered/--retrieved_from/
    --retrieved_domain_from/(none) args selected, and returns (prompts,
    store_full_prompt, tools_desc) -- one prompt per query in the same
    order, whether the per-sample prompt is worth logging in full (see
    shorten_prompt), and a human-readable description of the mode for
    logging.

    Shared verbatim by every tierN_oracle.py's run_async: candidate selection
    doesn't know or care which tier's queries it's building prompts for."""
    modes_selected = sum(x is not None for x in [args.topk, args.retrieved_from, args.retrieved_domain_from]) + int(args.domain_filtered)
    if modes_selected > 1:
        raise ValueError("--topk, --domain_filtered, --retrieved_from, --retrieved_domain_from are mutually exclusive")
    if args.topk is not None and args.topk < 1:
        # sample_topk_candidates does rng.sample(pool, k - 1); k=0 makes that
        # a negative sample size, which random.sample rejects with a generic
        # ValueError that doesn't say *why* -- fail with a clearer message
        # before it gets there.
        raise ValueError(f"--topk must be >= 1, got {args.topk}")

    tools_desc = (
        f"top-{args.topk} (oracle: GT+random distractors)" if args.topk
        else "domain-filtered (~53/86/13 per domain, ORACLE true domain)" if args.domain_filtered
        else f"top-{args.retrieved_topk} (REAL retriever, no GT guarantee)" if args.retrieved_from
        else "domain-filtered (~53/86/13 per domain, PREDICTED via retriever top-1)" if args.retrieved_domain_from
        else str(len(tools))
    )
    by_name = {t["tool_name"]: t for t in tools}

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
        retrieved = _load_retrieval_file(args.retrieved_from, queries, args.retrieved_topk, tools_csv_path)
        unknown_names = {n for names in retrieved.values() for n in names[: args.retrieved_topk]} - set(by_name)
        if unknown_names:
            raise ValueError(
                f"{args.retrieved_from} references tool name(s) not in {tools_csv_path}: "
                f"{sorted(unknown_names)[:5]} -- registry may have changed since the retriever was run"
            )
        prompts = []
        for q in queries:
            names = retrieved[q["query_idx"]][: args.retrieved_topk]
            candidates = [by_name[n] for n in names]
            prompts.append(build_user_prompt(q["query"], TOOL_FORMATS[args.tool_format](candidates), len(candidates)))
    elif args.retrieved_domain_from is not None:
        # Non-oracle --domain_filtered: don't assume the true domain is known
        # (that's what --domain_filtered does), predict it instead from the
        # retriever's own top-1 tool's registered domain, then hand the LLM
        # every tool in that PREDICTED domain. Only needs the top-1 candidate,
        # so retrieved_topk=1 here regardless of --retrieved_topk (that flag
        # is --retrieved_from's, not this mode's).
        retrieved = _load_retrieval_file(args.retrieved_domain_from, queries, 1, tools_csv_path)
        unknown_names = {names[0] for names in retrieved.values()} - set(by_name)
        if unknown_names:
            raise ValueError(
                f"{args.retrieved_domain_from} references tool name(s) not in {tools_csv_path}: "
                f"{sorted(unknown_names)[:5]} -- registry may have changed since the retriever was run"
            )
        prompts = []
        for q in queries:
            top1_tool = retrieved[q["query_idx"]][0]
            predicted_domain = by_name[top1_tool]["domain"]
            candidates = [t for t in tools if t["domain"] == predicted_domain]
            prompts.append(build_user_prompt(q["query"], TOOL_FORMATS[args.tool_format](candidates), len(candidates)))
    else:
        tools_str = TOOL_FORMATS[args.tool_format](tools)
        prompts = [build_user_prompt(q["query"], tools_str, len(tools)) for q in queries]

    # Only genuinely per-sample candidate sets are worth logging in full --
    # --domain_filtered/--retrieved_domain_from's tool list is one of 3 fixed
    # per-domain blocks repeated across many samples, same shorten_prompt
    # treatment as all-152.
    store_full_prompt = args.topk is not None or args.retrieved_from is not None

    return prompts, store_full_prompt, tools_desc
