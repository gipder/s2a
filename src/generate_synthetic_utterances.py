#!/usr/bin/env python3
"""
Generate synthetic training utterances per tool for the Audio2Tool retriever.

Audio2Tool ships no train split, so this synthesizes (utterance, tool) pairs
the same way the paper's own authors generated Tier1-8 queries -- from a
tool's function card, via an LLM -- but with a different LLM (Qwen3-32B here
vs. their GPT-5.2/Gemini/Claude Opus) and with test-set leakage filtering
applied afterward (filter_synthetic_utterances.py), since both draw from the
same narrow "natural phrasing for this exact function" distribution and could
coincidentally collide even without either side copying the other.

For each of the 152 tools, asks the LLM for --k diverse direct-command
utterances (Tier-1 style: short, no filler, no explicit argument values) in
one JSON-list call per tool, with an explicit diversity instruction (vary
length, register, phrasing strategy) so the model doesn't just produce k
near-paraphrases of each other.

Usage:
    ./script/run_generate_synthetic_utterances.sh
    K=5 N_TOOLS=5 ./script/run_generate_synthetic_utterances.sh   # quick pilot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

from openai import AsyncOpenAI

from tools_registry import load_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TOOLS_CSV = BASE_DIR / "data/Audio2Tool_mod/tools_registry.csv"

SYSTEM_PROMPT = (
    "You are generating training data for a voice-assistant tool-calling benchmark. "
    "Given a tool's specification, write short, natural spoken commands a real user "
    "might say to invoke it directly."
)


def build_prompt(tool: dict, k: int) -> str:
    return (
        f"Tool: {tool['signature']}\n"
        f"Description: {tool['description']}\n"
        f"Domain: {tool['domain']} / {tool['category']}\n\n"
        f"Write {k} DIFFERENT short spoken commands (2-8 words each) a user might say "
        f"to invoke this tool DIRECTLY. Each command must be a complete, natural sentence "
        f"that does NOT mention any specific argument value, number, mode name, or setting "
        f"-- describe the desired outcome in general terms instead. For example, for a "
        f"tool that sets a temperature: GOOD = \"turn up the heat\", \"make it warmer in "
        f"here\"; BAD = \"set the temperature to 72\", \"set the temperature to a level\" "
        f"(never reference \"a value\"/\"a level\"/\"a setting\" as a placeholder either -- "
        f"omit the argument from the sentence entirely). No filler words, no explanation, "
        f"one clear intent per utterance. Vary the phrasing style across the {k} "
        f"(imperative / question / casual / polite / terse) so they don't all sound alike.\n\n"
        f"Respond with ONLY a JSON array of {k} strings, nothing else."
    )


def parse_json_list(text: str) -> list[str]:
    """Extract a JSON string-array from free-form model output. Strips
    completed <think> blocks first (Qwen3 thinking mode); returns [] if no
    valid array is found (caller logs this as an under-target tool)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    return [x.strip() for x in items if isinstance(x, str) and x.strip()]


async def generate_one(
    client: AsyncOpenAI, model: str, tool: dict, k: int,
    idx: int, total: int, semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        prompt = build_prompt(tool, k)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,  # diversity matters more than determinism here
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        output = response.choices[0].message.content or ""
        utterances = parse_json_list(output)
        log.info("[%d/%d] %s: %d/%d utterances", idx + 1, total, tool["tool_name"], len(utterances), k)
        return {
            "tool_name": tool["tool_name"],
            "domain": tool["domain"],
            "category": tool["category"],
            "signature": tool["signature"],
            "description": tool["description"],
            "utterances": utterances,
            "raw_output": output,
        }


async def run_async(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    tools = load_tools(TOOLS_CSV)
    if args.n_tools is not None:
        tools = tools[: args.n_tools]

    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*[
        generate_one(client, args.model, t, args.k, i, len(tools), semaphore)
        for i, t in enumerate(tools)
    ])

    total_utterances = sum(len(r["utterances"]) for r in results)
    short = [r["tool_name"] for r in results if len(r["utterances"]) < args.k]
    log.info(
        "Generated %d utterances across %d tools (target %d/tool). "
        "%d tool(s) under target: %s",
        total_utterances, len(results), args.k, len(short), short[:10],
    )

    model_name = Path(args.model).name
    out_path = (
        Path(args.output) if args.output
        else BASE_DIR / f"experiment/synthetic_utterances/raw_{model_name}_k{args.k}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "k": args.k, "tools": results}, f, indent=2, ensure_ascii=False)
    log.info("Saved -> %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/Qwen3-32B")
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--k", type=int, default=20, help="Target utterances per tool")
    p.add_argument("--n_tools", type=int, default=None, help="Pilot on the first N tools only")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_async(parse_args()))
