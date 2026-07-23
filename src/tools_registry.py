"""Load and format the Audio2Tool 152-tool taxonomy (tools_registry.csv)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import List, TypedDict


class Tool(TypedDict):
    tool_id: str
    domain: str
    category: str
    tool_name: str
    signature: str
    description: str
    argument_defaults: str
    argument_constraints: str


def load_tools(csv_path: Path) -> List[Tool]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))  # type: ignore[arg-type]


def format_tools_by_domain(tools: List[Tool]) -> str:
    """Full tool list grouped by domain, one line per tool: signature +
    description + defaults/constraints. Matches the format used for the
    paper's main-table ASR-LLM baseline (no domain/retriever filtering).

    Renders argument_defaults/argument_constraints in the registry's own
    "key='value'" kwarg syntax. An earlier version rewrote this as "key: value"
    prose, hypothesizing the model was visually copy-pasting the kwarg syntax
    into calls the query never asked for arguments in -- that turned out to be
    a non-issue (Tier-1 grading only checks the tool name; see tier1_oracle.py
    query_one), and the prose version measurably hurt Tool-Acc (the defaults
    text is genuinely useful for disambiguating near-synonym tools), so this
    reverts to the original kwarg rendering."""
    by_domain: dict[str, list] = defaultdict(list)
    for t in tools:
        by_domain[t["domain"]].append(t)
    sections = []
    for domain, dtools in by_domain.items():
        lines = [f"[{domain}]"]
        for t in dtools:
            lines.append(
                f"  - {t['signature']}: {t['description']} "
                f"(defaults: {t['argument_defaults']}, constraints: {t['argument_constraints']})"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def format_tools_by_domain_category(tools: List[Tool]) -> str:
    """Like format_tools_by_domain, but nests by category within domain (the
    taxonomy's own 23-category structure) instead of one flat per-domain list.

    Several tool_names in this taxonomy are near-synonyms with near-identical
    descriptions (e.g. rebootDevice[maintenance]/restartDevice[automation],
    setLighting/setLightState) -- note these particular pairs are NOT even in
    the same category, so this grouping does not resolve them; it's an
    experiment in giving the model more taxonomy structure, not a fix for
    query-text-level ambiguity (e.g. findMyDevice vs findPairedPhone, whose
    gold labels overlap on identical phrasings like "find my phone" across
    smart_home/wearables -- no tool-list formatting can disambiguate that)."""
    by_domain: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for t in tools:
        by_domain[t["domain"]][t["category"]].append(t)
    sections = []
    for domain, by_cat in by_domain.items():
        lines = [f"[{domain}]"]
        for category, ctools in by_cat.items():
            lines.append(f"  {category}:")
            for t in ctools:
                # No defaults/constraints here (Tier-1 direct commands take no
                # arguments) -- cutting them keeps the listing focused on the
                # one thing Tier-1 actually needs disambiguated: tool_name vs.
                # description, not argument schema noise.
                lines.append(f"    - {t['tool_name']}: {t['description']}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
