"""
Evaluation metrics for Audio2Tool canonical action (function-call) parses.

Ported as-is from noise_aware_slu/src/action_metrics.py (STOP retriever project):
the canonical action grammar used there, `INTENT(SLOT="value", ...)`, is exactly
the grammar Audio2Tool's `expected_tool_call` field already uses (e.g.
`setZoneTemperature(zone="Driver", temperature=22.0)`), so the parser/scorer needs
no changes -- only the callers (data loading, prompt building) differ per project.

Audio2Tool's own grading policy is tier-specific though (confirmed against
https://audio2tool.github.io/'s worked examples): Tier-1 ground truth is
argument-free by design ("Tool: setZoneTemperature", no args shown at all),
so its EM is tool-name match only and tier1_oracle.py does NOT use em() below
for that reason -- see its module docstring. Tier-2 onward *does* grade
arguments ("Tool: X: {args...}"), which is what tool_acc()/em()/slot_f1()
here are for. Callers for those tiers should use em() as the real Exact
Match, not roll their own name-only check the way tier1_oracle.py correctly
does for Tier-1 specifically.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class ActionParseError(ValueError):
    """Raised when a canonical action string cannot be parsed."""


# ---------------------------------------------------------------------------
# Parser: "INTENT(SLOT=\"value\", SLOT2=INTENT2(SLOT3=\"v\"))" -> tree
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_canonical_action(text: str) -> Dict[str, Any]:
    """Parse a canonical action string into {"intent": str, "slots": dict}.

    Slot values are strings, nested action dicts, or lists thereof (repeated
    slots are promoted to a list).
    """
    s = text.strip()
    if not s:
        raise ActionParseError("empty action string")

    def skip_ws(p: int) -> int:
        while p < len(s) and s[p].isspace():
            p += 1
        return p

    def parse_ident(p: int) -> Tuple[str, int]:
        m = _IDENT_RE.match(s, p)
        if not m:
            raise ActionParseError(f"expected identifier at {p}: {s[p:p + 20]!r}")
        return m.group(0), m.end()

    def parse_string(p: int) -> Tuple[str, int]:
        # Double-quoted: standard JSON string (used in this project's own
        # prompts, e.g. tier1_oracle's `tool_name(arg="value")` instruction).
        if s[p] == '"':
            try:
                value, end = json.JSONDecoder().raw_decode(s, p)
            except json.JSONDecodeError as e:
                raise ActionParseError(f"bad quoted string at {p}: {e}") from e
            if not isinstance(value, str):
                raise ActionParseError(f"expected string literal at {p}")
            return value, end
        # Single-quoted: Audio2Tool's own expected_tool_call gold strings use
        # this for the 13/2146 Tier-1 rows (and presumably more on other
        # tiers) that carry a non-empty argument, e.g.
        # getApplianceState(deviceId='dryer_1') -- Python-repr style, not
        # JSON, so decode it by hand (backslash-escapes the quote or backslash
        # itself, nothing fancier -- this taxonomy's string values are plain
        # device ids/enum labels, never containing embedded newlines etc.).
        assert s[p] == "'"
        chars: List[str] = []
        i = p + 1
        while True:
            if i >= len(s):
                raise ActionParseError(f"unterminated single-quoted string at {p}")
            ch = s[i]
            if ch == "\\" and i + 1 < len(s) and s[i + 1] in ("'", "\\"):
                chars.append(s[i + 1])
                i += 2
                continue
            if ch == "'":
                return "".join(chars), i + 1
            chars.append(ch)
            i += 1

    def parse_call_body(name: str, p: int) -> Tuple[Dict[str, Any], int]:
        assert s[p] == "("
        p = skip_ws(p + 1)
        slots: Dict[str, Any] = {}
        if p < len(s) and s[p] == ")":
            return {"intent": name, "slots": slots}, p + 1
        while True:
            key, p = parse_ident(p)
            p = skip_ws(p)
            if p >= len(s) or s[p] != "=":
                raise ActionParseError(f"expected '=' after {key!r} at {p}")
            value, p = parse_value(skip_ws(p + 1))
            if key in slots:
                if isinstance(slots[key], list):
                    slots[key].append(value)
                else:
                    slots[key] = [slots[key], value]
            else:
                slots[key] = value
            p = skip_ws(p)
            if p < len(s) and s[p] == ",":
                p = skip_ws(p + 1)
                continue
            break
        p = skip_ws(p)
        if p >= len(s) or s[p] != ")":
            raise ActionParseError(f"expected ')' at {p}: {s[p:p + 20]!r}")
        return {"intent": name, "slots": slots}, p + 1

    def parse_value(p: int) -> Tuple[Any, int]:
        p = skip_ws(p)
        if p < len(s) and s[p] in ("'", '"'):
            return parse_string(p)
        if p < len(s) and s[p] == "[":
            items: List[Any] = []
            p = skip_ws(p + 1)
            if p < len(s) and s[p] == "]":
                return items, p + 1
            while True:
                v, p = parse_value(p)
                items.append(v)
                p = skip_ws(p)
                if p < len(s) and s[p] == ",":
                    p = skip_ws(p + 1)
                    continue
                break
            p = skip_ws(p)
            if p >= len(s) or s[p] != "]":
                raise ActionParseError(f"expected ']' at {p}")
            return items, p + 1
        ident, p = parse_ident(p)
        p = skip_ws(p)
        if p < len(s) and s[p] == "(":
            return parse_call_body(ident, p)
        raise ActionParseError(f"unexpected token at {p}: {s[p:p + 20]!r}")

    p = skip_ws(0)
    name, p = parse_ident(p)
    p = skip_ws(p)
    if p >= len(s) or s[p] != "(":
        raise ActionParseError(f"expected '(' after intent name {name!r}")
    result, p = parse_call_body(name, p)
    return result


def extract_action_call(text: str) -> Optional[str]:
    """Pull the first balanced `IDENT(...)` call out of free-form model output.

    Strips completed <think>...</think> spans first (Qwen3 thinking mode). If
    a <think> block was opened but never closed, generation was truncated
    (hit max_tokens) mid-reasoning and no real answer exists yet -- returning
    None here avoids scanning half-finished reasoning prose for a spurious
    "word(" match.

    Within the post-think text, prefers content after a "Tool call:" / "Action:"
    marker (the format our prompts ask for) over the first parenthesized span
    anywhere in the output, for the same false-positive reason.
    """
    if "<think>" in text and "</think>" not in text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    action_marker = re.search(r"(Tool call|Action)\s*:", text, flags=re.IGNORECASE)
    search_text = text[action_marker.end():] if action_marker else text

    m = re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", search_text)
    if not m and action_marker:
        search_text = text
        m = re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", search_text)
    if not m:
        return None
    text = search_text
    start = m.start()
    p = m.end()
    depth = 1
    in_string = False
    while p < len(text) and depth > 0:
        ch = text[p]
        if in_string:
            if ch == "\\":
                p += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        p += 1
    if depth != 0:
        return None
    return text[start:p]


# ---------------------------------------------------------------------------
# Normalization + structural comparison
# ---------------------------------------------------------------------------

def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split()).upper()


def _normalize_identifier(name: str) -> str:
    """Canonicalize a function/slot identifier regardless of the casing style
    the model chose (UPPER_SNAKE_CASE, lower_snake_case, camelCase, PascalCase)."""
    if any(c.islower() for c in name):
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return name.upper()


def _multiset_eq(a: List[Any], b: List[Any], eq) -> bool:
    b = list(b)
    for item_a in a:
        for i, item_b in enumerate(b):
            if eq(item_a, item_b):
                del b[i]
                break
        else:
            return False
    return not b


def _action_eq(a: Any, b: Any, compare_text: bool) -> bool:
    """compare_text=True -> full EM; False -> tool-name-only structural match."""
    if isinstance(a, dict) and isinstance(b, dict):
        if _normalize_identifier(a["intent"]) != _normalize_identifier(b["intent"]):
            return False
        if not compare_text:
            return True
        a_slots = {_normalize_identifier(k): v for k, v in a["slots"].items()}
        b_slots = {_normalize_identifier(k): v for k, v in b["slots"].items()}
        keys_a, keys_b = set(a_slots), set(b_slots)
        if keys_a != keys_b:
            return False
        for k in keys_a:
            va, vb = a_slots[k], b_slots[k]
            va_list = va if isinstance(va, list) else [va]
            vb_list = vb if isinstance(vb, list) else [vb]
            if len(va_list) != len(vb_list):
                return False
            if not _multiset_eq(va_list, vb_list, lambda x, y: _action_eq(x, y, compare_text)):
                return False
        return True
    if isinstance(a, dict) or isinstance(b, dict):
        return False
    if isinstance(a, list) or isinstance(b, list):
        return False
    if not compare_text:
        return True
    return _normalize_text(a) == _normalize_text(b)


def extract_tool_name(action_call: str) -> Optional[str]:
    """Lenient tool-name extraction: just the identifier before the first "(".

    Used for Tool Accuracy specifically (paper: "fraction of examples for which
    the predicted tool name matches the ground truth tool") -- deliberately does
    NOT require the argument list to be well-formed, since a hallucinated or
    malformed argument (e.g. an unquoted number) shouldn't zero out an otherwise
    correct tool-name prediction the way a full parse_canonical_action failure
    would.
    """
    m = _IDENT_RE.match(action_call.strip())
    return m.group(0) if m else None


def tool_acc(pred_tree: Any, ref_tree: Any) -> int:
    """Tool Accuracy: predicted tool name matches ground truth (paper's `Acc`)."""
    pred_intent = pred_tree.get("intent") if isinstance(pred_tree, dict) else None
    ref_intent = ref_tree.get("intent") if isinstance(ref_tree, dict) else None
    if pred_intent is None or ref_intent is None:
        return int(pred_intent == ref_intent)
    return int(_normalize_identifier(pred_intent) == _normalize_identifier(ref_intent))


def em(pred_tree: Any, ref_tree: Any) -> int:
    """Exact Match: tool name + all arguments match (paper's `EM`)."""
    return int(_action_eq(pred_tree, ref_tree, compare_text=True))


def _flatten_slot_pairs(node: Any, out: List[Tuple[str, str]]) -> None:
    if not isinstance(node, dict):
        return
    for key, value in node["slots"].items():
        key = _normalize_identifier(key)
        values = value if isinstance(value, list) else [value]
        for v in values:
            if isinstance(v, dict):
                out.append((key, _normalize_text(_leaf_text(v))))
                _flatten_slot_pairs(v, out)
            else:
                out.append((key, _normalize_text(v)))


def _leaf_text(node: Any) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return " ".join(_leaf_text(v) for v in node)
    if isinstance(node, dict):
        return " ".join(_leaf_text(v) for v in node["slots"].values())
    return ""


def slot_f1(pred_tree: Any, ref_tree: Any) -> float:
    """Micro-averaged slot/parameter F1 (paper's `F1`; undefined for Tier-1)."""
    pred_pairs: List[Tuple[str, str]] = []
    ref_pairs: List[Tuple[str, str]] = []
    _flatten_slot_pairs(pred_tree, pred_pairs)
    _flatten_slot_pairs(ref_tree, ref_pairs)
    pred_set, ref_set = set(pred_pairs), set(ref_pairs)
    if not pred_set and not ref_set:
        return 1.0
    if not pred_set or not ref_set:
        return 0.0
    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set)
    recall = tp / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_action_metrics(pred_strs: List[str], ref_strs: List[str]) -> Dict[str, Any]:
    """Compute Tool-Acc / EM / Slot-F1 over canonical action strings, matching
    Audio2Tool paper Table 3's Acc/EM/F1 definitions. Unparseable predictions
    count as wrong on every metric but do not raise."""
    assert len(pred_strs) == len(ref_strs), "preds/refs length mismatch"

    acc_scores: List[int] = []
    em_scores: List[int] = []
    f1_scores: List[float] = []
    parse_failures = 0

    for pred_str, ref_str in zip(pred_strs, ref_strs):
        ref_tree = parse_canonical_action(ref_str)
        try:
            pred_tree = parse_canonical_action(pred_str) if pred_str else None
        except ActionParseError:
            pred_tree = None
        if pred_tree is None:
            parse_failures += 1
            acc_scores.append(0)
            em_scores.append(0)
            f1_scores.append(0.0)
            continue
        acc_scores.append(tool_acc(pred_tree, ref_tree))
        em_scores.append(em(pred_tree, ref_tree))
        f1_scores.append(slot_f1(pred_tree, ref_tree))

    n = len(pred_strs)
    return {
        "tool_acc": sum(acc_scores) / n,
        "em": sum(em_scores) / n,
        "slot_f1": sum(f1_scores) / n,
        "parse_failures": parse_failures,
        "n": n,
    }
