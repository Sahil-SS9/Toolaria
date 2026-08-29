"""Content type detection and excerpt builders for Toolaria."""
import functools
import json
import re


# Exact-budget contract (2026-08-29 takeover of PR #4):
# - ``excerpt_max_chars`` is an EXACT cap on the excerpt block: the
#   assembled excerpt, including any truncation marker, is never longer
#   than the configured limit.
# - One shared seam: every payload kind (json, html, code, text) exits
#   through ``_assemble_excerpt``; no early return can bypass the budget.
# - When the budget binds, space is allocated by priority (anchors 40%,
#   head 40%, tail 20%) instead of prefix-slicing the assembled result —
#   a fat head can no longer delete the tail or promoted error lines.
# - The truncation marker counts inside the limit.
_TRUNCATION_MARKER = "[... excerpt truncated to excerpt_max_chars]"
LINE_CAP = 500
MIN_EXCERPT_MAX_CHARS = 200
_DEFAULT_EXCERPT_MAX_CHARS = 8000


def _normalized_cap(cfg: dict) -> int:
    """Return a usable ``excerpt_max_chars``.

    Accepts any int-convertible value; values below ``MIN_EXCERPT_MAX_CHARS``
    (including zero/negative) clamp to the minimum; malformed values fall
    back to the default. Register-time validation in ``_merge_cfg`` fails
    loudly on genuinely bad operator config; this tolerant path is the
    runtime safety net (cfg can reach here from tests/embedders).
    """
    raw = cfg.get("excerpt_max_chars", _DEFAULT_EXCERPT_MAX_CHARS)
    try:
        cap = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_EXCERPT_MAX_CHARS
    if cap < MIN_EXCERPT_MAX_CHARS:
        return MIN_EXCERPT_MAX_CHARS
    return cap


@functools.lru_cache(maxsize=8)
def _err_re(patterns: tuple):
    # Keyed by the pattern tuple so a config change takes effect.
    return re.compile("|".join(re.escape(p) for p in patterns), re.I)


def detect_type(raw: str):
    """Return (kind: str, meta: str): detects JSON, HTML, code, text, binary."""
    if raw is None:
        return ("binary", "None")
    if isinstance(raw, bytes):
        return ("binary", f"{len(raw)}b")
    if not isinstance(raw, str):
        return ("text", type(raw).__name__)

    stripped = raw.lstrip()
    if not stripped:
        return ("text", "empty")

    # JSON
    if stripped[0] in "{[":
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                return ("json", f"array[{len(obj)}]")
            if isinstance(obj, dict):
                ks = list(obj.keys())[:3]
                return ("json", f"object keys: {ks}")
            return ("json", type(obj).__name__)
        except (json.JSONDecodeError, ValueError):
            pass

    # HTML / XML
    if re.match(r"^\s*<!DOCTYPE\s+html|<html|<body|<div|<table|<svg|<xml", stripped, re.I):
        tag = re.match(r"^\s*<(\w+)", stripped, re.I)
        tname = tag.group(1).lower() if tag else "?"
        return ("html", f"<{tname}>")

    # Possible code (shebang or common keywords)
    if re.match(r"^\s*#!|^\s*(import |def |class |function |const |let |var |use |package |module )", stripped):
        return ("code", "source")

    return ("text", f"{len(raw):,} chars")


def _safe_json_head(obj, n):
    """Return first n items (array) or first n key:val pairs (dict)."""
    if isinstance(obj, list):
        return obj[:n]
    if isinstance(obj, dict):
        items = list(obj.items())[:n]
        return dict(items)
    return str(obj)[:2000]


def _safe_json_tail(obj, n):
    """Return last n items (array) or last n key:val pairs (dict)."""
    if isinstance(obj, list):
        return obj[-n:] if n > 0 else []
    if isinstance(obj, dict):
        items = list(obj.items())[-n:] if n > 0 else []
        return dict(items)
    return ""


# Section priority classes: header is always kept verbatim; anchors
# (promoted error/decision/action/value lines) outrank head, which
# outranks tail. Weights of the truncation-time content pool.
_POOL_WEIGHTS = {"anchor": 0.4, "head": 0.4, "tail": 0.2}


def _join_sections(sections) -> str:
    parts = []
    for sec in sections:
        if sec["lines"]:
            if sec["label"]:
                parts.append(sec["label"])
            parts.extend(sec["lines"])
    return "\n".join(parts)


def _fill_section(lines, budget, from_end: bool = False):
    """Take lines up to *budget* chars as-joined. Whole lines first; the
    last slot may be exactly filled by truncating the next candidate
    (``…`` suffix marks a mid-line cut). ``from_end`` keeps the END of the
    line list — tails carry the sentences closest to the end of output."""
    out: list[str] = []
    acc = 0
    order = range(len(lines) - 1, -1, -1) if from_end else range(len(lines))
    used_next_slot = False
    for i in order:
        ln = lines[i][:LINE_CAP]
        w = len(ln) + (1 if out else 0)
        if acc + w <= budget:
            out.append(ln)
            acc += w
        elif budget - acc >= 16 and not used_next_slot and out:
            # Exact-fill the remaining budget with a cut line. When filling
            # from the end, the cut line sits BEFORE the kept lines.
            room = budget - acc - 1  # separator
            out.append(ln[:room - 1] + "…")
            used_next_slot = True
            acc = budget
            break
        else:
            break
    if from_end:
        out.reverse()
    return out


def _assemble_excerpt(sections, cap: int):
    """Assemble sections under the exact cap. Returns (text, truncated).

    Contract: ``len(text) <= cap`` always; when the full assembly does not
    fit, the output ends with ``_TRUNCATION_MARKER`` and header sections
    are kept verbatim while content sections are allocated pools by
    priority (anchor > head > tail).
    """
    full = _join_sections(sections)
    if len(full) <= cap:
        return full, False

    header_secs = [s for s in sections if s["prio"] == "header"]
    content_secs = [s for s in sections if s["prio"] != "header"]
    fixed = "\n".join(
        "\n".join(([s["label"]] if s["label"] else []) + list(s["lines"]))
        for s in header_secs
    )
    # The marker is part of the output, so it counts inside the cap.
    pool = cap - len(fixed) - len(_TRUNCATION_MARKER) - 1
    if pool <= 0:
        # Degenerate configuration: header alone eats the budget. Ship a
        # hard-truncated header + marker, still within cap.
        keep = max(cap - len(_TRUNCATION_MARKER) - 1, 0)
        return fixed[:keep] + "\n" + _TRUNCATION_MARKER, True

    # Per-class pools by priority. Only classes that actually have sections
    # get pools (weights renormalised) — an absent anchor class must not
    # strand 40% of the budget unused. The last present class absorbs the
    # rounding remainder.
    present = [c for c in ("anchor", "head", "tail")
               if any(s["prio"] == c for s in content_secs)]
    weight_sum = sum(_POOL_WEIGHTS[c] for c in present)
    class_pools = {}
    allocated = 0
    for c in present[:-1]:
        share = int(pool * _POOL_WEIGHTS[c] / weight_sum)
        class_pools[c] = share
        allocated += share
    class_pools[present[-1]] = max(pool - allocated, 0)

    # Each section gets: its label (if any) + whole lines, measured as they
    # will be joined. Anchors share their pool equally (remainder to last).
    anchor_sec_indices = [i for i, s in enumerate(content_secs)
                          if s["prio"] == "anchor"]
    n_anchor = len(anchor_sec_indices)
    out_parts: list[str] = []
    for i, sec in enumerate(content_secs):
        cls = sec["prio"]
        if cls == "anchor":
            budget = class_pools["anchor"] // n_anchor if n_anchor else 0
        else:
            budget = class_pools[cls]
        # The section label counts against the section's budget.
        label = sec["label"] if sec["label"] else None
        budget -= (len(label) + 1) if label else 0
        kept = _fill_section(sec["lines"], budget if budget > 0 else 0,
                             from_end=(cls == "tail"))
        if not kept:
            continue
        out_parts.extend(([label] if label else []) + kept)

    text_parts = ([fixed] if fixed else []) + out_parts
    text = "\n".join(text_parts)
    truncated = True
    if len(text) > cap - len(_TRUNCATION_MARKER) - 1:
        keep = cap - len(_TRUNCATION_MARKER) - 1
        text = text[:keep]
    text = text + "\n" + _TRUNCATION_MARKER
    return text, truncated


def _build_excerpt_impl(raw: str, kind: str, cfg: dict):
    """Core excerpt builder. Returns (text, {"truncated": bool, "cap": int}).

    The excerpt is an EXACT-budget block: output length (including any
    truncation marker) never exceeds ``excerpt_max_chars``."""
    cap = _normalized_cap(cfg)

    if kind == "binary":
        text, truncated = _assemble_excerpt([{
            "prio": "header", "label": None,
            "lines": [f"[binary data, {len(raw) if isinstance(raw, bytes) else 'unknown'} bytes]"],
        }], cap)
        return text, {"truncated": truncated, "cap": cap}

    if not isinstance(raw, str):
        text, truncated = _assemble_excerpt([{
            "prio": "header", "label": None,
            "lines": [f"[{kind}: {(str(raw)[:200])}]"],
        }], cap)
        return text, {"truncated": truncated, "cap": cap}

    lines = raw.splitlines()
    hl = cfg.get("head_lines", 40)
    tl = cfg.get("tail_lines", 15)
    sections = []
    header = None

    if kind == "json":
        try:
            obj = json.loads(raw)
            head_n = cfg.get("json_head_items", 5)
            tail_n = cfg.get("json_tail_items", 2)
            head = json.dumps(_safe_json_head(obj, head_n), indent=2,
                              ensure_ascii=False)
            tail = json.dumps(_safe_json_tail(obj, tail_n), indent=2,
                              ensure_ascii=False)
            json_sections = [
                {"prio": "head", "label": "--- head ---",
                 "lines": [l[:LINE_CAP] for l in head.splitlines()]},
            ]
            # Show the tail only when the container has more items than the
            # head and tail together already cover.
            n_items = len(obj) if isinstance(obj, (list, dict)) else 0
            if n_items > head_n + tail_n:
                json_sections.append(
                    {"prio": "tail", "label": "--- tail ---",
                     "lines": [l[:LINE_CAP] for l in tail.splitlines()]})
            header = f"[JSON excerpt: {kind_desc(raw, obj)}]"
            # Extend LAST so any failure above leaves sections untouched
            # and the text fall-through cannot double-emit JSON parts.
            sections.extend(json_sections)
        except Exception:
            header = None  # fall through to text handler

    if header is None:
        # text / code / html (or JSON that failed to parse)
        kind_label = {"text": "text", "code": "code", "html": "HTML"}.get(kind, kind)
        header = f"[{kind_label} excerpt]"
        if len(lines) <= hl + tl:
            sections.append({"prio": "head", "label": None,
                             "lines": [l[:LINE_CAP] for l in lines]})
        else:
            sections.append({"prio": "head", "label": "--- head ---",
                             "lines": [l[:LINE_CAP] for l in lines[:hl]]})
            sections.append({"prio": "tail", "label": "--- tail ---",
                             "lines": [l[:LINE_CAP] for l in lines[-tl:]]})

    sections.insert(0, {"prio": "header", "label": None, "lines": [header]})

    # Error lines — use anchor_patterns if present, else legacy flat list.
    if cfg.get("anchor_patterns"):
        for label, matched in _anchor_sections(lines, cfg):
            sections.append({"prio": "anchor", "label": label, "lines": matched})
    else:
        errs = _error_lines(lines, cfg.get("error_line_patterns", []))
        if errs:
            sections.append({"prio": "anchor", "label": "--- error lines ---",
                             "lines": errs})

    text, truncated = _assemble_excerpt(sections, cap)
    return text, {"truncated": truncated, "cap": cap}


def build_excerpt(raw: str, kind: str, cfg: dict) -> str:
    """Build excerpt from raw + kind. cfg keys: head_lines, tail_lines,
    json_head_items, json_tail_items, error_line_patterns, anchor_patterns,
    excerpt_max_chars.

    The excerpt is an EXACT-budget block: output length (including any
    truncation marker) never exceeds ``excerpt_max_chars``."""
    return _build_excerpt_impl(raw, kind, cfg)[0]


def build_excerpt_meta(raw: str, kind: str, cfg: dict):
    """As ``build_excerpt`` but returns ``(excerpt, meta_dict)`` where meta
    carries ``truncated`` and the effective ``cap`` — so callers like the
    rescue handle can describe the preview honestly."""
    return _build_excerpt_impl(raw, kind, cfg)


def _error_lines(lines, patterns):
    if not patterns:
        return []
    lp = _err_re(tuple(patterns))
    return [l[:500] for l in lines if lp.search(l)][:20]


def _anchor_sections(lines: list[str], cfg: dict):
    """Yield (label, matched_lines) per anchor category.

    Reads ``anchor_patterns`` from config (a dict of category→pattern-list).
    Lines already appearing in the head/tail/error sections are NOT
    deduplicated — the model benefits from seeing them in context.
    """
    anchor_cfg = cfg.get("anchor_patterns")
    if not anchor_cfg or not isinstance(anchor_cfg, dict):
        return
    seen = set()
    category_labels = {
        "error": "error lines",
        "decision": "decision anchors",
        "action": "action anchors",
        "value": "value anchors",
    }
    for category, patterns in anchor_cfg.items():
        if not patterns or not isinstance(patterns, list):
            continue
        matched = _anchor_lines_for_category(lines, patterns)
        if matched:
            label = f"--- {category_labels.get(category, category + ' anchors')} ---"
            lines_out = []
            for m in matched[:10]:
                trimmed = m[:500]
                if trimmed in seen:
                    continue
                seen.add(trimmed)
                lines_out.append(trimmed)
            if lines_out:
                yield label, lines_out


@functools.lru_cache(maxsize=16)
def _anchor_re(patterns: tuple):
    return re.compile("|".join(re.escape(p) for p in patterns), re.I)


def _anchor_lines_for_category(lines: list[str], patterns: list[str]) -> list[str]:
    """Return lines matching any pattern in the category, up to 10."""
    if not patterns:
        return []
    lp = _anchor_re(tuple(patterns))
    return [l for l in lines if lp.search(l)][:10]


def kind_desc(raw, obj):
    if isinstance(obj, list):
        return f"array[{len(obj)}]"
    if isinstance(obj, dict):
        return f"object, {len(obj)} keys"
    return type(obj).__name__