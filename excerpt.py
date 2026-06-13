"""Content type detection and excerpt builders for Toolaria."""
import functools
import json
import re


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


def build_excerpt(raw: str, kind: str, cfg: dict):
    """Build excerpt from raw + kind. cfg keys: head_lines, tail_lines,
    json_head_items, json_tail_items, error_line_patterns, anchor_patterns."""
    if kind == "binary":
        return f"[binary data, {len(raw) if isinstance(raw, bytes) else 'unknown'} bytes]"

    if not isinstance(raw, str):
        return f"[{kind}: {(str(raw)[:200])}]"

    lines = raw.splitlines()
    hl = cfg.get("head_lines", 40)
    tl = cfg.get("tail_lines", 15)

    if kind == "json":
        try:
            obj = json.loads(raw)
            head_n = cfg.get("json_head_items", 5)
            tail_n = cfg.get("json_tail_items", 2)
            head = json.dumps(_safe_json_head(obj, head_n), indent=2,
                              ensure_ascii=False)
            tail = json.dumps(_safe_json_tail(obj, tail_n), indent=2,
                              ensure_ascii=False)
            parts = [f"[JSON excerpt: {kind_desc(raw, obj)}]"]
            parts.append("--- head ---")
            parts.append(head)
            # Show the tail only when the container has more items than the
            # head and tail together already cover.
            n_items = len(obj) if isinstance(obj, (list, dict)) else 0
            if n_items > head_n + tail_n:
                parts.append("--- tail ---")
                parts.append(tail)
            # Error lines — use anchor_patterns if present, else legacy flat list.
            if cfg.get("anchor_patterns"):
                _append_anchor_lines(parts, lines, cfg)
            else:
                errs = _error_lines(lines, cfg.get("error_line_patterns", []))
                if errs:
                    parts.append("--- error lines ---")
                    parts.extend(errs[:10])
            return "\n".join(parts)
        except Exception:
            pass  # fall through to text handler

    # text / code / html
    kind_label = {"text": "text", "code": "code", "html": "HTML"}.get(kind, kind)
    parts = [f"[{kind_label} excerpt]"]
    if len(lines) <= hl + tl:
        parts.append(raw[:cfg.get("excerpt_max_chars", 8000)])
    else:
        parts.append("--- head ---")
        parts.append("\n".join(lines[:hl]))
        parts.append("--- tail ---")
        parts.append("\n".join(lines[-tl:]))
    # Error/decision/action/value lines — use anchor_patterns if present,
    # else legacy flat error_line_patterns for backward compatibility.
    if cfg.get("anchor_patterns"):
        _append_anchor_lines(parts, lines, cfg)
    else:
        errs = _error_lines(lines, cfg.get("error_line_patterns", []))
        if errs:
            parts.append("--- error lines ---")
            parts.extend(errs[:10])
    return "\n".join(parts)


def _error_lines(lines, patterns):
    if not patterns:
        return []
    lp = _err_re(tuple(patterns))
    return [l[:500] for l in lines if lp.search(l)][:20]


def _append_anchor_lines(parts: list, lines: list[str], cfg: dict) -> None:
    """Append lines matching anchor pattern categories to the excerpt.

    Reads ``anchor_patterns`` from config (a dict of category→pattern-list).
    Each category gets its own section header. Lines already appearing in
    the head/tail/error sections are NOT deduplicated — the model benefits
    from seeing them in context.
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
            label = category_labels.get(category, f"{category} anchors")
            parts.append(f"--- {label} ---")
            for m in matched:
                if m not in seen:
                    parts.append(m[:500])
                    seen.add(m)


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
