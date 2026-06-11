"""Content type detection and excerpt builders for Toolaria."""
import json
import re

_ERROR_RE = None

def _err_re(patterns):
    global _ERROR_RE
    if _ERROR_RE is None:
        _ERROR_RE = re.compile("|".join(re.escape(p) for p in patterns), re.I)
    return _ERROR_RE


def detect_type(raw: str):
    """Return (kind: str, meta: str) — detects JSON, HTML, code, text, binary."""
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
    json_head_items, json_tail_items, error_line_patterns."""
    if kind == "binary":
        return f"[binary data — {len(raw) if isinstance(raw, bytes) else 'unknown'} bytes]"

    if not isinstance(raw, str):
        return f"[{kind} — {(str(raw)[:200])}]"

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
            parts = [f"[JSON excerpt — {kind_desc(raw, obj)}]"]
            parts.append("--- head ---")
            parts.append(head)
            if len(lines) > head_n * 4:
                parts.append("--- tail ---")
                parts.append(tail)
            # Error lines
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
        parts.append(raw[:8000])
    else:
        parts.append("--- head ---")
        parts.append("\n".join(lines[:hl]))
        parts.append("--- tail ---")
        parts.append("\n".join(lines[-tl:]))
    errs = _error_lines(lines, cfg.get("error_line_patterns", []))
    if errs:
        parts.append("--- error lines ---")
        parts.extend(errs[:10])
    return "\n".join(parts)


def _error_lines(lines, patterns):
    if not patterns:
        return []
    lp = _err_re(patterns)
    return [l[:500] for l in lines if lp.search(l)][:20]


def kind_desc(raw, obj):
    if isinstance(obj, list):
        return f"array[{len(obj)}]"
    if isinstance(obj, dict):
        return f"object, {len(obj)} keys"
    return type(obj).__name__
