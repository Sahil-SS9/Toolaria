"""Structural index for rescued blobs: a cheap, deterministic table of
contents the model navigates instead of greping blind.

Built at rescue time (no embeddings, no network). Per content kind:
  - JSON: a path/schema map of the top levels, with numeric column stats for
    arrays of objects (so a 10k-row result shows ranges, not rows).
  - HTML: the heading hierarchy (h1-h6) with line offsets.
  - text/log: clusters of error-like lines with counts and first line numbers.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right


def build_outline(raw: str, kind: str, cfg: dict | None = None) -> dict:
    """Return a structural outline dict for *raw* of detected *kind*."""
    cfg = cfg or {}
    try:
        if kind == "json":
            return _json_outline(raw, cfg)
        if kind == "html":
            return _html_outline(raw, cfg)
        return _text_outline(raw, cfg)
    except Exception as exc:  # never let indexing break rescue
        return {"kind": kind, "error": f"outline failed: {exc}"}


def render_outline(outline: dict) -> str:
    """Render an outline dict as compact model-facing text."""
    kind = outline.get("kind", "?")
    if "error" in outline:
        return f"[outline unavailable: {outline['error']}]"
    if kind == "json":
        return _render_json(outline)
    if kind == "html":
        return _render_html(outline)
    return _render_text(outline)


# ── JSON ──────────────────────────────────────────────────────────────────


def _json_outline(raw: str, cfg: dict) -> dict:
    obj = json.loads(raw)
    max_keys = cfg.get("outline_json_max_keys", 40)
    if isinstance(obj, list):
        return {
            "kind": "json",
            "root": "array",
            "length": len(obj),
            "element": _describe_elements(obj, max_keys),
        }
    if isinstance(obj, dict):
        return {
            "kind": "json",
            "root": "object",
            "keys": _describe_object(obj, max_keys),
        }
    return {"kind": "json", "root": type(obj).__name__}


def _describe_object(obj: dict, max_keys: int) -> list[dict]:
    out = []
    for k, v in list(obj.items())[:max_keys]:
        out.append({"key": str(k), "type": _type_name(v), "summary": _value_summary(v)})
    if len(obj) > max_keys:
        out.append({"key": f"... {len(obj) - max_keys} more keys", "type": "", "summary": ""})
    return out


def _describe_elements(arr: list, max_keys: int) -> dict:
    """For an array of objects, summarise the common columns and numeric stats."""
    if not arr:
        return {"type": "empty"}
    sample = arr[:1000]
    if all(isinstance(e, dict) for e in sample):
        cols: dict[str, dict] = {}
        for e in sample:
            for k, v in e.items():
                col = cols.setdefault(str(k), {"types": set(), "nums": []})
                col["types"].add(_type_name(v))
                # bool is a subclass of int; keep it out of numeric stats.
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    col["nums"].append(v)
        columns = []
        for k, col in list(cols.items())[:max_keys]:
            entry = {"column": k, "types": sorted(col["types"])}
            nums = col["nums"]
            if nums:
                entry["stats"] = {
                    "count": len(nums),
                    "min": min(nums),
                    "max": max(nums),
                    "mean": round(sum(nums) / len(nums), 4),
                }
            columns.append(entry)
        return {"type": "object", "columns": columns, "sampled": len(sample)}
    return {"type": _type_name(sample[0]), "example": _value_summary(sample[0])}


def _render_json(outline: dict) -> str:
    lines = ["[JSON outline]"]
    if outline["root"] == "array":
        el = outline.get("element", {})
        lines.append(f"root: array, length {outline.get('length', '?')}")
        if el.get("type") == "object":
            lines.append(f"elements: objects (sampled {el.get('sampled')}):")
            for c in el.get("columns", []):
                stats = c.get("stats")
                s = f"  - {c['column']} ({'/'.join(c['types'])})"
                if stats:
                    s += (f"  [n={stats['count']} min={stats['min']} "
                          f"max={stats['max']} mean={stats['mean']}]")
                lines.append(s)
        else:
            lines.append(f"elements: {el.get('type')}  e.g. {el.get('example', '')}")
    elif outline["root"] == "object":
        lines.append("root: object")
        for k in outline.get("keys", []):
            lines.append(f"  - {k['key']}: {k['type']}  {k['summary']}".rstrip())
    else:
        lines.append(f"root: {outline['root']}")
    return "\n".join(lines)


# ── HTML ──────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"<(h[1-6])\b[^>]*>(.*?)</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _html_outline(raw: str, cfg: dict) -> dict:
    max_h = cfg.get("outline_html_max_headings", 60)
    headings = []
    # Map character offset to line number cheaply.
    line_starts = _line_starts(raw)
    for m in _HEADING_RE.finditer(raw):
        level = int(m.group(1)[1])
        text = _TAG_RE.sub("", m.group(2)).strip()
        if not text:
            continue
        line = _offset_to_line(line_starts, m.start())
        headings.append({"level": level, "text": text[:120], "line": line})
        if len(headings) >= max_h:
            break
    return {"kind": "html", "headings": headings}


def _render_html(outline: dict) -> str:
    headings = outline.get("headings", [])
    if not headings:
        return "[HTML outline: no headings found]"
    lines = ["[HTML outline] (line numbers for rescuer_fetch range mode)"]
    for h in headings:
        lines.append(f"  {'  ' * (h['level'] - 1)}h{h['level']} L{h['line']}: {h['text']}")
    return "\n".join(lines)


# ── text / log ─────────────────────────────────────────────────────────────

_ERR_RE = re.compile(
    r"\b(error|err|exception|traceback|fatal|fail(?:ed|ure)?|warn(?:ing)?|"
    r"critical|panic|denied|refused|timeout|exit code [1-9])\b",
    re.I,
)


def _text_outline(raw: str, cfg: dict) -> dict:
    max_clusters = cfg.get("outline_text_max_clusters", 30)
    lines = raw.splitlines()
    clusters: dict[str, dict] = {}
    for n, line in enumerate(lines):
        m = _ERR_RE.search(line)
        if not m:
            continue
        key = m.group(1).lower()
        c = clusters.setdefault(key, {"count": 0, "first_line": n, "sample": line[:160]})
        c["count"] += 1
    ranked = sorted(clusters.items(), key=lambda kv: -kv[1]["count"])[:max_clusters]
    return {
        "kind": "text",
        "total_lines": len(lines),
        "clusters": [{"signal": k, **v} for k, v in ranked],
    }


def _render_text(outline: dict) -> str:
    clusters = outline.get("clusters", [])
    lines = [f"[text outline] {outline.get('total_lines', '?')} lines"]
    if not clusters:
        lines.append("  no error/warning signals detected")
        return "\n".join(lines)
    lines.append("  signals (line numbers for rescuer_fetch range mode):")
    for c in clusters:
        lines.append(f"  - {c['signal']} x{c['count']} (first L{c['first_line']}): "
                     f"{c['sample']}")
    return "\n".join(lines)


# ── helpers ────────────────────────────────────────────────────────────────


def _type_name(v) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return f"array[{len(v)}]"
    if isinstance(v, dict):
        return f"object[{len(v)}]"
    if v is None:
        return "null"
    return type(v).__name__


def _value_summary(v) -> str:
    if isinstance(v, (list, dict)):
        return ""
    s = str(v)
    return f"= {s[:60]}" if len(s) <= 60 else f"= {s[:57]}..."


def _line_starts(raw: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(raw):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _offset_to_line(line_starts: list[int], offset: int) -> int:
    # 0-based line number of the greatest start <= offset
    return bisect_right(line_starts, offset) - 1
