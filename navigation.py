"""Derived line navigation; raw content-addressed blobs are never rewritten."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

NAVIGATION_VERSION = 1
_MAX_JSON_INDENT_DEPTH = 64


@dataclass(frozen=True)
class NavigationView:
    text: str
    kind: str = "raw"

    @property
    def key(self) -> str:
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        return f"v{NAVIGATION_VERSION}:{self.kind}:{digest}"

    @property
    def line_count(self) -> int:
        return len(self.text.splitlines())

    @property
    def notice(self) -> str:
        if self.kind == "raw":
            return ""
        return (f"[navigation: {self.kind}; {self.line_count} lines; "
                "line numbers refer to this view]\n")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate wrapper key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("non-finite JSON number")


def navigation_view(raw: str, tool_name: str = "") -> NavigationView:
    """Unwrap one known MCP result envelope, then format single-line JSON.

    Only named MCP tools with exactly one string-valued ``result`` field
    qualify. Extra metadata, duplicate keys, other tools and nested envelopes
    are never silently discarded. Existing multi-line text/JSON keeps its
    coordinates. Invalid/deep JSON falls back to its original representation.
    """
    text = raw
    wrapped = False
    if tool_name.startswith("mcp_") and raw.lstrip().startswith("{"):
        try:
            envelope = json.loads(raw, object_pairs_hook=_unique_object,
                                  parse_constant=_nonfinite)
            if (isinstance(envelope, dict) and set(envelope) == {"result"}
                    and isinstance(envelope["result"], str)):
                # A JSON escape may decode to a lone surrogate. Keep the
                # original envelope rather than materialise invalid UTF-8.
                envelope["result"].encode("utf-8")
                text = envelope["result"]
                wrapped = True
        except (ValueError, RecursionError):
            pass

    is_json = False
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            obj = json.loads(text, parse_constant=_nonfinite)
            is_json = isinstance(obj, (dict, list))
        except (ValueError, RecursionError):
            pass
    if is_json and "\n" not in stripped and "\r" not in stripped:
        try:
            text = _pretty_json(stripped)
        except ValueError:
            # Bounding indentation also bounds whitespace amplification;
            # retain the usable original/decoded view for deep payloads.
            pass
    if wrapped:
        kind = "mcp-json" if is_json else "mcp-text"
    elif text != raw:
        kind = "json"
    else:
        kind = "raw"
    return NavigationView(text, kind)


def _pretty_json(text: str) -> str:
    """Insert whitespace only, preserving number lexemes and duplicate keys.

    json.loads validates first, but json.dumps would rewrite numeric values
    and collapse repeated keys. This linear scan keeps every non-whitespace
    token, including the spelling/escapes of string contents, unchanged.
    """
    pieces = []
    depth = 0
    quoted = escaped = False
    previous = ""
    for i, char in enumerate(text):
        if quoted:
            pieces.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            previous = char
            continue
        if char.isspace():
            continue
        if char == '"':
            quoted = True
            pieces.append(char)
        elif char in "{[":
            pieces.append(char)
            depth += 1
            if depth > _MAX_JSON_INDENT_DEPTH:
                raise ValueError("JSON navigation nesting limit exceeded")
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] not in "}]":
                pieces.append("\n" + "  " * depth)
        elif char in "}]":
            depth -= 1
            if previous not in "{[":
                pieces.append("\n" + "  " * depth)
            pieces.append(char)
        elif char == ",":
            pieces.append(",\n" + "  " * depth)
        elif char == ":":
            pieces.append(": ")
        else:
            pieces.append(char)
        previous = char
    return "".join(pieces)
