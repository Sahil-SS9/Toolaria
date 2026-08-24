"""Sensitivity labels for rescued tool results.

Phase 2 T2.1 / T2.4. Labels classify blobs by how sensitive their content
is, so the audit script (T2.2) can report the value-flow and the
enforcement tier (T2.3) can gate credential-grade blobs.

Labels (in ascending sensitivity):
  - public      — default; web/browser, anything we are not sure about.
  - internal    — repo / org content. Not exported, not destructive.
  - personal    — mail / direct messages. Could carry secrets in the body.
  - credential  — explicit allowlist gate. Full reads refused; only
                  masked slices (range/grep/search) and allowlisted
                  destinations receive content. 24h hard TTL.

Two entry points:
  - label_for_tool(tool_name, cfg): operator-configured tool→label map
    unioned with built-in defaults.
  - label_for_args(args, base_label, cfg): scans args for secret-shaped
    values (sk-…, bearer …, BEGIN PRIVATE KEY …) and upgrades the label.
    Never downgrades.

Config:
  - sensitivity_tool_labels: {tool_name: label} — operator overrides.
    Merged on top of the built-in defaults; invalid label names raise
    ValueError at call time so a broken config fails loud.

The classification is deterministic and config-driven. There is no
LLM-in-path and no per-call network — the value of automation is to be
correct under every operator's config, not to be clever.
"""
from __future__ import annotations

import re
from typing import Any


VALID_LABELS = frozenset({"credential", "personal", "internal", "public"})


# Built-in defaults: ships out of the box so a fresh install classifies
# common tool classes without any operator config.
#
# - mail / peer-messaging → personal (may carry secrets in body)
# - web / browser → public (false-positive budget depends on this — a
#   normal browser navigate should not gate behind credential controls)
_BUILTIN_TOOL_LABELS: dict[str, str] = {
    # Mail-send class (rescue-able per Phase 0 T0.3)
    "send_email": "personal",
    "send_mail": "personal",
    "post_email": "personal",
    "compose_email": "personal",
    "mail_send": "personal",
    # Peer-messaging class
    "peer_send_message": "personal",
    "peer_broadcast": "personal",
    # Web / browser class — public by default
    "web_extract": "public",
    "web_search": "public",
    "browser_navigate": "public",
    "browser_snapshot": "public",
    "browser_console": "public",
    "browser_get_images": "public",
}


# Secondary signals for label_for_args (T2.4). Any of these in an args
# value upgrades the label to credential — same patterns already used
# by T1.3 args-redaction, so a leak-detection rule that fires on capture
# also fires on classification.
_LABEL_UPGRADE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{8,}"),            # sk-XXXXXXXXX
    re.compile(r"bearer\s+\S+", re.IGNORECASE),    # Bearer <token>
    re.compile(r"BEGIN PRIVATE KEY"),             # PEM private key marker
)


def _parse_tool_label_map(raw: Any) -> dict[str, str]:
    """Validate an operator-supplied sensitivity_tool_labels map.

    A malformed operator config raises ``ValueError`` so a broken
    config fails loud at register time (Phase 3 FIX-4). All offending
    entries are accumulated and reported in a single message so the
    operator sees the full list of typos, not one-at-a-time.

    Empty / None is a valid no-op.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            "sensitivity_tool_labels must be a dict mapping tool name "
            f"to label, got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    errors: list[str] = []
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            errors.append(f"entry must be string→string; got {k!r}={v!r}")
            continue
        if v not in VALID_LABELS:
            errors.append(
                f"sensitivity_tool_labels[{k!r}]={v!r} is not a valid "
                f"label; choose from {sorted(VALID_LABELS)}"
            )
            continue
        out[k] = v
    if errors:
        # All bad entries in one message so the operator fixes them in
        # one pass instead of N register-reload cycles.
        raise ValueError(
            "sensitivity_tool_labels has invalid entries: "
            + "; ".join(errors)
        )
    return out


def label_for_tool(tool_name: Any, cfg: dict) -> str:
    """Return the sensitivity label for *tool_name* under *cfg*.

    Resolution order:
      1. cfg["sensitivity_tool_labels"][tool_name]  (operator override)
      2. _BUILTIN_TOOL_LABELS[tool_name]          (shipped defaults)
      3. "public"                                 (safe default)

    An empty / non-string tool_name falls through to "public" — a
    defensive default that never raises on weird input.
    """
    if not isinstance(tool_name, str) or not tool_name:
        return "public"
    try:
        overrides = _parse_tool_label_map(cfg.get("sensitivity_tool_labels"))
    except ValueError:
        # Re-raise on parse error so a broken operator config is loud.
        raise
    if tool_name in overrides:
        return overrides[tool_name]
    return _BUILTIN_TOOL_LABELS.get(tool_name, "public")


def _args_look_credential_shaped(args: Any) -> bool:
    """Recursively scan args for secret-shaped strings.

    Matches:
      - sk-XXXXXXXXX (8+ alphanumerics after the prefix)
      - bearer <token>
      - BEGIN PRIVATE KEY (PEM marker)

    Non-string scalars (numbers, bools, None) cannot match; they are
    skipped. The scan is recursive so a nested {"headers":
    {"Authorization": "Bearer ..."}} is caught.
    """
    if args is None:
        return False
    if isinstance(args, str):
        return any(p.search(args) for p in _LABEL_UPGRADE_PATTERNS)
    if isinstance(args, dict):
        return any(_args_look_credential_shaped(v) for v in args.values())
    if isinstance(args, (list, tuple)):
        return any(_args_look_credential_shaped(v) for v in args)
    return False


def label_for_args(args: Any, base_label: str, cfg: dict) -> str:
    """Upgrade *base_label* when *args* contain a credential-shaped value.

    Never downgrades: a `personal` base stays `personal` even if the args
    look clean (so an operator who manually tagged something is not
    overridden by the scan). `credential` is the ceiling.

    Pure function: *args* is never mutated. Tested explicitly.
    """
    if base_label not in VALID_LABELS:
        # Defensive: a malformed base label is treated like public. The
        # rising-tide pattern (operational tool raises on bad config)
        # lives in label_for_tool; here we want to be permissive so a
        # weird caller doesn't crash the rescue path.
        base_label = "public"
    if base_label == "credential":
        return "credential"
    if _args_look_credential_shaped(args):
        return "credential"
    return base_label
