# Toolaria Design Document

This is the spill-to-disk / context-offloading pattern (see the README's Prior
art section); the design notes here cover Toolaria's specific choices, not the
novelty of the idea.

## Problem

Hermes Agent truncates tool outputs by size (20K bytes, 2000 lines) before
the messages array, but MCP and web tool outputs bypass this truncation and
can inject 50-200K chars into the context. This consumes token budget on
bloat the model doesn't need.

## Solution

Intercept oversized results at the `transform_tool_result` hook, store the
full output to disk, return a compact excerpt + rescue handle, and let the
model retrieve specific slices on demand.

## Design constraints

1. **Zero model behaviour change.** The model sees a natural text excerpt
   plus a structured handle block. It can ignore the handle and work with
   the excerpt (best-effort) or use `rescuer_fetch` for precise access.
2. **Deterministic.** No LLM judgement in the rescue path. All decisions
   (size threshold, tool allow-list, excerpt structure) are config-driven.
3. **Fail-safe storage.** If the blob store is unavailable or a write fails,
   the original result passes through untouched. A handle is never emitted
   for content that is not durably on disk; a handle that cannot be fetched
   is worse than no rescue.
4. **Fail-open interception.** When the registry import fails, `_is_rescuable`
   errs towards rescuing, but the unconditional excludes list still protects
   critical tools.
5. **Graceful expiry.** TTL sweep (72h default) and size cap (500MB default)
   bound disk growth. An expired blob leaves a tombstone so a stale handle
   degrades into re-run guidance rather than a bare error.

## Excluded tools

The following are never intercepted regardless of config:

- `delegate_task`, `session_search`: bounded results by design
- `cronjob`, `skill_view`, `skill_manage`, `skill_request`: system tools
- `kanban_create`, `open_kanban`: board operations
- `clarify`, `memory`: interactive tools with small outputs

## Fail-open safety in production

The `_is_rescuable()` check returns `True` when the Hermes tool registry
import fails. This is correct: rescuing too aggressively (duplicate handle,
small overhead) is safer than missing a 200K result that floods context.

The unconditional excludes list (`_UNCONDITIONAL_EXCLUDES`) ensures that
critical system tools are never intercepted even under registry failure.
This guard runs before the config-based check, so it cannot be overridden
by user config.

## Scalability

- Blobs: one file per unique SHA256. Identical outputs from different tools
  share the same blob, but each session tracks its own reference.
- Grep: with the `regex` package every pattern runs under a per-search
  timeout, the only reliable defence against catastrophic backtracking
  (a between-lines timeout cannot interrupt a single C-level `re.search`).
  Without `regex`, grep falls back to literal substring search and refuses
  metacharacter patterns. Pattern length and per-line slice are also capped.
- Sweep: `lazy_sweep()` runs on session start and end, tombstoning expired
  blobs and deleting blob files once no live reference remains.
