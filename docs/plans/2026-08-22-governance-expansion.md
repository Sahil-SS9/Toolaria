# Toolaria Governance Expansion — Implementation Plan

> **Status:** APPROVED by Sahil 2026-08-22. Decisions locked: (1) proceed; (2) public promotion (T5.5) additionally gated on full build + thorough test completion, not merely feature-complete; (3) simplified capability/direction summary delivered in chat.
> **Evidence base:** Triple-verified review 2026-08-22 (direct source reads, security threat-model review, independent code review, research validation). Zero unresolved contradictions.

**Goal:** Evolve Toolaria from a size-only context-rescue plugin into the fleet's data-governance chokepoint: hardened defaults, measured behaviour, sensitivity-aware value-flow auditing, and risk-tiered pre-execution entity binding.

**Architecture:** All work lands in `/home/kensei/repos/hermes-toolaria` (canonical, main @ 87f3935). Layered additions only — no rewrite of the rescue path. Classification is deterministic and config-driven (DESIGN.md constraints hold throughout). Enforcement ships measurement-first: every policy change is judged by the instrumentation shipped one tier earlier.

**Tech stack:** Python 3.12 stdlib-first; optional deps stay optional (`sentence-transformers`, `regex`, `cryptography`). No new mandatory runtime dependencies.

---

## Grilled decisions (locked during review)

| # | Decision | Choice | Reason |
|---|----------|--------|--------|
| D1 | Enforcement posture | Hybrid tiered: audit-everything default; hard enforcement only for credential-grade | Entity-binding paper: enforcement costs completion rate; earn enforcement with data |
| D2 | Redaction point | Read-side masking only | Scrub-before-put changes SHA256 → breaks addressing/dedup |
| D3 | Label persistence | Labels carried through tombstones AND respected by size-cap sweeps | Code review found both paths drop custom fields / ignore labels |
| D4 | Passref destination policy | Separate `passref_external_destinations` list, never merged into `exclude_tools` | Merging silently stops rescuing those tools' oversized results (dead handles) |
| D5 | Reacquisition probe | Phase 1 timestamp-proxy metrics; phase 2 turn-position capture behind a flag | No turn counter exists in plugin or forwarded by host today |
| D6 | Args provenance | Capture with size caps + secret-pattern redaction of arg values | Args introduce a new secrets-at-rest class inside the store itself |
| D7 | Entity governor scope | Consequential tool classes only (send/write/config-change), bypass allowlist for low-risk | Paper: deferral cost must be bounded and measured |
| D8 | Encryption | Last, credential-labels only, Fernet key in `~/.hermes/secrets/` (0600) | Correct threat boundary for single-user VPS: other users + backup theft, not root/agent |

## Open items flagged for follow-up (not blocking)

- Host-side ask: forward turn/position counter in hook kwargs if available (improves probe phase 2 fidelity).
- Dezzy input on clarification UX wording for entity-governor deferrals (before Tier 3 enforcement mode).
- Corrective notes to `docs/wiki/_meta/paper-mashups.md` (SelfCompact/LRE misattribution) routed to Light.

---

## Build phases

### Phase 0 — Hardening (small, ships first, independent)

| ID | Task | Files | Key detail |
|----|------|-------|-----------|
| T0.1 | Explicit permissions | `blobstore.py` (`__init__`, `put`, `_write_idx_file`) | Set dirs 0700 / files 0600 explicitly (never umask-reliant); best-effort chmod migration of existing store on init |
| T0.2 | Close fail-open catchment | `__init__.py` `_UNCONDITIONAL_EXCLUDES` | Add terminal/shell/exec-class + write-class tools so broken-registry fail-open never rescues token-bearing shell output |
| T0.3 | External-destination deny list | `passref.py`, `config.yaml` | New `passref_external_destinations` key (mail-send/social-post/webhook/peer-messaging classes); expansion into these returns an honest marker instead of content; precedence vs allowlist/denylist documented and tested |
| T0.4 | Fail-loud middleware fallback | `__init__.py` `register()` (~140) | Else-branch warning when host lacks middleware; suppress the `tla:<id>` instruction line in handles when passref is unavailable/dead |
| T0.5 | Surface chain mode | `__init__.py` tool schema, `blobstore.py` docstrings, README | Add `chain` to enum + docs; fix ARCHITECTURE.md session-filename drift |

**Exit criteria:** full suite green; live store perms verified 0700/0600; dogfood cron clean 48h.

### Phase 1 — Instrumentation (measurement before policy)

| ID | Task | Files | Key detail |
|----|------|-------|-----------|
| T1.1 | Probe phase 1: timestamp metrics | `blobstore.py` (`_refresh_blob`, flush), new `reporting/reacquisition_report.py` | Persist `first_fetch_ts` + fetch-count aggregates; offline script reports put→first-fetch latency (p50/p95) and fetches-per-blob histogram over session indexes; graceful on missing fields |
| T1.2 | Probe phase 2: sequence capture | `blobstore.py`, `reporting/` | Config-gated JSONL sidecar of per-fetch events `{ts, sid, bid, turn?}`; survives restart; report adds turns-between-rescue-and-recovery when turn data present, degrades to wall-clock proxy otherwise |
| T1.3 | Metadata v2: args provenance | `__init__.py` (`_rescue`), `blobstore.py` (`put`, index schema) | Capture redacted args snapshot (size-capped; secret-looking keys/values pattern-masked) into index entry |
| T1.4 | Integrity verification on fetch | `blobstore.py` `fetch()` (~353) | Full-SHA256 compare after read; mismatch → deterministic refusal marker + integrity event; hashing perf-guarded for multi-MB blobs |
| T1.5 | Expansion audit ledger | `passref.py` (`_sub`, `_tool_request`) | INFO-level JSONL: `{blob_id, dst_tool, chars, session, ts, decision}` per expansion incl. denials; replaces debug-only logging |

**Exit criteria:** report script runs against live historical data; ledger visible in dogfood check; baseline reacquisition numbers recorded in `.hermes/plans/`.

### Phase 2 — Data-governance core

| ID | Task | Files | Key detail |
|----|------|-------|-----------|
| T2.1 | Sensitivity labels | `config.yaml` (`sensitivity_tool_labels`, `sensitivity_patterns`), `blobstore.py` (`put`, tombstone conversion ~585/647, `_sweep_by_size`) | Tool→label map (mail→personal, github→internal, web→public default); optional content-regex secondary signal (audit-only); labels survive tombstoning; size-cap sweep respects label TTL floor |
| T2.2 | Value-flow auditor — audit mode | `passref.py`, `/rescuer` status cmd | Every labelled-content expansion logged with label + destination class surfaced in `/rescuer`; no behaviour change |
| T2.3 | Value-flow auditor — enforcement tier | `passref.py`, `blobstore.py` sweeps | Credential-grade only: no expand without explicit allowlist; masked range/grep/search snippets; `full` refused; 24h hard TTL overriding hot-blob pinning; staged behind `enforcement_mode` flag, default off until audit data reviewed |
| T2.4 | Label-query hook | new `labels.py` public function | `label_for_args(args)` exposed for host dispatch-layer consultation (covers manual-copy path); thin, read-only |

**Exit criteria:** 48h audit-mode run shows real flow data; enforcement flag flipped only after Sahil reviews the audit summary.

### Phase 3 — Pre-execution entity governor

| ID | Task | Files | Key detail |
|----|------|-------|-----------|
| T3.1 | Hook integration | new `governor.py`, `__init__.py register()` | Register blocking `pre_tool_call` (host contract verified: block/approve/modify, single-fire); audit-log-only first |
| T3.2 | Risk-tier classification | `governor.py`, `config.yaml` | `governor_consequential_tools` (send/write/config-change classes) + `governor_bypass_allowlist` |
| T3.3 | Ambiguity handling | `governor.py` | Entity-reference parse → disambiguation context (session state/Mnemosyne query) → defer-with-clarification on high-risk ambiguity; never silent-block |
| T3.4 | Enforcement gate | same | Flip from audit to enforce only when Phase 1 harness shows deferral/completion cost within agreed bounds; Dezzy-reviewed UX text |

**Exit criteria:** deferral rate + completion impact reported from harness; Sahil approves enforcement flip.

### Phase 4 — Integrity & versioning

| ID | Task | Files | Key detail |
|----|------|-------|-----------|
| T4.1 | Version stamps + currency | `blobstore.py`, `excerpt/handle builder` | Version field per blob; retrieval surfaces conflict markers when a newer version of an instruction-like blob exists (memlock synergy) |
| T4.2 | Encrypted-at-rest tier | `blobstore.py`, `requirements.txt` (optional extra) | Fernet encryption for credential-labelled blobs only; key at `~/.hermes/secrets/toolaria.key` (0600); deletion leaves standard tombstone trail; plaintext-absence invariant tested |

### Phase 5 — Ecosystem (parallel, low coupling)

| ID | Task | Detail |
|----|------|--------|
| T5.1 | Deploy script | `scripts/deploy.sh`: repo→`~/.hermes/plugins/toolaria/` sync + gateway restart, `--dry-run` diff mode |
| T5.2 | README/test invocation | Document `uv run --with pytest --with regex python -m pytest tests/ -q`; contributor quickstart refresh |
| T5.3 | Wiki corrections | Route SelfCompact/LRE misattribution corrections to Light (paper-mashups.md) |
| T5.4 | MCP Pattern Audit | Hand to Wesker as ops deliverable (not plugin scope) |
| T5.5 | Public promotion package | **GATED on Sahil sign-off.** Standalone-repo promotion per Hermes upstream rubric |

---

## Master tracking checklist

Update immediately after each item changes state — same working turn, never batched.

Phase 0 — Hardening
- [ ] T0.1 Explicit 0700/0600 permissions (+ existing-store chmod migration)
- [ ] T0.2 Shell/write-class tools in `_UNCONDITIONAL_EXCLUDES`
- [ ] T0.3 `passref_external_destinations` deny list
- [ ] T0.4 Fail-loud middleware fallback + conditional `tla:` handle text
- [ ] T0.5 Chain mode in schema/docs + ARCHITECTURE.md drift fix
- [ ] P0 exit: suite green, perms verified, 48h dogfood clean

Phase 1 — Instrumentation
- [ ] T1.1 Timestamp-proxy reacquisition metrics + report script
- [ ] T1.2 Sequence capture (turn-position, gated) + degraded-mode reporting
- [ ] T1.3 Args provenance with caps + secret-redaction
- [ ] T1.4 Full-hash integrity verify on fetch
- [ ] T1.5 JSONL expansion audit ledger (INFO)
- [ ] P1 exit: baseline numbers recorded, ledger in dogfood

Phase 2 — Data-governance core
- [ ] T2.1 Sensitivity labels (tombstone-safe, size-sweep-aware)
- [ ] T2.2 Auditor audit-mode + `/rescuer` surfacing
- [ ] T2.3 Credential-grade enforcement (flagged, default off)
- [ ] T2.4 `label_for_args()` dispatch hook
- [ ] P2 exit: 48h audit run reviewed; enforcement flip approved by Sahil

Phase 3 — Pre-execution entity governor
- [ ] T3.1 `pre_tool_call` integration (audit-only first)
- [ ] T3.2 Risk-tier + bypass config
- [ ] T3.3 Ambiguity defer-with-clarification
- [ ] T3.4 Enforcement flip after cost evidence + UX review
- [ ] P3 exit: deferral/completion report accepted by Sahil

Phase 4 — Integrity & versioning
- [ ] T4.1 Version stamps + currency conflict markers
- [ ] T4.2 Credential-tier encryption at rest

Phase 5 — Ecosystem
- [ ] T5.1 Deploy script with dry-run
- [ ] T5.2 README uv test invocation + quickstart refresh
- [ ] T5.3 Wiki corrections routed to Light
- [ ] T5.4 MCP Pattern Audit handed to Wesker
- [ ] T5.5 Public promotion package (SAHIL SIGN-OFF REQUIRED)

---

## Testing requirements

Canonical runner (every gate): `uv run --with pytest --with regex python -m pytest tests/ -q` — expected: all prior tests plus new ones pass. TDD discipline: every code task starts RED (failing test written and witnessed before implementation). No test may depend on network, real Mnemosyne, or live store contents; fixtures only.

Global regression gates (run at every phase exit):
- G1 Full suite green, including all pre-existing behavioural tests unchanged.
- G2 E2E rescue→fetch→passref happy path still byte-identical for unlabelled/public blobs (zero model-behaviour-change constraint).
- G3 Fail-safe storage invariant: no handle emitted unless blob durably on disk (existing tests + new label/enforcement variants).
- G4 Determinism: identical inputs → identical outputs across two runs (guards against accidental LLM-in-path).
- G5 Perf guard: rescue-path latency delta ≤5% at median sizes (17.7KB) and ≤10% at p95 (multi-hundred-KB) using benchmark fixture timings.
- G6 Dogfood cron extended each phase: new feature signals asserted live (labels present, ledger growing, perms correct, report runnable).

Per-component test matrices:

- T0.1 Perms: create store under `os.umask(0)` and assert 0700 dirs / 0600 files (proves explicit set, not umask luck); idx writes, blob writes, sidecar writes each checked; init-time chmod migration converts a deliberately-permissive fixture tree; failure of chmod does not crash init (logged).
- T0.2 Excludes: with simulated broken registry import (`_is_rescuable` fail-open path), assert terminal/shell/write-class results pass through unrescued; normal tools still rescue.
- T0.3 Destinations: expansion into listed destination returns honest marker (content absent, bytes compared); those tools' oversized results STILL rescued (not merged with `exclude_tools`); precedence matrix tests: allowed-tools ∩ destinations, denylist ∩ destinations; config parse rejects malformed entries.
- T0.4 Fallback: fake ctx without `register_middleware` → warning logged (caplog) and generated handle text omits the `tla:` instruction line entirely; with middleware present, instruction retained; `passref_enabled: false` behaves identically to missing middleware for handle text.
- T0.5 Chain: registered tool schema enum includes `chain`; `fetch(mode="chain")` end-to-end via handler; docs contain the mode.
- T1.1 Report math: golden-file test — synthetic session indexes with known timestamps produce expected p50/p95/histogram exactly; empty/missing-field inputs produce empty report, not crash; report reads live-format indexes (fixture cloned from production shape).
- T1.2 Sequences: JSONL sidecar append on each fetch; restart reload preserves ordering (fixture: write N events, reload, assert order); flag-off = no sidecar writes at all; turn-less environment degrades report to wall-clock mode (explicitly labelled in output).
- T1.3 Args: snapshot capped at configured size; keys/values matching secret patterns (api_key/token/password/authorization/bearer) redacted in stored form; original result content untouched (hash equality before/after); args-absent tools produce null provenance cleanly.
- T1.4 Integrity: corrupted blob fixture → deterministic refusal marker (exact string asserted) + integrity event in log; valid blob passes with overhead within perf guard; verification skippable via config for tests only (documented, default on).
- T1.5 Ledger: every expansion/denial produces exactly one well-formed JSONL line (schema-validated); INFO-level visible while debug suppressed; total-budget-exceeded and cross-session-denial events recorded with correct decision values.
- T2.1 Labels: map-hit, map-miss-default, regex-secondary assignment; tombstone conversion preserves label (regression for finding D3); size-cap sweep respects label TTL floor (D3b); invalid label names rejected at config load; label absent → treated as public/unclassified everywhere.
- T2.2 Audit mode: labelled expansion logged with destination class; `/rescuer` output includes flow summary counts; NO behaviour difference vs pre-T2 for any fetch/expansion (byte-diff harness).
- T2.3 Enforcement: credential blob → non-allowlisted destination = marker-not-content (bytes asserted); masked range/grep/search output for credential blobs (pattern-mask verified per mode); `full` refused; 24h hard TTL beats hot-pinning in sweep simulation; enforcement-off = pure audit; public blobs completely unaffected (false-positive budget test).
- T2.4 Label hook: `label_for_args` returns label for known shapes, default otherwise; read-only (no store mutation) verified by store-state snapshot comparison.
- T3 Governor: consequential classification per config; bypass allowlist short-circuits before any analysis; ambiguous high-risk call → defer message containing clarification question, NOT a block-error string; unambiguous call passes untouched; entity extraction immune to prompt-injection-style content in args (adversarial fixtures); audit→enforce flip covered by independent flag tests; deferral-rate counters feed the Phase 1 harness format.
- T4.1 Versions: second put of changed content bumps version; stale-handle fetch surfaces conflict marker; unchanged content dedups (same id, same version); memlock-shaped instruction fixture drives the currency-check scenario.
- T4.2 Encryption: roundtrip encrypt/decrypt; key file created 0600; ciphertext-at-rest assertion (raw file lacks known plaintext substrings) once enabled; disabled state = current plaintext behaviour byte-identical; deleted encrypted blob leaves standard tombstone; missing/corrupt key → deterministic fail-safe (no partial plaintext ever returned).
- T5.1 Deploy script: dry-run prints planned copy set without writing; real run diffs clean tree → plugin dir; refuses when repo dirty (unless forced).
- T5.2 Docs: fresh-checkout smoke — documented command runs suite green in a temp venv clone.

Release process per phase: implement → full gates G1–G6 → commit per task → deploy via T5.1 script (once it exists; manual cp before) → gateway restart → 48h dogfood observation → phase exit noted in this checklist → next phase starts.

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Enforcement hurts completion rate | Agent friction, dead ends | Audit-first staging; enforcement flips gated on harness numbers (D1, T3.4) |
| Label misconfiguration silences legitimate flows | Lost functionality | Default labels conservative (public); invalid configs rejected at load; byte-diff harness proves audit-mode inert |
| Args provenance creates new secrets-at-rest | Worse security than no feature | Pattern redaction + caps before capture (D6); ledger/provenance covered by same perms hardening as blobs |
| Hash-verify latency on large blobs | Rescue path slowdown | Perf guard G5; skip-flag for benchmarks only |
| Host API drift (hook/middleware contracts) | Breakage on Hermes updates | Contract tests pin observed behaviour; fail-loud warnings on absence (T0.4 pattern) |
| Scope creep in governor | Identity bloat beyond intent | Phase 3 blocked on Sahil confirming identity expansion (decision b) |

## What I need from Sahil before Phase 0 starts

1. Approve this plan (any edits folded in before decomposition).
2. Confirm Tier 3 identity expansion (rescue tool → pre-execution governor) matches your intent — else Phase 3 is struck and its label-query hook (T2.4) remains as the dispatch-layer bridge.
3. Note public promotion (T5.5) proceeds nowhere without separate explicit sign-off.
