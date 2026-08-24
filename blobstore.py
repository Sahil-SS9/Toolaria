"""SHA256-addressed blob store for rescued tool results. Global keys, per-session indexes."""
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import threading
from pathlib import Path

try:
    import fcntl  # POSIX-only file locking for cross-process RMW safety.
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX fallback path
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False

try:
    from .excerpt import detect_type as _detect_type
    from .index import build_outline as _struct_outline
    from .index import render_outline as _render_outline
    from .chunking import chunk_lines as _chunk_lines
    from . import semantic as _sem
    from labels import (label_for_tool, label_for_args as _label_for_args,
                              VALID_LABELS, _LABEL_UPGRADE_PATTERNS,
                              _BUILTIN_TOOL_LABELS)
    from entities import (get_registry as _entity_registry,
                              extract_entities as _extract_entities,
                              distinct_entity_kinds as _distinct_entity_kinds)
except ImportError:
    from excerpt import detect_type as _detect_type  # type: ignore[no-redef]
    from index import build_outline as _struct_outline  # type: ignore[no-redef]
    from index import render_outline as _render_outline  # type: ignore[no-redef]
    from chunking import chunk_lines as _chunk_lines  # type: ignore[no-redef]
    import semantic as _sem  # type: ignore[no-redef]
    from labels import (label_for_tool, label_for_args as _label_for_args,  # type: ignore[no-redef]
                         VALID_LABELS, _LABEL_UPGRADE_PATTERNS,  # type: ignore[no-redef]
                         _BUILTIN_TOOL_LABELS)  # type: ignore[no-redef]
    from entities import (get_registry as _entity_registry,  # type: ignore[no-redef]
                            extract_entities as _extract_entities,  # type: ignore[no-redef]
                            distinct_entity_kinds as _distinct_entity_kinds)  # type: ignore[no-redef]


# Sensitivity ordering for content-aware label resolution (FIX-1). The
# highest index in this tuple wins when comparing two labels; this is the
# single source of truth for "credential > personal > internal > public"
# comparisons in BlobStore. passref consumes it via
# BlobStore._max_label_for_blob (fail-closed lookup), not its own copy.
_LABEL_SENSITIVITY = {"public": 0, "internal": 1, "personal": 2,
                       "credential": 3}
_SLICE_MASK_MARKER = "[masked:credential-shape]"
# Over-fetch headroom for masked grep/chain: raw output is gathered at 4x
# cap so post-mask re-capping still fills the budget. (Correctness agent:
# named once — the two call sites must not drift.)
_MASK_OVERFETCH_FACTOR = 4
# Scan window when testing a line against credential-shape patterns.
# (Clarity agent: was hardcoded [:2000] in three places; one constant.)
_MASK_LINE_SCAN_LEN = 2000
_SLICE_BUDGET_MARKER_PREFIX = (
    "[Toolaria: credential slice budget exhausted for "
)
_SLICE_BUDGET_MARKER_SUFFIX = (
    "; use allowlisted destinations]"
)
_CREDENTIAL_SLICE_BUDGET_DEFAULT = 2000


# ── T3.1: entity_kinds resolver ────────────────────────────────────────────


def _compute_entity_kinds(args, cfg) -> list[str]:
    """Return the sorted distinct entity kinds referenced by *args*.

    Best-effort: a broken entity_registry (e.g. an uncompilable regex
    that survived validation) yields an empty list with a logged
    warning rather than crashing the rescue path. The
    ``entity_registry`` itself is validated at register time (FIX-4
    posture), so under normal operation this is O(args size).
    """
    if not cfg:
        return []
    try:
        registry = _entity_registry(cfg)
    except Exception as exc:
        logger.warning(
            "toolaria: entity_registry resolve failed: %s; "
            "falling back to empty entity_kinds", exc,
        )
        return []
    if not registry:
        return []
    try:
        matches = _extract_entities(args, registry)
    except Exception as exc:
        logger.warning(
            "toolaria: extract_entities failed: %s; "
            "falling back to empty entity_kinds", exc,
        )
        return []
    return _distinct_entity_kinds(matches)


# ── Phase 1 helpers (T1.3 redaction, T1.4 integrity marker) ─────────────

# Deterministic marker returned (and logged) when fetch-time integrity verify
# fails. Exact string is part of the T1.4 contract — the test matrix asserts
# the literal shape, and downstream parsers may key off the prefix.
INTEGRITY_FAIL_MARKER_PREFIX = (
    "[Toolaria: integrity check failed for blob "
)
INTEGRITY_FAIL_MARKER_SUFFIX = "; content does not match index hash]"

# Secret-key name pattern (matches by key, case-insensitive): any field whose
# name looks like an API key / token / password / bearer / secret must be
# scrubbed BEFORE the args snapshot lands in the index.
# Boundaries are letter-only lookarounds, NOT \b: underscore counts as a word
# character, so \btoken\b cannot match inside access_token / refresh_token /
# client_secret — exactly the snake_case shapes real APIs use. The lookarounds
# still reject benign lookalikes (tokenizer: 'token' + 'i'; keynote: no hit).
_SECRET_KEY_RE = re.compile(
    r"(?i)(?<![A-Za-z])(api[_-]?key|token|password|authorization|bearer|secret)(?![A-Za-z])"
)
# Secret-value pattern (matches the literal text, regardless of key):
# - sk-XXXXXXX (8+ alphanumerics after the prefix) — common API key shapes
# - "bearer <token>" — Authorization headers leaking into args
_SECRET_VALUE_RE = re.compile(
    r"sk-[A-Za-z0-9]{8,}|bearer\s+\S+", re.IGNORECASE
)
_REDACTED = "[REDACTED]"


def _redact_args_snapshot(args, max_chars: int):
    """Produce a JSON-serializable, redaction-safe snapshot of *args*.

    Recurses through dicts/lists; values whose KEY matches a secret
    pattern are replaced with the REDACTED marker, and any string VALUE
    that matches a secret pattern (sk-… or bearer …) is also masked
    regardless of its key. The result is truncated to ``max_chars``.

    None or empty args return None (null provenance). Non-JSON-native
    values fall back to ``repr``; an unserializable input becomes an
    empty dict so a broken tool never blocks a rescue.
    """
    if args is None:
        return None
    try:
        snap = _scrub(args)
        text = json.dumps(snap, sort_keys=True, default=repr, ensure_ascii=False)
    except Exception:
        return {}
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[Toolaria: args_snapshot truncated at {max_chars} chars]"
    return text


def _scrub(value):
    """Recursively redact *value* in-place-ish (returns a new container)."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k)
            if _SECRET_KEY_RE.search(key):
                out[key] = _REDACTED
            else:
                out[key] = _scrub(v)
        return out
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, tuple):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub(_REDACTED, value)
    return value

_LOCK = threading.Lock()
_BLOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class _FlockAcquire:
    """Context manager wrapping fcntl.flock(LOCK_EX) + release.

    flock is a per-process, per-file-descriptor advisory lock:
    re-acquiring an EX lock on the same fd is a no-op (it just
    increments the per-process reference count), and closing any
    fd to that file releases the lock. We open a dedicated fd per
    charge so each release is independent, and we release via
    LOCK_UN + close in __exit__ so a second process can acquire
    immediately after this charge returns.
    """
    __slots__ = ("_fd", "_path", "_locked")

    def __init__(self, fd, path):
        self._fd = fd
        self._path = path
        self._locked = False

    def __enter__(self):
        if fcntl is None:
            return self
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        self._locked = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._locked and fcntl is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                self._locked = False
        return False


class _SliceBudgetExceeded(Exception):
    """Internal control-flow signal raised by
    ``BlobStore._charge_slice_budget`` when the per-blob
    credential-slice byte budget has been exceeded. The fetch()
    dispatcher catches this and returns the deterministic budget-
    exhausted marker to the caller."""

    def __init__(self, marker: str):
        super().__init__(marker)
        self.marker = marker


# Phase 0 hardening (T0.1): explicit perms on every file/dir created by the
# store. mkdir and write_bytes honour umask, so a permissive umask (or a
# pre-existing tree left by an older install) would otherwise leak blobs at
# world-readable. Setting the mode explicitly after every create makes the
# guarantee independent of umask and of installation history.
_DIR_MODE = 0o700
_FILE_MODE = 0o600

logger = logging.getLogger(__name__)


def _chmod_safe(path, mode: int) -> bool:
    """Best-effort chmod that logs and swallows permission errors.

    Returns True on success, False on failure. A chmod failure (e.g. a
    read-only mount, an immutable file, or an unsupported platform) must
    never crash store init: the store is still usable, just without the
    perm tightening. The warning makes the gap visible for the operator.
    """
    try:
        os.chmod(path, mode)
        return True
    except OSError as exc:
        logger.warning("toolaria: chmod %s to 0o%o failed: %s", path, mode, exc)
        return False


def _tighten_perms(root: Path, mode: int) -> None:
    """Walk *root* (a dir) and chmod every dir/file under it to *mode*.

    Used for init-time migration of older permissive stores (0755 dirs,
    0644 files). Failures on individual entries are swallowed (logged by
    _chmod_safe); migration is best-effort.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        _chmod_safe(Path(dirpath), _DIR_MODE)
        for name in filenames:
            _chmod_safe(Path(dirpath) / name, mode)

# The grep engine. Arbitrary user regex against adversarial blob content is a
# ReDoS hazard that no static denylist fully closes (e.g. a*a*a*...X or
# (a|a)*X backtrack exponentially in C, where a between-lines timeout never
# fires). The `regex` module honours a mid-search timeout, so when it is
# present every pattern is bounded. Without it we fall back to literal
# substring search only (linear, safe); metacharacter patterns are refused
# with a hint to install `regex`.
try:
    import regex as _regex_engine
    _HAVE_REGEX = True
except ImportError:
    _regex_engine = None
    _HAVE_REGEX = False

# Patterns containing any of these are "regex" rather than literal; refused on
# the fallback path.
_META_CHARS = set(r".^$*+?{}[]\|()")
# Control characters are never allowed in a pattern.
_CONTROL_RE = re.compile(r"[\x00-\x1f]")


class BlobStore:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        bp = Path(cfg.get("store_path", "~/.hermes/toolaria")).expanduser().resolve()
        # Create root first so the subsequent sub-dir mkdirs do not need
        # parents=True (and thus skip the root chmod below).
        bp.mkdir(parents=True, exist_ok=True)
        _chmod_safe(bp, _DIR_MODE)
        self.store_path = bp
        self.blob_dir = bp / "blobs"
        self.meta_dir = bp / "sessions"
        self.sidecar_dir = bp / "sidecars"
        self.blob_dir.mkdir(exist_ok=True)
        self.meta_dir.mkdir(exist_ok=True)
        self.sidecar_dir.mkdir(exist_ok=True)
        for d in (self.blob_dir, self.meta_dir, self.sidecar_dir):
            _chmod_safe(d, _DIR_MODE)
        # Migration: tighten any older permissive tree the install may have
        # left behind (0755 dirs / 0644 files). Best-effort; chmod failures
        # are logged but do not stop init.
        _tighten_perms(self.blob_dir, _FILE_MODE)
        _tighten_perms(self.meta_dir, _FILE_MODE)
        _tighten_perms(self.sidecar_dir, _FILE_MODE)
        # Hot-blob tracking: in-memory fetch log keyed by (safe_sid, blob_id).
        # Records fetch timestamps; recency-weighted count computed during sweep.
        # Persisted to index entries during sweep; reloaded on init.
        self._fetch_log: dict[tuple[str, str], list[float]] = {}
        self._load_fetch_log()
        # T1.2: sequences sidecar lazily created on first write when enabled.
        self._sequences_dir = bp / "sequences"
        # RISKY-3 (Phase 4): dedicated lockfile for cross-process
        # serialisation of credential-slice budget RMW. Eagerly
        # touched at init so the lockfile exists from the moment
        # the store is ready (an operator can `ls` it, and two
        # concurrent processes can immediately race on it without
        # the first charge paying the touch cost). The fcntl.flock
        # is wrapped around the entire read-modify-write of
        # credential_served_chars so two processes charging the same
        # blob can't lose increments.
        self._budget_lock_path = bp / ".budget.lock"
        try:
            self._budget_lock_path.touch(exist_ok=True)
            _chmod_safe(self._budget_lock_path, _FILE_MODE)
        except OSError as exc:
            logger.debug(
                "toolaria: could not pre-create .budget.lock at %s: %s",
                self._budget_lock_path, exc,
            )
        # CAREFUL (a) Phase 4: in-memory fallback accumulator so the
        # budget is still enforced within the process when
        # _write_idx_file fails (disk full, permission denied,
        # operator-broken mount). Cleared on every successful
        # persist so a transient failure does not silently inflate
        # the in-memory tally.
        self._budget_fallback: dict[str, int] = {}
        # CAREFUL (b) Phase 4: log the legacy empty-session
        # fallback at most once per process so the operator sees
        # the legacy dispatcher is in use without spamming every
        # charge with a WARNING line.
        self._empty_session_logged = False

    # ── sidecars (per-blob index/vector artefacts) ──

    def sidecar_path(self, blob_id: str, suffix: str) -> Path | None:
        if not _BLOB_ID_RE.match(blob_id):
            return None
        return self.sidecar_dir / f"{blob_id}.{suffix}.json"

    def read_sidecar(self, blob_id: str, suffix: str):
        p = self.sidecar_path(blob_id, suffix)
        if p and p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def write_sidecar(self, blob_id: str, suffix: str, data) -> None:
        p = self.sidecar_path(blob_id, suffix)
        if p is None:
            return
        # Prefix the temp with the blob id so an orphan from a crash between
        # mkstemp and replace is still caught by delete_sidecars' glob.
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f"{blob_id}.tmp", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, p)
            _chmod_safe(p, _FILE_MODE)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def delete_sidecars(self, blob_id: str) -> None:
        for p in self.sidecar_dir.glob(f"{blob_id}.*.json"):
            try:
                p.unlink()
            except OSError:
                pass

    def blob_text(self, blob_id: str) -> str | None:
        """Decoded blob content, or None if missing or binary."""
        bpath = self.blob_dir / blob_id
        if not bpath.exists():
            return None
        try:
            return bpath.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            return None

    def build_outline(self, blob_id: str, text: str) -> dict:
        """Build and cache the structural outline for a blob (cheap, sync).
        Safe to call at rescue time."""
        kind, _ = _detect_type(text)
        outline = _struct_outline(text, kind, self.cfg)
        self.write_sidecar(blob_id, "outline", outline)
        return outline

    def _outline(self, blob_id: str, text: str) -> str:
        cached = self.read_sidecar(blob_id, "outline")
        if cached is None:
            cached = self.build_outline(blob_id, text)
        return _render_outline(cached)

    # ── semantic search ──

    def _chunks(self, blob_id: str, text: str) -> tuple[list[dict], bool]:
        """Line-aligned chunks for a blob, cached as a sidecar.
        Returns (chunks, truncated) where truncated means the blob was larger
        than search_max_chunks chunks and only the head was indexed."""
        cached = self.read_sidecar(blob_id, "chunks")
        if cached is not None:
            return cached.get("chunks", []), cached.get("truncated", False)
        target = self.cfg.get("search_chunk_chars", 1200)
        overlap = self.cfg.get("search_chunk_overlap_lines", 2)
        max_chunks = self.cfg.get("search_max_chunks", 400)
        # Cap per-chunk text so a single huge line cannot hand a giant string
        # to the embedder; range/grep still reach the full line in the blob.
        text_cap = target * 4
        chunks = []
        for c in _chunk_lines(text, target, overlap):
            d = c.as_dict()
            d["text"] = d["text"][:text_cap]
            chunks.append(d)
        truncated = len(chunks) > max_chunks
        chunks = chunks[:max_chunks]
        self.write_sidecar(blob_id, "chunks", {"chunks": chunks, "truncated": truncated})
        return chunks, truncated

    def _chunk_vectors(self, blob_id: str, chunks: list[dict],
                       model_name: str) -> list[list[float]] | None:
        """Embeddings for a blob's chunks, cached and keyed by model name.
        None when embeddings are unavailable."""
        if not _sem.embeddings_available():
            return None
        cached = self.read_sidecar(blob_id, "vectors")
        if cached and cached.get("model") == model_name \
                and len(cached.get("vectors", [])) == len(chunks):
            return cached["vectors"]
        vectors = _sem.embed([c["text"] for c in chunks], model_name)
        if vectors is None:
            return None
        self.write_sidecar(blob_id, "vectors",
                           {"model": model_name, "vectors": vectors})
        return vectors

    def search(self, blob_id: str, query: str, text: str) -> str:
        if not query:
            return "Error: search requires query=..."
        chunks, truncated = self._chunks(blob_id, text)
        if not chunks:
            return "[search: blob is empty]"
        top_k = int(self.cfg.get("search_top_k", 5))
        snippet = int(self.cfg.get("search_snippet_chars", 400))
        model_name = self.cfg.get("embedding_model", "all-MiniLM-L6-v2")
        trunc_note = ("" if not truncated else
                      " [note: blob too large to fully index; only the head "
                      "was searched, use grep/range for the rest]")

        vectors = self._chunk_vectors(blob_id, chunks, model_name)
        method, ranked = _sem.rank(
            [c["text"] for c in chunks], vectors, query, model_name, top_k,
        )
        if not ranked:
            return f"[search ({method}): no matches for '{query}']{trunc_note}"

        out = [f"[search ({method}) top {len(ranked)} for '{query}'; "
               f"line numbers for rescuer_fetch range mode]{trunc_note}"]
        for idx, score in ranked:
            c = chunks[idx]
            body = c["text"][:snippet]
            out.append(
                f"--- score {score:.3f}  lines {c['start_line']}..{c['end_line']} ---\n"
                f"{body}"
            )
        return "\n".join(out)

    # ── blob i/o ──────────────────────────

    def put(self, content: str, tool_name: str = "", session_id: str = "",
            args=None, label=None) -> str:
        """Store content, return short blob_id (first 12 hex of SHA256).

        *session_id* is the owning session; callers must pass it so the
        per-session index stays correct under concurrent sessions.

        *args* (T1.3) is the caller's tool args; if provided, a redacted
        snapshot is persisted into the index entry under
        ``args_snapshot`` so a future rescuer can audit which inputs
        produced a blob. The content hash is unaffected — redaction only
        touches metadata. ``args_snapshot_max_chars`` caps the snapshot
        length; pass ``args_snapshot_max_chars=0`` to disable capture
        even when args is provided (benchmark-only escape hatch).

        *label* (T2.1/T2.4) is an explicit sensitivity label override.
        When None, the label is resolved from the tool name +
        args-shape. When provided, it bypasses the tool/args heuristic
        and is stored verbatim. Useful for callers that have already
        classified (e.g. the host dispatch layer consulting
        ``label_for_args`` before the rescue fires).

        FIX-1 (Phase 3): the label is content-owned, not session-owned.
        ``put()`` NEVER downgrades an existing higher label on the
        same blob_id in this session's index, and when stamping a new
        entry it carries the max across every session that already
        holds the blob — so a credential rescue is preserved across
        re-rescue of identical bytes through a public tool.
        """
        if isinstance(content, str):
            raw = content.encode("utf-8")
        else:
            raw = content
        bhash = hashlib.sha256(raw).hexdigest()
        bid = bhash[:12]
        bpath = self.blob_dir / bid
        sid = session_id or "unknown"
        # T1.3: redact args BEFORE storage; hash equality of the content
        # bytes is unaffected because we only touch the index entry.
        max_chars = int(self.cfg.get("args_snapshot_max_chars", 2000))
        if max_chars <= 0:
            args_snapshot = None
        else:
            args_snapshot = _redact_args_snapshot(args, max_chars)
        # T2.1: resolve sensitivity label. Explicit > tool-map >
        # args-shape. label_for_args never downgrades, so passing an
        # operator-configured "personal" through the args heuristic still
        # yields "personal" (or "credential" if args look secret-shaped).
        #
        # FIX-4 hardening: a bad operator map could raise here; wrap the
        # resolution in try/except so the rescue path never crashes —
        # fall back to built-in defaults with a WARNING.
        try:
            if label is None:
                tool_label = label_for_tool(tool_name, self.cfg)
                resolved = _label_for_args(args, tool_label, self.cfg)
            else:
                # Explicit label still respects args-shape upgrade — but only
                # if the caller hasn't already gone to the ceiling. A "public"
                # explicit that gets credential-shaped args becomes
                # credential; "credential" stays credential. Downgrade is
                # never implied by an explicit (operator-set) label.
                tool_label = label_for_tool(tool_name, self.cfg)
                resolved = (_label_for_args(args, label, self.cfg)
                            if label != "credential" else label)
        except Exception as exc:
            logger.warning(
                "toolaria: label resolution failed for %s (tool=%s): %s; "
                "falling back to built-in defaults",
                bid, tool_name, exc,
            )
            try:
                tool_label = _BUILTIN_TOOL_LABELS.get(tool_name or "", "public")
                resolved = _label_for_args(args, tool_label, self.cfg)
            except Exception:
                resolved = "public"
        label = resolved
        with _LOCK:
            if not bpath.exists():
                bpath.write_bytes(raw)
                _chmod_safe(bpath, _FILE_MODE)
            idx = self._load_idx(sid)
            idx.setdefault("blobs", {})
            existing = idx["blobs"].get(bid, {})
            # FIX-1 upgrade-only: never overwrite a higher existing label
            # in THIS session's index with a lower one. Cross-session max
            # is computed below via the per-blob scan.
            new_rank = self._label_rank(label)
            existing_rank = self._label_rank(existing.get("label"))
            # Cross-session max label (FIX-1): if any other session holds
            # a higher label for the same blob_id, the new entry must
            # carry that max so this session's lookup also sees it.
            cross_max = self._max_label_for_blob(bid)
            cross_rank = self._label_rank(cross_max)
            final_rank = max(new_rank, existing_rank, cross_rank)
            final_label = label
            for candidate in (existing.get("label"), cross_max):
                if self._label_rank(candidate) == final_rank:
                    final_label = candidate
                    break
            idx["blobs"][bid] = {
                "t": time.time(),
                "tool": tool_name,
                "size": len(raw),
                "hash": bhash,
                "args_snapshot": args_snapshot,
                # T2.1: persist label alongside other metadata. Kept on
                # tombstones (D3) so the audit script can report flow
                # even after sweeps.
                "label": final_label,
                # T3.1: entity kinds referenced by the blob's args.
                # Sorted list (JSON-safe) of distinct kinds; empty
                # list when entity_registry is empty so the field is
                # always present and the audit script never crashes on
                # a missing key. Both sweep paths preserve this field
                # alongside label (T3.1 / D3).
                "entity_kinds": _compute_entity_kinds(args, self.cfg),
            }
            # FIX-2: credential-slice cumulative-chars counter (resets on
            # sweep). Seed only to make the field visible to audit
            # tooling; _charge_slice_budget's .get default handles absence.
            if final_label == "credential":
                idx["blobs"][bid].setdefault("credential_served_chars", 0)
            self._save_idx(idx, sid)
        return bid

    def backfill_labels(self) -> None:
        """Phase-3 migration: walk every index entry lacking a label and
        stamp it from ``label_for_tool`` + ``label_for_args``.

        Called from ``register()`` so a pre-T2.1 store comes up with
        labels everywhere — under ``enforcement_enabled=True``, a
        label-less entry must NOT fall back to 'public' (fail-closed);
        the migration is what makes fail-closed safe.

        Best-effort: a malformed entry is logged and skipped (we never
        break load for an unclassifiable record).
        """
        for ip in sorted(self.meta_dir.glob("*.json")):
            try:
                idx = self._read_idx_file(ip)
            except Exception as exc:
                logger.warning(
                    "toolaria: backfill could not read %s: %s", ip, exc)
                continue
            blobs = idx.get("blobs", {})
            changed = False
            for bid, entry in list(blobs.items()):
                if not isinstance(entry, dict):
                    continue
                if entry.get("label") in VALID_LABELS:
                    continue
                try:
                    tool_name = entry.get("tool", "") or ""
                    tool_label = label_for_tool(tool_name, self.cfg)
                    snap = entry.get("args_snapshot")
                    label = _label_for_args(snap, tool_label, self.cfg)
                except Exception as exc:
                    logger.warning(
                        "toolaria: backfill classify failed for %s/%s: %s; "
                        "defaulting to 'public'",
                        ip.stem, bid, exc,
                    )
                    label = "public"
                entry["label"] = label
                changed = True
            if changed:
                try:
                    self._write_idx_file(ip, idx)
                except Exception as exc:
                    logger.warning(
                        "toolaria: backfill could not write %s: %s",
                        ip, exc,
                    )

    def _refresh_blob(self, blob_id: str, session_id: str,
                       turn: int | None = None) -> None:
        """Bump the access time of a blob's index entry so a result the model
        is still fetching survives the next TTL sweep.

        Scoped to the owning session only: refresh is a best-effort touch, not
        correctness-critical, so the all-session fan-out (one rewrite per
        index file per fetch) is not worth the write amplification. Throttled
        so back-to-back fetches do not rewrite the file each time.

        T1.1: also persist ``first_fetch_ts`` (set once on first observation)
        and ``fetch_count`` (monotonic total) into the index entry on every
        fetch — the reacquisition report reads these from session indexes
        so they must survive a restart without waiting for the next sweep.

        Also records the fetch in the in-memory fetch log for hot-blob
        tracking — no disk write here; the counter is flushed to the index
        during the next sweep.

        T1.2: when ``sequence_capture`` is enabled, append one
        ``{ts, sid, blob_id, turn?}`` line to the sidecar. Flag-off ⇒
        zero sidecar writes (the helper short-circuits)."""
        if not session_id:
            return
        now = time.time()
        safe_sid = self._safe_sid(session_id)
        with _LOCK:
            ip = self._idx_path(session_id)
            idx = self._read_idx_file(ip)
            entry = idx.get("blobs", {}).get(blob_id)
            if entry and "swept_at" not in entry:
                # T1.1: instrument with first-fetch timestamp + total fetch
                # count. These are persisted immediately so the report sees
                # current numbers without waiting for a sweep.
                if "first_fetch_ts" not in entry:
                    entry["first_fetch_ts"] = now
                entry["fetch_count"] = entry.get("fetch_count", 0) + 1
                # Throttled recency bump (existing hot-blob TTL behaviour).
                if now - entry.get("t", 0) > 60:
                    entry["t"] = now
                self._write_idx_file(ip, idx)
        # Hot-blob tracking: record fetch timestamp in memory (no disk write).
        key = (safe_sid, blob_id)
        self._fetch_log.setdefault(key, []).append(now)
        # Trim entries older than 7 days to bound memory.
        cutoff = now - 604800
        self._fetch_log[key] = [t for t in self._fetch_log[key] if t > cutoff]
        # T1.2: sequence sidecar (default OFF; see config.yaml).
        self._record_sequence(blob_id, session_id, now, turn)

    def _sequences_path(self) -> "Path | None":
        """Return the JSONL sidecar path, or None when capture is disabled."""
        if not self.cfg.get("sequence_capture", False):
            return None
        return self._sequences_dir / "events.jsonl"

    def _record_sequence(self, blob_id: str, session_id: str, now: float,
                          turn: int | None) -> None:
        """Append one JSONL event to the T1.2 sidecar if enabled.

        Best-effort: a write failure is logged at DEBUG but never raises,
        because the fetch path must not be broken by an audit-trail I/O
        hiccup. Ordering is preserved (open with ``a`` append mode, single
        line per write) so reload-by-line rebuilds the original sequence."""
        path = self._sequences_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _chmod_safe(path.parent, _DIR_MODE)
            record: dict = {
                "ts": now,
                "sid": session_id,
                "blob_id": blob_id,
            }
            if turn is not None:
                record["turn"] = turn
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            _chmod_safe(path, _FILE_MODE)
        except Exception as exc:
            logger.debug("toolaria: sequence append failed: %s", exc)

    def _tombstone_msg(self, blob_id: str, session_id: str = "") -> str | None:
        """Return model-facing guidance if the blob was swept but a tombstone
        survives, else None.

        Scoped to the owning session: a tombstone names a tool and result
        size, so it must not be served cross-session on a guessed id."""
        if not session_id:
            return None
        meta = self._read_idx_file(self._idx_path(session_id)) \
            .get("blobs", {}).get(blob_id, {})
        if not meta or "swept_at" not in meta:
            return None
        tool = meta.get("tool", "the source tool")
        size = meta.get("size", 0)
        return (
            f"[Swept] Blob {blob_id} (from {tool}, {size:,} chars) expired "
            f"after the retention window. The content is gone; re-run {tool} "
            f"to regenerate it."
        )

    def session_references(self, blob_id: str, session_id: str) -> bool:
        """True if *session_id*'s index holds a LIVE (non-tombstone) entry for
        the blob.

        Pass-by-reference uses this to confine expansion to the calling
        session: blobs are content-addressed and shared, so a global read
        would let one session expand another's blob by guessing a 12-hex id."""
        if not session_id:
            return False
        entry = self._read_idx_file(self._idx_path(session_id)) \
            .get("blobs", {}).get(blob_id)
        return bool(entry) and "swept_at" not in entry

    @staticmethod
    def _label_rank(label) -> int:
        """Return the sensitivity rank (0..3) of *label*, or -1 for
        unknown / None. Used by the content-aware label comparator."""
        if not label:
            return -1
        return _LABEL_SENSITIVITY.get(label, -1)

    def _max_label_for_blob(self, blob_id: str):
        """Return the highest-sensitivity label present for *blob_id*
        across every session index (live + tombstone).

        The label is a property of the content (blob_id = SHA256 prefix),
        NOT of the session. Two sessions that rescue the same bytes
        through different tool classes store different labels for the
        same blob_id; the enforcement gate must consult the highest of
        those — never just the calling-session's entry.

        Order: credential > personal > internal > public. Returns
        ``None`` if no session holds an entry for the blob at all.
        """
        best_rank = -1
        best_label = None
        # Scan every session index (sorted for determinism; the rank
        # comparison is what actually decides the winner).
        for ip in sorted(self.meta_dir.glob("*.json")):
            idx = self._read_idx_file(ip)
            entry = idx.get("blobs", {}).get(blob_id)
            if not entry:
                continue
            label = entry.get("label")
            rank = self._label_rank(label)
            if rank > best_rank:
                best_rank = rank
                best_label = label
        return best_label

    def _find_meta(self, blob_id: str, session_id: str = "",
                   include_swept: bool = False) -> dict:
        """Index metadata for a blob: the given session's entry, or the first
        entry found across all sessions (blobs are content-addressed, so any
        session's metadata describes the same bytes).

        Live entries are preferred; a tombstone is returned only when
        *include_swept* is set and no live entry exists."""
        paths = ([self._idx_path(session_id)] if session_id else []) \
            + sorted(self.meta_dir.glob("*.json"))
        tomb: dict = {}
        for ip in paths:
            meta = self._read_idx_file(ip).get("blobs", {}).get(blob_id)
            if not meta:
                continue
            if "swept_at" in meta:
                tomb = tomb or meta
                continue
            return meta
        return tomb if include_swept else {}

    def fetch(self, blob_id: str, mode: str, start=0, count=20,
              pattern=None, query=None, session_id: str = ""):
        """Retrieve a slice of a blob.
        Modes: outline, search, range, grep, stat, full."""
        if not _BLOB_ID_RE.match(blob_id):
            return f"Error: invalid blob id '{blob_id}' (expected 12 hex chars)"

        cap = self.cfg.get("fetch_max_chars", 4000)
        bpath = self.blob_dir / blob_id

        # Session scoping: when the host forwards session_id, the fetch tool
        # should not become a cross-session oracle for guessed capability ids.
        # Keep the empty-session fallback for older/single-session Hermes
        # dispatchers that do not pass session_id yet.
        if session_id and not self.session_references(blob_id, session_id):
            tomb = self._tombstone_msg(blob_id, session_id)
            if tomb:
                return tomb
            return f"Error: blob {blob_id} not available in this session"

        if not bpath.exists():
            tomb = self._tombstone_msg(blob_id, session_id)
            if tomb:
                return tomb
            return f"Error: blob {blob_id} not found (may have been swept)"

        # T2.3 + Phase 3 FIX-1 + FIX-3 + gate ordering (Phase 3 MEDIUM):
        # the credential gate runs BEFORE _refresh_blob so a refused
        # full-fetch does not bump fetch_count or stamp first_fetch_ts
        # (the gate is a pure refusal, not a content read). FIX-1 also
        # requires the label to be content-aware: we read the max label
        # across every session index, not just the calling session's
        # entry, so a credential blob that was re-rescued as public in
        # another session is still gated here.
        if (self._enforcement_enabled() and mode == "full"):
            blob_label = self._max_label_for_blob(blob_id)
            # FIX-3 fail-closed: under enforcement ON, an unresolved
            # label (None / missing everywhere) is treated as
            # credential, never as public. Under enforcement OFF the
            # per-session entry resolves as before so audit runs on
            # mixed-version stores.
            if blob_label is None:
                blob_label = "credential"
            if blob_label == "credential":
                from passref import (CREDENTIAL_REFUSE_MARKER_PREFIX,
                                     CREDENTIAL_REFUSE_MARKER_SUFFIX)
                return (f"{CREDENTIAL_REFUSE_MARKER_PREFIX}{blob_id}"
                        f"{CREDENTIAL_REFUSE_MARKER_SUFFIX}")

        # Touch the blob so an actively-used result does not expire mid-task.
        self._refresh_blob(blob_id, session_id)

        try:
            if mode == "stat":
                st = bpath.stat()
                meta = self._find_meta(blob_id, session_id)
                return (
                    f"blob: {blob_id}\n"
                    f"size: {st.st_size:,} bytes\n"
                    f"stored: {time.ctime(st.st_ctime)}\n"
                    f"tool: {meta.get('tool', '?')}"
                )
            raw = bpath.read_bytes()
        except FileNotFoundError:
            # Swept by a concurrent sweep between the existence check and read.
            return (self._tombstone_msg(blob_id, session_id)
                    or f"Error: blob {blob_id} not found (may have been swept)")
        # T1.4: integrity verification. Full SHA256 over the on-disk bytes
        # compared with the index-recorded hash. A mismatch returns an
        # exact deterministic marker and logs at WARNING. The verification
        # is gated by config (default on) so benchmarks can opt out
        # without touching the rescue/fetch contract. The ``raw`` bytes
        # are only decoded AFTER verification so a corrupted blob is
        # never surfaced to the model as data.
        if self.cfg.get("verify_integrity", True):
            meta = self._find_meta(blob_id, session_id)
            expected = meta.get("hash") if meta else None
            if expected:
                actual = hashlib.sha256(raw).hexdigest()
                if actual != expected:
                    logger.warning(
                        "toolaria: integrity check failed for blob %s "
                        "(expected %s…, got %s…)",
                        blob_id, expected[:8], actual[:8],
                    )
                    return (
                        f"{INTEGRITY_FAIL_MARKER_PREFIX}{blob_id}"
                        f"{INTEGRITY_FAIL_MARKER_SUFFIX}"
                    )

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"Error: blob {blob_id} is binary ({len(raw)} bytes)"

        if mode == "outline":
            return self._outline(blob_id, text)

        # Phase 3 FIX-2: under enforcement ON + credential label, slices
        # (range/grep/search/chain) must MASK credential-shaped lines so
        # paging cannot reconstruct the secret via repeated slice reads.
        # The structural-only modes (outline/stat) are already safe and
        # untouched. Non-matching lines pass through unchanged; line
        # count is preserved so paging semantics still work for the
        # operator.
        mask_slices = (self._enforcement_enabled()
                        and self._max_label_for_blob(blob_id) == "credential")

        if mode == "search":
            # Phase 3 FIX-2: under enforcement + credential, mask the
            # snippets too. The search output is post-filtered for any
            # line that triggers a credential-shape pattern.
            result = self.search(blob_id, query or "", text)
            if mask_slices:
                masked_lines = self._mask_lines(result.splitlines())
                result = "\n".join(masked_lines)
            # Charge the budget for the served chars (masked or not).
            try:
                self._charge_slice_budget(
                    blob_id, session_id, len(result),
                    mask_slices=mask_slices,
                )
            except _SliceBudgetExceeded as exc:
                return exc.marker
            return result

        if mode == "full":
            max_full = self.cfg.get("full_fetch_max_chars", 50000)
            if self.cfg.get("refuse_full_fetch", True) and len(text) > max_full:
                return (
                    f"Refused: blob is {len(text):,} chars, over the "
                    f"{max_full:,} char full-fetch limit. "
                    f"Use mode='range' or mode='grep' instead."
                )
            return text

        lines = text.splitlines()

        try:
            return self._dispatch_slice(
                mode, blob_id, session_id, lines, start, count, pattern, cap,
                mask_slices=mask_slices,
            )
        except _SliceBudgetExceeded as exc:
            return exc.marker

    def _dispatch_slice(self, mode, blob_id, session_id, lines, start, count,
                         pattern, cap, *, mask_slices: bool):
        """Phase-3 slice dispatcher. Wraps the per-mode logic in one
        place so the budget-exceeded signal is raised consistently
        regardless of which slice mode fired."""
        if mode == "range":
            total = len(lines)
            start = max(0, start)
            note = ""
            if start >= total and total > 0:
                note = f"[start {start} past end; clamped]\n"
                start = max(0, total - max(1, count))
            end = min(total, start + max(1, count))
            slice_lines = lines[start:end]
            if mask_slices:
                slice_lines = self._mask_lines(slice_lines)
            body = "\n".join(slice_lines)[:cap]
            served = len(body)
            self._charge_slice_budget(blob_id, session_id, served,
                                       mask_slices=mask_slices)
            return (
                f"{note}[lines {start}..{end - 1} of {total}]\n{body}"
            )

        if mode == "grep":
            if not pattern:
                return "Error: grep requires pattern=..."
            matched, served = self._grep_with_mask(
                lines, pattern, cap, mask_lines=mask_slices)
            self._charge_slice_budget(blob_id, session_id, served,
                                       mask_slices=mask_slices)
            return matched

        if mode == "chain":
            matched, served = self._chain_with_mask(
                lines, pattern, count, cap, mask_lines=mask_slices)
            self._charge_slice_budget(blob_id, session_id, served,
                                       mask_slices=mask_slices)
            return matched

        return f"Error: unknown mode '{mode}'"

    # ── credential-slice masking helpers (Phase 3 FIX-2) ─────────────

    @staticmethod
    def _mask_lines(lines) -> list:
        """Replace any line whose text matches a credential-upgrade
        pattern with the structural mask marker; preserve line count."""
        out = []
        for ln in lines:
            text = (ln[:_MASK_LINE_SCAN_LEN]
                    if isinstance(ln, str) else "")
            if any(p.search(text) for p in _LABEL_UPGRADE_PATTERNS):
                out.append(_SLICE_MASK_MARKER)
            else:
                out.append(ln)
        return out

    def _grep_with_mask(self, lines, pattern, cap, *, mask_lines: bool):
        """Grep helper that respects the FIX-2 mask contract: matched
        lines that are credential-shaped are returned as the mask
        marker instead of the raw text, but the line-number prefix and
        the body layout are preserved so paging still works. Returns
        (output, served_chars).

        RISKY-1 (Phase 4): _grep_safe truncates each matched line body
        to 500 chars before the post-filter runs. A credential shape
        that appears past position 500 of a long line was therefore
        NOT visible to the mask check, so the truncated 500-char body
        leaked unmasked under enforcement ON. Range mode (which scans
        ``line[:_MASK_LINE_SCAN_LEN]``) masked the same line. The fix:
        after parsing the line-number ``n`` from a grep output line,
        run the upgrade-pattern check against the ORIGINAL full line
        ``lines[int(n)][:_MASK_LINE_SCAN_LEN]`` instead of the
        truncated body. Output shape / line-number prefixes are
        unchanged for non-matching lines.
        """
        if not mask_lines:
            return self._grep_safe(lines, pattern, cap), 0
        # Run grep with a generous cap so we can post-filter without
        # losing line numbers; we'll re-cap after masking.
        raw = self._grep_safe(lines, pattern,
                              cap * _MASK_OVERFETCH_FACTOR)
        out_lines: list = []
        for ln in raw.splitlines():
            # Lines with the "{n}: {body}" shape are match lines.
            # Header/footer lines (timeouts, no-match, 50-match cap)
            # lack that shape and pass through verbatim.
            head, sep, body = ln.partition(": ")
            if sep and head.isdigit():
                # RISKY-1: check the ORIGINAL full line up to the
                # mask scan window, not the truncated body that
                # _grep_safe already capped at 500 chars.
                idx = int(head)
                if 0 <= idx < len(lines):
                    original = lines[idx]
                else:
                    # Defensive: an out-of-range n (shouldn't happen)
                    # falls back to the truncated body so we never
                    # raise from the mask pass.
                    original = body
                scan = (original[:_MASK_LINE_SCAN_LEN]
                        if isinstance(original, str) else "")
                if any(p.search(scan) for p in _LABEL_UPGRADE_PATTERNS):
                    out_lines.append(f"{head}: {_SLICE_MASK_MARKER}")
                    continue
            out_lines.append(ln)
        out = "\n".join(out_lines)
        if len(out) > cap:
            out = out[:cap]
        return out, min(len(out), cap)

    def _chain_with_mask(self, lines, pattern, count, cap, *, mask_lines: bool):
        """Chain helper with the same FIX-2 masking contract as grep."""
        if not mask_lines:
            return self._chain(lines, pattern, count, cap), 0
        raw = self._chain(lines, pattern, count,
                          cap * _MASK_OVERFETCH_FACTOR)
        out_lines: list = []
        for ln in raw.splitlines():
            if any(p.search(ln[:_MASK_LINE_SCAN_LEN])
                   for p in _LABEL_UPGRADE_PATTERNS):
                out_lines.append(_SLICE_MASK_MARKER)
            else:
                out_lines.append(ln)
        out = "\n".join(out_lines)
        if len(out) > cap:
            out = out[:cap]
        return out, min(len(out), cap)

    def _charge_slice_budget(self, blob_id: str, session_id: str,
                              served: int, *, mask_slices: bool) -> None:
        """FIX-2: track cumulative chars served for *blob_id* since the
        last sweep reset. Refuse further slice reads once
        ``credential_slice_total_max_chars`` is exceeded.

        The counter is persisted on the calling session's index entry;
        sweeping clears it (see ``_sweep_by_ttl``). No-op when
        ``mask_slices`` is False (enforcement off, or non-credential blob).

        RISKY-3 (Phase 4): the in-process threading.Lock is a thread
        guard only. Two processes charging concurrently against the
        same store path used to lose increments (verified: 100+100
        -> 100). The RMW is now wrapped in an fcntl.flock on a
        dedicated ``<store_path>/.budget.lock`` file so cross-process
        charges serialise. The lockfile is created at init; non-POSIX
        platforms (Windows native) skip the flock cleanly and rely on
        the in-process lock with a DEBUG log.

        CAREFUL (b) Phase 4: legacy dispatchers that call with an
        empty session_id now fall back to ``_global`` so they still
        get budgeting. The fallback is logged at most once per
        process.
        """
        if not mask_slices or not served:
            return
        budget = int(self.cfg.get(
            "credential_slice_total_max_chars",
            _CREDENTIAL_SLICE_BUDGET_DEFAULT))
        if budget <= 0:
            return
        # CAREFUL (b): empty session_id from legacy dispatchers must
        # still get budgeting. Fall back to a shared '_global' session
        # so all such callers share one counter per blob.
        effective_sid = session_id or "_global"
        if not session_id and not self._empty_session_logged:
            logger.info(
                "toolaria: empty session_id passed to _charge_slice_budget; "
                "falling back to '_global' budget (legacy dispatcher in "
                "use — log emitted once per process)"
            )
            self._empty_session_logged = True
        # RISKY-3: outer in-process lock + cross-process fcntl.flock.
        # The threading.Lock keeps two threads in the same process
        # from racing on the file descriptor; the fcntl.flock keeps
        # two processes from racing on the same store path. Both are
        # held across the entire read-modify-write so the budget
        # never loses increments.
        with _LOCK:
            self._charge_slice_budget_locked(
                blob_id, effective_sid, served, budget,
            )

    def _charge_slice_budget_locked(self, blob_id: str, session_id: str,
                                      served: int, budget: int) -> None:
        """Inner RMW for _charge_slice_budget. Caller holds _LOCK.

        Separated so the outer lock + flock wrapping is clearly
        distinct from the actual read-modify-write logic. This is
        what serialises across processes.
        """
        ip = self._idx_path(session_id)
        lock_path = self._budget_lock_path
        lock_fd = None
        flock_ctx = None
        if _HAVE_FCNTL:
            try:
                # Best-effort chmod so the lockfile keeps the
                # store-wide 0600 perms even if a previous install
                # left it permissive.
                _chmod_safe(lock_path, _FILE_MODE)
                lock_fd = open(lock_path, "w")
                flock_ctx = _FlockAcquire(lock_fd, lock_path)
                flock_ctx.__enter__()
            except OSError as exc:
                logger.debug(
                    "toolaria: could not acquire .budget.lock (%s); "
                    "falling back to in-process _LOCK only: %s",
                    lock_path, exc,
                )
                if lock_fd is not None:
                    try:
                        lock_fd.close()
                    except OSError:
                        pass
                lock_fd = None
                flock_ctx = None
        else:  # pragma: no cover - non-POSIX fallback
            logger.debug(
                "toolaria: fcntl unavailable; cross-process budget "
                "serialisation is best-effort (single-process _LOCK only)"
            )
        try:
            idx = self._read_idx_file(ip)
            entry = idx.get("blobs", {}).get(blob_id)
            if not entry or "swept_at" in entry:
                return
            # CAREFUL (a): int() coercion with a WARNING + clamp
            # negative to 0, so a corrupted index can't crash the
            # budget path or skew the counter.
            raw_current = entry.get("credential_served_chars", 0)
            try:
                current = int(raw_current)
            except (TypeError, ValueError):
                logger.warning(
                    "toolaria: credential_served_chars=%r for %s is not "
                    "an integer; treating as 0",
                    raw_current, blob_id,
                )
                current = 0
            if current < 0:
                logger.warning(
                    "toolaria: credential_served_chars=%d for %s is "
                    "negative; clamping to 0",
                    current, blob_id,
                )
                current = 0
            # CAREFUL (a): in-memory fallback tracks amounts that
            # failed to persist in this process so a transient disk
            # failure does not silently disable enforcement.
            # - When the disk has caught up to (or past) our in-memory
            #   tally, clear the fallback so we stop folding it.
            # - When the disk is behind, fold the unpersisted amount
            #   into ``current`` so the budget check sees the real
            #   served total.
            fallback = self._budget_fallback.get(blob_id, 0)
            if fallback and current >= fallback:
                self._budget_fallback.pop(blob_id, None)
                fallback = 0
            elif fallback:
                # The disk counter represents the persisted portion;
                # the in-memory fallback tracks what failed to land.
                # effective = persisted + in-memory unpersisted.
                current = current + fallback
            new_total = current + served
            entry["credential_served_chars"] = new_total
            try:
                self._write_idx_file(ip, idx)
                # Persist succeeded — the disk now reflects the new
                # total; any in-memory fallback is obsolete.
                self._budget_fallback.pop(blob_id, None)
            except Exception as exc:
                # CAREFUL (a): WARNING (not DEBUG) and in-memory
                # fallback so budgeting survives persistence loss
                # within the process.
                logger.warning(
                    "toolaria: slice-budget counter write failed for %s "
                    "(keeping in-memory accumulator; budget still "
                    "enforced in-process): %s",
                    blob_id, exc,
                )
                self._budget_fallback[blob_id] = (
                    self._budget_fallback.get(blob_id, 0) + served)
            if new_total > budget:
                raise _SliceBudgetExceeded(
                    (f"{_SLICE_BUDGET_MARKER_PREFIX}{blob_id}"
                     f"{_SLICE_BUDGET_MARKER_SUFFIX}"),
                )
        finally:
            if flock_ctx is not None:
                try:
                    flock_ctx.__exit__(None, None, None)
                except OSError:
                    pass
            if lock_fd is not None:
                try:
                    lock_fd.close()
                except OSError:
                    pass

    # ── grep with timeout/complexity cap ───

    def _grep_safe(self, lines: list, pattern: str, cap: int) -> str:
        """Search each line for *pattern*, bounded against ReDoS.

        With the `regex` module: full regex, each search capped by a
        mid-match wall-clock timeout. Without it: literal substring only
        (linear, safe); metacharacter patterns are refused."""
        plen = len(pattern)
        pmax = self.cfg.get("grep_max_pattern_len", 80)
        if plen > pmax:
            return f"Error: pattern too long ({plen} > {pmax})"
        if _CONTROL_RE.search(pattern):
            return "Error: pattern contains control characters"

        per_line_timeout = self.cfg.get("grep_timeout_ms", 500) / 1000.0
        wall_timeout = max(per_line_timeout, 2.0)
        line_cap = self.cfg.get("grep_max_line_len", 2000)

        if _HAVE_REGEX:
            try:
                preg = _regex_engine.compile(pattern, _regex_engine.I)
            except _regex_engine.error as e:
                return f"Error: invalid regex: {e}"

            def matches(line: str) -> bool:
                try:
                    return bool(preg.search(line[:line_cap],
                                            timeout=per_line_timeout))
                except TimeoutError:
                    return False
        else:
            if set(pattern) & _META_CHARS:
                return ("Error: regex patterns need the optional 'regex' "
                        "package; install it, or use a literal substring")
            needle = pattern.lower()

            def matches(line: str) -> bool:
                return needle in line[:line_cap].lower()

        t0 = time.time()
        results = []
        total = len(lines)
        for n, line in enumerate(lines):
            if time.time() - t0 > wall_timeout:
                results.append(
                    f"[grep timed out after {wall_timeout}s; "
                    f"{len(results)} matches]")
                break
            if matches(line):
                results.append(f"{n}: {line[:500]}")
                if len(results) >= 50:
                    results.append("[50 matches; capped]")
                    break
        if not results:
            return f"[no matches for pattern '{pattern}' in {total} lines]"
        return "\n".join(results)[:cap]

    # ── chain (composite grep→context) ─────

    def _chain(self, lines: list, pattern: str, count: int, cap: int) -> str:
        """Composite fetch: grep for pattern, return context around matches.

        Bounded to a single grep→context two-step (not a full pipeline DSL).
        The 80% case: \"find X and show me what's around it\" without the
        model making separate grep + range calls."""
        if not pattern:
            return "Error: chain mode requires pattern="
        ctx = max(1, min(count, 20))  # context_lines, default from count
        total = len(lines)
        # Reuse the same matching logic as grep mode.
        per_line_timeout = self.cfg.get("grep_timeout_ms", 500) / 1000.0
        wall_timeout = max(per_line_timeout, 2.0)
        line_cap = self.cfg.get("grep_max_line_len", 2000)
        if _HAVE_REGEX:
            try:
                preg = _regex_engine.compile(pattern, _regex_engine.I)
            except _regex_engine.error as e:
                return f"Error: invalid regex: {e}"
            def _cm(line):  # noqa: E306
                try:
                    return bool(preg.search(line[:line_cap],
                                            timeout=per_line_timeout))
                except TimeoutError:
                    return False
        else:
            if set(pattern) & _META_CHARS:
                return ("Error: regex patterns need the optional 'regex' "
                        "package; install it, or use a literal substring")
            needle = pattern.lower()
            def _cm(line):  # noqa: E306
                return needle in line[:line_cap].lower()
        # Find matching line numbers.
        t0 = time.time()
        match_lines = []
        for n, line in enumerate(lines):
            if time.time() - t0 > wall_timeout:
                break
            if _cm(line):
                match_lines.append(n)
                if len(match_lines) >= 30:
                    break
        if not match_lines:
            return f"[chain: no matches for pattern '{pattern}' in {total} lines]"
        # Build context windows around each match, merging overlaps.
        windows = []
        for ml in match_lines:
            w_start = max(0, ml - ctx)
            w_end = min(total, ml + ctx + 1)
            windows.append((w_start, w_end))
        # Merge overlapping windows.
        merged = []
        for ws, we in windows:
            if merged and ws <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], we))
            else:
                merged.append((ws, we))
        # Render merged windows.
        out = [f"[chain: {len(match_lines)} match(es) for '{pattern}' in "
               f"{total} lines, ±{ctx} context lines]"]
        for ws, we in merged:
            gap = "" if ws == merged[0][0] else "...\n"
            out.append(f"{gap}[lines {ws}..{we - 1}]")
            out.append("\n".join(lines[ws:we]))
        return "\n".join(out)[:cap]

    # ── sweep ──────────────────────────────

    def lazy_sweep(self):
        """Expire blobs past TTL or over the size limit, oldest first.

        An expired index entry becomes a tombstone (keeps tool name and size,
        drops the content) so a stale handle degrades into actionable
        guidance rather than a bare error. The blob file is deleted once no
        session holds a live reference. Tombstones themselves expire after
        tombstone_ttl_hours."""
        ttl = self.cfg.get("ttl_hours", 72) * 3600
        tomb_ttl = self.cfg.get("tombstone_ttl_hours", 720) * 3600
        max_mb = self.cfg.get("max_store_mb", 500)
        now = time.time()
        with _LOCK:
            self._sweep_by_ttl(now, ttl, tomb_ttl)
            self._sweep_by_size(now, max_mb)

    @staticmethod
    def _is_live(entry: dict) -> bool:
        return "swept_at" not in entry

    @staticmethod
    def _truthy(value) -> bool:
        """Phase 3 MEDIUM bool-coercion helper.

        YAML-parsed config values are strings unless quoted correctly:
        ``enforcement_enabled: "false"`` (quoted) lands as the string
        ``"false"`` in Python, and ``bool("false")`` is ``True``. The
        enforcement gates use explicit truthy semantics so a quoted
        ``"false"`` actually disables the gate.

        Truthy set: ``{"1", "true", "yes", "on"}`` (case-insensitive).
        Anything else (including the empty string, ``"0"``, ``"no"``,
        ``"off"``, ``None``) is False.
        """
        if isinstance(value, bool):
            return value
        if not isinstance(value, str):
            return bool(value)
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _enforcement_enabled(self) -> bool:
        """T2.3: master switch for credential-grade enforcement.

        Phase 3 MEDIUM: explicit truthy semantics (not bare bool()) so
        a YAML-quoted ``"false"`` actually disables the gate."""
        return self._truthy(self.cfg.get("enforcement_enabled", False))

    def _credential_ttl_seconds(self) -> int:
        """Eagerly coerce ``credential_ttl_hours`` (Phase 3 MEDIUM).

        A quoted-string ``"24"`` or ``None`` from YAML previously
        crashed the sweep with a TypeError inside
        ``min(int, str)``. Coerce via try/except, fall back to the
        24h default with a WARNING so the sweep always completes."""
        raw = self.cfg.get("credential_ttl_hours", 24)
        try:
            n = int(raw)
            if n <= 0:
                raise ValueError("non-positive")
            return n * 3600
        except Exception as exc:
            logger.warning(
                "toolaria: credential_ttl_hours=%r is not a positive "
                "integer (%s); falling back to 24h", raw, exc,
            )
            return 24 * 3600

    def _audit_summary(self, count: int = 20) -> str:
        """T2.2: read-only last-N expansion ledger summary for /rescuer audit.

        Reads the JSONL expansion ledger directly — no blob bytes are
        touched and no fetch counters move. Degrades to a friendly line
        when the ledger is absent or empty.

        Phase 3 MEDIUM: ``count`` is clamped to ``[0..1000]`` and any
        value ``<= 0`` is treated as "return the friendly empty message"
        so a request for "the last 0 events" no longer returns the full
        ledger mislabeled. Memory is bounded by the clamped count via
        ``collections.deque(maxlen=n)`` so the function's I/O cost is
        also bounded.
        """
        # Clamp first — empty/stupid requests short-circuit to the
        # friendly line before any I/O so a misconfigured operator
        # can't accidentally pull unbounded rows.
        try:
            n = int(count)
        except Exception:
            n = 0
        if n <= 0:
            return ("Toolaria audit: no data yet "
                    "(count must be a positive integer; got "
                    f"{count!r}).")
        n = min(n, 1000)
        ledger = self.meta_dir.parent / "ledger" / "expansions.jsonl"
        if not ledger.exists():
            return ("Toolaria audit: no data yet "
                    "(no expansions recorded; ledger file not present).")
        rows: list[dict] = []
        try:
            with open(ledger, "r", encoding="utf-8") as fh:
                # Stream the tail so memory stays bounded by ``n``,
                # not by the lifetime of the ledger.
                from collections import deque
                tail: "deque[dict]" = deque(maxlen=n)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tail.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # skip malformed lines, keep summarising
                rows = list(tail)
        except OSError:
            return "Toolaria audit: ledger unreadable (I/O error)."
        if not rows:
            return "Toolaria audit: ledger present but empty."
        by_dst: dict[str, dict] = {}
        for row in rows:
            dst = row.get("dst_tool", "?")
            slot = by_dst.setdefault(
                dst, {"expanded": 0, "denied": 0, "chars": 0})
            if row.get("decision") == "expanded":
                slot["expanded"] += 1
                slot["chars"] += row.get("chars", 0) or 0
            else:
                slot["denied"] += 1
        lines = [f"Toolaria audit (last {len(rows)} events):"]
        for dst, slot in sorted(by_dst.items(),
                                key=lambda kv: -kv[1]["expanded"]):
            lines.append(
                f"  {dst}: expanded={slot['expanded']} "
                f"denied={slot['denied']} chars={slot['chars']:,}")
        return "\n".join(lines)


    def _sweep_by_ttl(self, now, ttl, tomb_ttl):
        # For each session: expire live entries past their effective TTL into
        # tombstones, and drop tombstones past the tombstone TTL.
        #
        # Hot-blob exemption: blobs with a recency-weighted fetch count above
        # `hot_fetch_threshold` use an extended `hot_ttl_hours` instead of the
        # standard TTL. The weighted count decays over time so a blob that was
        # hot last week but hasn't been touched since naturally drops back to
        # the cold eviction pool.
        hot_ttl = self.cfg.get("hot_ttl_hours", 168) * 3600
        threshold = self.cfg.get("hot_fetch_threshold", 3.0)
        half_life = self.cfg.get("fetch_decay_half_life_hours", 24) * 3600
        for ip in sorted(self.meta_dir.glob("*.json")):
            safe_sid = ip.stem
            idx = self._read_idx_file(ip)
            blobs = idx.get("blobs", {})
            changed = False
            for bid, meta in list(blobs.items()):
                if self._is_live(meta):
                    # Compute recency-weighted fetch count.
                    key = (safe_sid, bid)
                    fetch_times = self._fetch_log.get(key, [])
                    if fetch_times and half_life > 0:
                        weighted = sum(
                            0.5 ** ((now - t) / half_life) for t in fetch_times
                        )
                        # Round to 4dp to avoid floating-point edge where
                        # N recent fetches sum to 1.999999 instead of N.0.
                        weighted = round(weighted, 4)
                    else:
                        weighted = 0.0
                    effective_ttl = hot_ttl if weighted >= threshold else ttl
                    # T2.3: credential-labelled blobs have a hard 24h TTL
                    # that overrides hot-pinning — an actively-fetched
                    # credential must still age out on schedule.
                    # Phase 3 MEDIUM: credential_ttl_hours is eagerly
                    # coerced (see _credential_ttl_seconds) so a bad
                    # value falls back to 24h instead of crashing the
                    # sweep with TypeError.
                    if (self._enforcement_enabled()
                            and meta.get("label") == "credential"):
                        effective_ttl = min(
                            effective_ttl,
                            self._credential_ttl_seconds(),
                        )
                    # Phase 3 FIX-2: reset the credential-slice budget
                    # counter on every live entry — the budget is
                    # "since the last sweep reset" so a long-running
                    # blob's budget is not exhausted by yesterday's
                    # reads.
                    #
                    # RISKY-2 (Phase 4): the counter used to be popped
                    # BEFORE the TTL-expiry check below, so a sweep
                    # that expired nothing still wiped live budgets —
                    # and lazy_sweep runs on session start AND end so
                    # budgets were reset twice per session without any
                    # TTL work. The counter now drops naturally inside
                    # the tombstone-transition branch (the fresh
                    # tombstone dict below explicitly lists only
                    # swept_at/tool/size/label), so we no longer pop
                    # it here.
                    if now - meta.get("t", 0) > effective_ttl:
                        blobs[bid] = {
                            "swept_at": now,
                            "tool": meta.get("tool", ""),
                            "size": meta.get("size", 0),
                            # T2.1 / D3: carry the label into the
                            # tombstone so the audit script can still
                            # attribute historical flow after sweep.
                            "label": meta.get("label", "public"),
                            # T3.1: also carry the entity_kinds set
                            # across the TTL sweep so the audit script
                            # reports what kinds of entity references
                            # lived in the tombstoned blob.
                            "entity_kinds": list(
                                meta.get("entity_kinds", []) or []),
                        }
                        changed = True
                elif now - meta.get("swept_at", 0) > tomb_ttl:
                    del blobs[bid]
                    changed = True
            # Flush fetch-count snapshots into index entries for persistence
            # before writing. Force a write if the flush added fields even
            # when no sweep/tombstone change occurred.
            if self._flush_fetch_log(safe_sid, idx):
                changed = True
            if changed:
                self._write_idx_file(ip, idx)

        # Delete blob files no session holds a LIVE reference to, along with
        # their sidecar index/vector artefacts.
        for bf in self.blob_dir.iterdir():
            if bf.is_file() and _BLOB_ID_RE.match(bf.name):
                if not self._any_live_refs(bf.name):
                    bf.unlink()
                    self.delete_sidecars(bf.name)

    def _sidecar_bytes(self, bid: str) -> int:
        """On-disk bytes of a blob's sidecars. These are deleted with the blob
        on eviction, so they count toward the store cap alongside it."""
        total = 0
        for p in self.sidecar_dir.glob(f"{bid}.*.json"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def _sweep_by_size(self, now, max_mb):
        max_bytes = max_mb * 1024 * 1024
        all_blobs = []
        for bf in self.blob_dir.iterdir():
            if bf.is_file() and _BLOB_ID_RE.match(bf.name):
                # A blob's cap weight is its file plus its sidecars, since
                # eviction frees both.
                sz = bf.stat().st_size + self._sidecar_bytes(bf.name)
                all_blobs.append((bf.stat().st_ctime, sz, bf))
        all_blobs.sort()  # oldest first
        total = sum(sz for _, sz, _ in all_blobs)
        for _, sz, bf in all_blobs:
            if total <= max_bytes:
                break
            self._tombstone_everywhere(bf.name, now)
            if bf.exists():
                bf.unlink()
            self.delete_sidecars(bf.name)
            total -= sz

    def _tombstone_everywhere(self, bid: str, now) -> None:
        """Convert every live reference to a blob into a tombstone."""
        for ip in sorted(self.meta_dir.glob("*.json")):
            idx = self._read_idx_file(ip)
            entry = idx.get("blobs", {}).get(bid)
            if entry and self._is_live(entry):
                idx["blobs"][bid] = {
                    "swept_at": now,
                    "tool": entry.get("tool", ""),
                    "size": entry.get("size", 0),
                    # T2.1 / D3: preserve the label across the size-cap
                    # sweep path too (different code path from TTL —
                    # both must keep the label).
                    "label": entry.get("label", "public"),
                    # T3.1: also carry the entity_kinds set across the
                    # size-cap sweep (same D3 rationale as label).
                    "entity_kinds": list(
                        entry.get("entity_kinds", []) or []),
                }
                self._write_idx_file(ip, idx)

    def _any_live_refs(self, bid: str) -> bool:
        """True if any session index holds a live (non-tombstone) reference."""
        for ip in self.meta_dir.glob("*.json"):
            entry = self._read_idx_file(ip).get("blobs", {}).get(bid)
            if entry and self._is_live(entry):
                return True
        return False

    # ── session index helpers ──────────────

    @staticmethod
    def _safe_sid(session_id: str) -> str:
        """Map a session id to an injective, traversal-safe filename stem.

        A readable prefix of the slugged id aids debugging; a hash suffix
        guarantees distinct ids never collide onto one index file."""
        sid = session_id or "unknown"
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)[:32]
        digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:12]
        return f"{slug}-{digest}"

    def _idx_path(self, session_id: str):
        return self.meta_dir / f"{self._safe_sid(session_id)}.json"

    @staticmethod
    def _read_idx_file(path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {}

    @staticmethod
    def _write_idx_file(path: Path, idx: dict) -> None:
        # Atomic write: temp file + os.replace so a concurrent reader never
        # sees partial JSON.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(idx, f)
            os.replace(tmp, path)
            _chmod_safe(path, _FILE_MODE)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load_idx(self, session_id: str):
        return self._read_idx_file(self._idx_path(session_id))

    def _save_idx(self, idx: dict, session_id: str):
        self._write_idx_file(self._idx_path(session_id), idx)

    # ── hot-blob fetch log ─────────────────

    def _load_fetch_log(self) -> None:
        """Load persisted fetch counts from existing session indexes.

        Each index entry may carry a ``fetch_weight`` and ``fetch_count``
        snapshot written by ``_flush_fetch_log`` during the last sweep.
        These are loaded as synthetic timestamps (one per count, stacked at
        the current time) so the recency-weighted formula degrades
        gracefully: a cold restart treats all prior fetches as equally aged,
        which is conservative (underestimates hotness, never over-pins)."""
        now = time.time()
        half_life = self.cfg.get("fetch_decay_half_life_hours", 24) * 3600
        for ip in sorted(self.meta_dir.glob("*.json")):
            safe_sid = ip.stem
            idx = self._read_idx_file(ip)
            for bid, meta in idx.get("blobs", {}).items():
                count = meta.get("fetch_count", 0)
                weight = meta.get("fetch_weight", 0.0)
                if count > 0 and weight > 0:
                    # Reconstruct a synthetic timestamp at the weighted-mean
                    # age so the recency formula produces the persisted weight.
                    # weight = count * 0.5^(age/half_life)
                    # => age = half_life * log2(count/weight)
                    import math
                    ratio = count / weight if weight > 0 else 1.0
                    age = half_life * math.log2(max(ratio, 1.0))
                    ts = now - age
                    key = (safe_sid, bid)
                    self._fetch_log[key] = [ts] * count

    def _flush_fetch_log(self, safe_sid: str, idx: dict) -> bool:
        """Write fetch-count snapshots into the session index for persistence.

        Returns True if any index entry was modified, so the caller can
        force a write even when no sweep/tombstone change occurred.

        T1.1: respect the immediately-persisted ``fetch_count`` (incremented
        on every fetch in ``_refresh_blob``) and the once-only
        ``first_fetch_ts`` written there too. The in-memory ``_fetch_log``
        is a 7-day sliding window used purely for the recency-weighted
        ``fetch_weight``; its ``len(times)`` is NOT the total fetch count
        after a restart, so we preserve the larger persisted total."""
        now = time.time()
        half_life = self.cfg.get("fetch_decay_half_life_hours", 24) * 3600
        blobs = idx.get("blobs", {})
        modified = False
        for bid in blobs:
            key = (safe_sid, bid)
            times = self._fetch_log.get(key, [])
            if times and half_life > 0:
                weighted = sum(0.5 ** ((now - t) / half_life) for t in times)
                entry = blobs[bid]
                # Keep the in-memory window length AND the persisted total in
                # sync: prefer the larger so a restart that lost timestamps
                # does not shrink the count.
                existing_count = int(entry.get("fetch_count", 0) or 0)
                merged_count = max(len(times), existing_count)
                if merged_count != existing_count or \
                        entry.get("fetch_weight") != round(weighted, 4):
                    entry["fetch_count"] = merged_count
                    entry["fetch_weight"] = round(weighted, 4)
                    modified = True
        return modified
