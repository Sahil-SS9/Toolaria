"""SHA256-addressed blob store for rescued tool results. Global keys, per-session indexes."""
import hashlib
import json
import os
import re
import tempfile
import time
import threading
from pathlib import Path

_LOCK = threading.Lock()
_BLOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")

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
        self.blob_dir = bp / "blobs"
        self.meta_dir = bp / "sessions"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    # ── blob i/o ──────────────────────────

    def put(self, content: str, tool_name: str = "", session_id: str = "") -> str:
        """Store content, return short blob_id (first 12 hex of SHA256).

        *session_id* is the owning session; callers must pass it so the
        per-session index stays correct under concurrent sessions."""
        if isinstance(content, str):
            raw = content.encode("utf-8")
        else:
            raw = content
        bhash = hashlib.sha256(raw).hexdigest()
        bid = bhash[:12]
        bpath = self.blob_dir / bid
        sid = session_id or "unknown"
        with _LOCK:
            if not bpath.exists():
                bpath.write_bytes(raw)
            idx = self._load_idx(sid)
            idx.setdefault("blobs", {})
            idx["blobs"][bid] = {
                "t": time.time(),
                "tool": tool_name,
                "size": len(raw),
                "hash": bhash,
            }
            self._save_idx(idx, sid)
        return bid

    def _find_meta(self, blob_id: str, session_id: str = "") -> dict:
        """Index metadata for a blob: the given session's entry, or the first
        entry found across all sessions (blobs are content-addressed, so any
        session's metadata describes the same bytes)."""
        if session_id:
            meta = self._load_idx(session_id).get("blobs", {}).get(blob_id)
            if meta:
                return meta
        for sf in sorted(self.meta_dir.glob("*.json")):
            try:
                idx = json.loads(sf.read_text())
            except Exception:
                continue
            meta = idx.get("blobs", {}).get(blob_id)
            if meta:
                return meta
        return {}

    def fetch(self, blob_id: str, mode: str, start=0, count=20,
              pattern=None, session_id: str = ""):
        """Retrieve a slice of a blob. Modes: range, grep, stat, full."""
        if not _BLOB_ID_RE.match(blob_id):
            return f"Error: invalid blob id '{blob_id}' (expected 12 hex chars)"

        cap = self.cfg.get("fetch_max_chars", 4000)
        bpath = self.blob_dir / blob_id
        if not bpath.exists():
            return f"Error: blob {blob_id} not found (may have been swept)"

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
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"Error: blob {blob_id} is binary ({len(raw)} bytes)"

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

        if mode == "range":
            total = len(lines)
            start = max(0, start)
            note = ""
            if start >= total and total > 0:
                note = f"[start {start} past end; clamped]\n"
                start = max(0, total - max(1, count))
            end = min(total, start + max(1, count))
            body = "\n".join(lines[start:end])[:cap]
            return (
                f"{note}[lines {start}..{end - 1} of {total}]\n{body}"
            )

        if mode == "grep":
            if not pattern:
                return "Error: grep requires pattern=..."
            return self._grep_safe(lines, pattern, cap)

        return f"Error: unknown mode '{mode}'"

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

    # ── sweep ──────────────────────────────

    def lazy_sweep(self):
        """Remove blobs past TTL or over size limit. Oldest-first.
        Cross-session: a blob is deleted only when NO session index
        still references it."""
        ttl = self.cfg.get("ttl_hours", 72) * 3600
        max_mb = self.cfg.get("max_store_mb", 500)
        now = time.time()
        with _LOCK:
            self._sweep_by_ttl(now, ttl)
            self._sweep_by_size(now, max_mb)

    def _sweep_by_ttl(self, now, ttl):
        # Collect global reference count per blob across ALL sessions
        ref_count: dict[str, int] = {}
        for sf in sorted(self.meta_dir.glob("*.json")):
            try:
                idx = json.loads(sf.read_text())
            except Exception:
                continue
            for bid in idx.get("blobs", {}):
                ref_count[bid] = ref_count.get(bid, 0) + 1

        # For each session, remove TTL-expired entries from index
        for sf in sorted(self.meta_dir.glob("*.json")):
            try:
                idx = json.loads(sf.read_text())
            except Exception:
                continue
            blobs = idx.get("blobs", {})
            removed = 0
            for bid, meta in list(blobs.items()):
                if now - meta.get("t", 0) > ttl:
                    del blobs[bid]
                    ref_count[bid] = ref_count.get(bid, 0) - 1
                    removed += 1
            if removed:
                sf.write_text(json.dumps(idx))

        # Only delete files when no session references them
        for bid, count in ref_count.items():
            if count <= 0:
                bpath = self.blob_dir / bid
                if bpath.exists():
                    bpath.unlink()

    def _sweep_by_size(self, now, max_mb):
        max_bytes = max_mb * 1024 * 1024
        all_blobs = []
        for bf in self.blob_dir.iterdir():
            if bf.is_file() and len(bf.name) == 12:
                all_blobs.append((bf.stat().st_ctime, bf.stat().st_size, bf))
        all_blobs.sort()  # oldest first
        total = sum(sz for _, sz, _ in all_blobs)
        for _, sz, bf in all_blobs:
            if total <= max_bytes:
                break
            # Only evict if no session references it
            if self._any_session_refs(bf.name):
                continue
            if bf.exists():
                bf.unlink()
            total -= sz

    def _any_session_refs(self, bid: str) -> bool:
        """True if any session index still references this blob."""
        for sf in self.meta_dir.glob("*.json"):
            try:
                idx = json.loads(sf.read_text())
            except Exception:
                continue
            if bid in idx.get("blobs", {}):
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

    def _load_idx(self, session_id: str):
        ip = self._idx_path(session_id)
        if ip.exists():
            try:
                return json.loads(ip.read_text())
            except Exception:
                pass
        return {}

    def _save_idx(self, idx: dict, session_id: str):
        # Atomic write: temp file + os.replace so a concurrent reader from
        # another session thread never sees partial JSON.
        target = self._idx_path(session_id)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(idx, f)
            os.replace(tmp, target)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
