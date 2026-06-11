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
# Allow spaces so multi-word patterns work (e.g. "error failed").
# Single-line regex is bounded by quantifier rejection and 500ms wall clock.
_SAFE_PAT = re.compile(r"^[a-zA-Z0-9_.*?^$()\[\]{}\\\|+\-/ ]+$")
_NESTED_QUANT_RE = re.compile(r"[*+?{}]\s*[*+?{}]|\{[^}]*\}[*+?{]|\+\+|\*\*|\?\?")
_BLOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class BlobStore:
    def __init__(self, cfg: dict, session_id: str):
        self.cfg = cfg
        bp = Path(cfg.get("store_path", "~/.hermes/toolaria")).expanduser().resolve()
        self.blob_dir = bp / "blobs"
        self.meta_dir = bp / "sessions"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id  # startup default only — callers pass sid explicitly

    # ── blob i/o ──────────────────────────

    def put(self, content: str, tool_name: str = "", session_id: str = "") -> str:
        """Store content, return short blob_id (first 12 hex of SHA256).

        *session_id* is the owning session — required for correct per-session
        indexing.  Must be passed explicitly by every caller; the instance
        ``_session_id`` is a startup default only and must never be mutated
        by concurrent callers."""
        if isinstance(content, str):
            raw = content.encode("utf-8")
        else:
            raw = content
        bhash = hashlib.sha256(raw).hexdigest()
        bid = bhash[:12]
        bpath = self.blob_dir / bid
        sid = session_id or self._session_id
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

    def fetch(self, blob_id: str, mode: str, start=0, count=20,
              pattern=None, cap=4000, session_id: str = ""):
        """Retrieve slice of a blob. modes: range, grep, stat, full.
        blob_id is validated at the caller (plugin __init__._fetch)."""
        sid = session_id or self._session_id
        if not _BLOB_ID_RE.match(blob_id):
            return f"Error: invalid blob id '{blob_id}' — expected 12 hex chars"

        bpath = self.blob_dir / blob_id
        if not bpath.exists():
            return f"Error: blob {blob_id} not found (may have been swept)"

        if mode == "stat":
            st = bpath.stat()
            idx = self._load_idx(sid)
            meta = idx.get("blobs", {}).get(blob_id, {})
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
            return text[:cap]

        lines = text.splitlines()

        if mode == "range":
            start = max(0, start)
            end = start + max(1, count)
            return "\n".join(lines[start:end])[:cap]

        if mode == "grep":
            if not pattern:
                return "Error: grep requires pattern=..."
            return self._grep_safe(lines, pattern, cap)

        return f"Error: unknown mode '{mode}'"

    # ── grep with timeout/complexity cap ───

    def _grep_safe(self, lines: list, pattern: str, cap: int) -> str:
        """Regex grep with 500ms wall-clock timeout and anti-backtracking caps.
        Timeout check runs BETWEEN lines — a single-line regex is bounded by
        the per-line length (capped by tool output limits upstream) and by
        nested quantifier rejection below."""
        plen = len(pattern)
        pmax = self.cfg.get("grep_max_pattern_len", 80)
        if plen > pmax:
            return f"Error: pattern too long ({plen} > {pmax})"
        if not _SAFE_PAT.match(pattern):
            return "Error: pattern contains unsafe characters"
        if _NESTED_QUANT_RE.search(pattern):
            return "Error: nested quantifiers detected — risk of catastrophic backtracking"

        timeout = self.cfg.get("grep_timeout_ms", 500) / 1000.0
        t0 = time.time()
        results = []
        try:
            preg = re.compile(pattern, re.I)
        except re.error as e:
            return f"Error: invalid regex: {e}"

        for line in lines:
            if time.time() - t0 > timeout:
                results.append(f"[grep timed out after {timeout}s — {len(results)} matches]")
                break
            if preg.search(line):
                results.append(line[:500])
                if len(results) >= 50:
                    results.append("[50 matches — capped]")
                    break
        if not results:
            return f"[no matches for pattern '{pattern}']"
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

    def _idx_path(self, session_id: str = ""):
        return self.meta_dir / f"{session_id or self._session_id}.json"

    def _load_idx(self, session_id: str = ""):
        ip = self._idx_path(session_id or self._session_id)
        if ip.exists():
            try:
                return json.loads(ip.read_text())
            except Exception:
                pass
        return {}

    def _save_idx(self, idx: dict, session_id: str = ""):
        # Atomic write: temp file + os.replace to prevent concurrent read
        # of partial JSON from a different session thread.
        import tempfile
        target = self._idx_path(session_id or self._session_id)
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
