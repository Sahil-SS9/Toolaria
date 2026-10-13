"""Regressions for shared-store sweep races and repeated index I/O (#8/#9).

Process tests pin the actual byte-write/index-save and deletion interleavings.
They use disposable stores; no host processes or live Hermes state are touched.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
from pathlib import Path
import time
from unittest.mock import patch

import pytest

import blobstore as bs
from blobstore import BlobStore


POSIX_PROCESSES = pytest.mark.skipif(
    not bs._HAVE_FCNTL, reason="cross-process store coordination requires POSIX flock")


def _config(path):
    return {"store_path": str(path), "ttl_hours": 72,
            "tombstone_ttl_hours": 720, "max_store_mb": 500,
            "full_fetch_max_chars": 50000, "enforcement_enabled": False}


def _finish(process, connection):
    assert connection.poll(10), "store worker did not return"
    result = connection.recv()
    process.join(10)
    assert process.exitcode == 0, result
    assert "error" not in result, result
    return result


def _stop(processes):
    for process in processes:
        process.join(1)
        if process.is_alive():
            process.terminate()
            process.join(5)


def _writer(cfg, text, sid, published, release, attempted, connection):
    try:
        if attempted is not None:
            attempted.set()
        store = BlobStore(cfg)
        if published is not None:
            save = store._save_idx

            def paused_save(idx, session_id):
                published.set()
                assert release.wait(10), "writer was not released"
                return save(idx, session_id)

            store._save_idx = paused_save
        bid = store.put(text, "web_extract", session_id=sid)
        connection.send({"bid": bid,
                         "fetch": store.fetch(bid, "full", session_id=sid)})
    except BaseException as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def _sweeper(cfg, attempted, connection):
    try:
        attempted.set()
        store = BlobStore(cfg)
        store.lazy_sweep()
        connection.send({"swept": True})
    except BaseException as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def _launch(context, target, args):
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(*args, child))
    process.start()
    child.close()
    return process, parent


def test_initial_sweep_failure_does_not_disable_rescuing(
        toolaria, fake_ctx_cls, base_cfg, monkeypatch, caplog):
    def failed_sweep(self):
        raise FileNotFoundError("a concurrent sweeper removed the old blob")

    monkeypatch.setattr(toolaria.BlobStore, "lazy_sweep", failed_sweep)
    context = fake_ctx_cls({"toolaria": base_cfg})
    toolaria.register(context)
    rescued = context.hooks["transform_tool_result"](
        tool_name="web_extract", result="diagnostic line\n" * 1000,
        session_id="startup")
    assert toolaria._store is not None, "a maintenance failure disabled a usable store"
    assert rescued and "tool result rescued" in rescued
    assert "sweep" in caplog.text.lower()
    assert "rescuing disabled" not in caplog.text


@pytest.mark.parametrize("mode", ["ttl", "size"])
def test_sweep_tolerates_a_blob_already_removed_at_unlink(tmp_path, monkeypatch, mode):
    cfg = _config(tmp_path / "store")
    cfg.update(ttl_hours=1, max_store_mb=0 if mode == "size" else 500)
    store = BlobStore(cfg)
    bid = store.put("shared expired result", "web_extract", session_id="owner")
    if mode == "ttl":
        idx = store._load_idx("owner")
        idx["blobs"][bid]["t"] = 1
        store._save_idx(idx, "owner")
    target = store.blob_dir / bid
    original = Path.unlink

    def raced_unlink(path, *args, **kwargs):
        if path == target and path.exists():
            original(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", raced_unlink)
    store.lazy_sweep()
    assert not target.exists()
    assert "swept_at" in store._load_idx("owner")["blobs"][bid]


def test_size_sweep_tolerates_disappearance_between_is_file_and_stat(tmp_path, monkeypatch):
    store = BlobStore(_config(tmp_path / "store"))
    bid = store.put("a size-sweep race", "web_extract", session_id="owner")
    target = store.blob_dir / bid
    monkeypatch.setattr(store, "_sweep_by_ttl", lambda *args: None)
    original = Path.is_file

    def raced_is_file(path, *args, **kwargs):
        is_file = original(path, *args, **kwargs)
        if path == target and is_file:
            target.unlink()
        return is_file

    monkeypatch.setattr(Path, "is_file", raced_is_file)
    store.lazy_sweep()
    assert not target.exists()


@pytest.mark.parametrize("mode", ["ttl", "size"])
def test_sweep_reads_each_session_index_only_a_constant_number_of_times(tmp_path, monkeypatch, mode):
    cfg = _config(tmp_path / "store")
    if mode == "size":
        cfg["max_store_mb"] = 0
    store = BlobStore(cfg)
    blobs, sessions = 40, 80
    entries = {}
    for i in range(blobs):
        raw = f"expired result {i}".encode()
        digest = hashlib.sha256(raw).hexdigest()
        bid = digest[:12]
        (store.blob_dir / bid).write_bytes(raw)
        entries[bid] = {"t": 1 if mode == "ttl" else time.time(),
                        "tool": "web_extract", "size": len(raw),
                        "hash": digest, "label": "public"}
    for i in range(sessions):
        store._save_idx({"blobs": entries if i == sessions - 1 else {}}, f"s-{i}")
    original = store._read_idx_file
    reads = 0

    def counted(path):
        nonlocal reads
        reads += 1
        return original(path)

    monkeypatch.setattr(store, "_read_idx_file", counted)
    store.lazy_sweep()
    assert reads <= 2 * sessions, f"sweep reread indexes per blob: {reads} reads"
    assert not list(store.blob_dir.iterdir())


@POSIX_PROCESSES
def test_sweep_cannot_delete_a_blob_between_publication_and_index_save(tmp_path):
    context = mp.get_context("fork")
    cfg = _config(tmp_path / "store")
    store = BlobStore(cfg)
    published, release, attempted = context.Event(), context.Event(), context.Event()
    text = "a new live result published during cleanup"
    writer, writer_pipe = _launch(context, _writer,
                                  (cfg, text, "writer", published, release, None))
    processes = [writer]
    try:
        assert published.wait(10), "writer never reached index publication"
        sweeper, sweep_pipe = _launch(context, _sweeper, (cfg, attempted))
        processes.append(sweeper)
        assert attempted.wait(10), "sweep was not attempted"
        # Old code completes and deletes the unindexed blob here; guarded
        # code waits for the writer's publication. The content assertion
        # below, not this bounded scheduling wait, is the correctness gate.
        sweep_pipe.poll(0.5)
        release.set()
        written = _finish(writer, writer_pipe)
        _finish(sweeper, sweep_pipe)
        bid = written["bid"]
        assert written["fetch"] == text, "writer returned a dead handle"
        assert store.session_references(bid, "writer")
        assert store.fetch(bid, "full", session_id="writer") == text
    finally:
        release.set()
        _stop(processes)


@POSIX_PROCESSES
def test_concurrent_puts_preserve_both_references_in_one_session(tmp_path):
    context = mp.get_context("fork")
    cfg = _config(tmp_path / "store")
    store = BlobStore(cfg)
    published, release, attempted = context.Event(), context.Event(), context.Event()
    first, first_pipe = _launch(context, _writer,
                                (cfg, "first result", "shared", published, release, None))
    processes = [first]
    try:
        assert published.wait(10)
        second, second_pipe = _launch(context, _writer,
                                      (cfg, "second result", "shared", None, None, attempted))
        processes.append(second)
        assert attempted.wait(10)
        second_pipe.poll(0.5)
        release.set()
        first_result, second_result = _finish(first, first_pipe), _finish(second, second_pipe)
        for result, content in [(first_result, "first result"), (second_result, "second result")]:
            assert store.session_references(result["bid"], "shared"), "a concurrent put lost its index entry"
            assert store.fetch(result["bid"], "full", session_id="shared") == content
    finally:
        release.set()
        _stop(processes)


def test_live_reference_in_another_session_prevents_deletion(tmp_path):
    store = BlobStore(_config(tmp_path / "store"))
    content = "shared content with a fresh owner"
    bid = store.put(content, "web_extract", session_id="old-owner")
    store.put(content, "web_extract", session_id="fresh-owner")
    idx = store._load_idx("old-owner")
    idx["blobs"][bid]["t"] = time.time() - 100 * 3600
    store._save_idx(idx, "old-owner")
    store.lazy_sweep()
    assert "swept_at" in store._load_idx("old-owner")["blobs"][bid]
    assert store.fetch(bid, "full", session_id="fresh-owner") == content


def _paused_sweeper(cfg, first_bid, deleting, release, connection):
    try:
        store = BlobStore(cfg)
        original = Path.unlink

        def pause_first_delete(path, *args, **kwargs):
            if path == store.blob_dir / first_bid:
                deleting.set()
                assert release.wait(10), "sweeper was not released"
            return original(path, *args, **kwargs)

        iterdir = Path.iterdir

        def ordered_blobs(path):
            if path == store.blob_dir:
                yield path / first_bid
                yield from (p for p in iterdir(path) if p.name != first_bid)
            else:
                yield from iterdir(path)

        with patch.object(Path, "unlink", pause_first_delete), \
                patch.object(Path, "iterdir", ordered_blobs):
            store.lazy_sweep()
        connection.send({"swept": True})
    except BaseException as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


@POSIX_PROCESSES
def test_concurrent_adoption_survives_the_live_reference_snapshot(tmp_path):
    context = mp.get_context("fork")
    cfg = _config(tmp_path / "store")
    store = BlobStore(cfg)
    text = "orphan adopted while the sweep deletes another orphan"
    orphan_ids = []
    for content in ["first orphan", text]:
        bid = hashlib.sha256(content.encode()).hexdigest()[:12]
        (store.blob_dir / bid).write_text(content)
        orphan_ids.append(bid)
    deleting, release, attempted = context.Event(), context.Event(), context.Event()
    sweeper, sweep_pipe = _launch(context, _paused_sweeper,
                                  (cfg, orphan_ids[0], deleting, release))
    processes = [sweeper]
    try:
        assert deleting.wait(10)
        writer, writer_pipe = _launch(context, _writer,
                                      (cfg, text, "adopter", None, None, attempted))
        processes.append(writer)
        assert attempted.wait(10)
        writer_pipe.poll(0.5)
        release.set()
        written = _finish(writer, writer_pipe)
        _finish(sweeper, sweep_pipe)
        assert written["fetch"] == text
        assert store.fetch(written["bid"], "full", session_id="adopter") == text
    finally:
        release.set()
        _stop(processes)


@POSIX_PROCESSES
def test_forked_writer_does_not_inherit_the_parents_lock_ownership(tmp_path):
    context = mp.get_context("fork")
    cfg = _config(tmp_path / "store")
    store = BlobStore(cfg)
    attempted = context.Event()
    processes = []
    try:
        with store._store_guard():
            writer, pipe = _launch(context, _writer,
                                    (cfg, "after parent unlock", "child", None, None, attempted))
            processes.append(writer)
            assert attempted.wait(10)
            assert not pipe.poll(0.2), "child bypassed its parent's active flock"
        written = _finish(writer, pipe)
        assert written["fetch"] == "after parent unlock"
    finally:
        _stop(processes)


@POSIX_PROCESSES
def test_lock_failure_does_not_publish_an_unguarded_blob(tmp_path, monkeypatch):
    store = BlobStore(_config(tmp_path / "store"))

    def denied_lock(fd, operation):
        raise PermissionError("store lock could not be acquired")

    monkeypatch.setattr(bs.fcntl, "flock", denied_lock)
    with pytest.raises(PermissionError, match="could not be acquired"):
        store.put("not published", "web_extract", session_id="owner")
    assert not list(store.blob_dir.iterdir())
    assert not store._load_idx("owner")


@POSIX_PROCESSES
def test_store_lock_is_private_and_keeps_the_same_inode(tmp_path):
    store = BlobStore(_config(tmp_path / "store"))
    before = store._store_lock_path.stat()
    assert before.st_mode & 0o777 == 0o600
    store.put("stable inode", "web_extract", session_id="owner")
    store.lazy_sweep()
    after = store._store_lock_path.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def _register_worker(module, context_cls, cfg, barrier, connection):
    try:
        barrier.wait(timeout=10)
        context = context_cls({"toolaria": cfg})
        module.register(context)
        rescued = context.hooks["transform_tool_result"](
            tool_name="web_extract", result="new diagnostic line\n" * 1000,
            session_id=f"startup-{mp.current_process().pid}")
        connection.send({"store_ready": module._store is not None,
                         "rescued": bool(rescued)})
    except BaseException as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


@POSIX_PROCESSES
def test_two_process_startups_both_keep_a_working_store(tmp_path, toolaria, fake_ctx_cls):
    context = mp.get_context("fork")
    cfg = _config(tmp_path / "store")
    store = BlobStore(cfg)
    bid = store.put("expired before concurrent starts", "web_extract", session_id="old")
    idx = store._load_idx("old")
    idx["blobs"][bid]["t"] = 1
    store._save_idx(idx, "old")
    barrier = context.Barrier(2)
    workers = [_launch(context, _register_worker, (toolaria, fake_ctx_cls, cfg, barrier))
               for _ in range(2)]
    try:
        for process, pipe in workers:
            assert _finish(process, pipe) == {"store_ready": True, "rescued": True}
    finally:
        _stop([process for process, _ in workers])
