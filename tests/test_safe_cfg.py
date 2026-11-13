"""Plugin settings must come from the host's supported API.

Regression: `_safe_cfg` read ``ctx.config.get("toolaria")``. Current hosts have no
``ctx.config`` at all, so every configured setting was silently ignored and the
plugin fell through to its local defaults — which put profile-scoped loads on the
``~/.hermes`` store.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

import toolaria
from toolaria.blobstore import BlobStore

_PDIR = Path(__file__).resolve().parent.parent


class CtxWithGetConfig:
    """The host shape: get_config(key) over plugins.entries.<id>.settings."""

    def __init__(self, settings):
        self._s = settings
        self.calls = []

    def get_config(self, key, default=None):
        self.calls.append(key)
        return self._s.get(key, default)


class LegacyCtx:
    """The shape toolaria was written against, which no longer exists."""

    def __init__(self, config):
        self.config = config


class ExplodingCtx:
    def get_config(self, key, default=None):
        raise RuntimeError("host said no")


def test_settings_read_through_get_config():
    ctx = CtxWithGetConfig({"store_path": r"C:\store\explicit", "max_result_chars": 900})
    cfg = toolaria._safe_cfg(ctx)
    assert cfg["store_path"] == r"C:\store\explicit"
    assert cfg["max_result_chars"] == 900
    # The plugin-local default keys are probed, so a user override is never missed.
    assert "max_result_chars" in ctx.calls and "excerpt_max_chars" in ctx.calls


def test_get_config_result_drives_the_store(monkeypatch, tmp_path):
    """The whole point: settings set the store path, so no default can win."""
    target = tmp_path / "explicit-store"
    ctx = CtxWithGetConfig({"store_path": str(target)})
    cfg = toolaria._safe_cfg(ctx)
    store = BlobStore(cfg)
    assert store.store_path == target.resolve()


def test_legacy_ctx_shape_is_accepted_but_stays_in_home(monkeypatch, tmp_path):
    """The old ``.config`` dict still works as a compatibility layer — and an
    empty one must not leave the store outside the Hermes home."""
    target = tmp_path / "legacy-store"
    cfg = toolaria._safe_cfg(LegacyCtx({"toolaria": {"store_path": str(target)}}))
    assert cfg["store_path"] == str(target)

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"toolaria": {}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    empty = toolaria._safe_cfg(LegacyCtx({"toolaria": {}}))
    assert empty == {}
    store = BlobStore(toolaria._merge_cfg(empty))
    assert str(store.store_path).startswith(str(home))
    assert "~/.hermes" not in str(store.store_path)


def test_exploding_get_config_falls_back(monkeypatch, tmp_path):
    home = tmp_path / "home2"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"toolaria": {"store_path": str(tmp_path / "fallback")}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert toolaria._safe_cfg(ExplodingCtx())["store_path"] == str(tmp_path / "fallback")


def test_no_sources_returns_empty_and_defaults_apply(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "bare"))
    cfg = toolaria._safe_cfg(CtxWithGetConfig({}))
    assert cfg == {}
    # Empty user layer + plugin defaults must still resolve inside HERMES_HOME.
    store = BlobStore(toolaria._merge_cfg(cfg))
    assert str(store.store_path).startswith(str(tmp_path / "bare"))
    assert "~/.hermes" not in str(store.store_path)
