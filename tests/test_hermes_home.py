"""Hermes-home resolution must never invent a second store.

Regression: profile-scoped plugin loads ran without ``HERMES_HOME`` set, so the
old ``HERMES_HOME or ~/.hermes`` fallback created an empty parallel store at
``~/.hermes/toolaria`` beside the real install.
"""
import os
from pathlib import Path

import pytest

from toolaria.hermes_home import default_store_path, hermes_home


def test_hermes_home_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    assert hermes_home() == tmp_path / "launch"
    assert default_store_path() == tmp_path / "launch" / "toolaria"


def test_empty_hermes_home_is_not_authoritative(monkeypatch, tmp_path):
    """An empty string must not resolve to the cwd-relative empty path."""
    monkeypatch.setenv("HERMES_HOME", "")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    (tmp_path / "local" / "hermes").mkdir(parents=True)
    assert hermes_home() == tmp_path / "local" / "hermes"


@pytest.mark.skipif(os.name != "nt", reason="LOCALAPPDATA launch scope is Windows-only")
def test_windows_falls_back_to_launch_scope_not_dot_hermes(monkeypatch, tmp_path):
    """The actual regression: unset HERMES_HOME must not mean ~/.hermes."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    launch = tmp_path / "local" / "hermes"
    launch.mkdir(parents=True)

    home = hermes_home()

    assert home == launch
    assert home != Path(os.path.expanduser("~/.hermes"))
    assert str(home).lower().endswith(str(Path("local") / "hermes").lower())


def test_legacy_fallback_warns(monkeypatch, tmp_path, caplog):
    """No HERMES_HOME and no launch scope: fall back loudly, not silently."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "missing"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    with caplog.at_level("WARNING"):
        home = hermes_home()

    assert home == Path(os.path.expanduser("~/.hermes"))
    assert any("HERMES_HOME is unset" in r.getMessage() for r in caplog.records)


def test_resolution_does_not_create_directories(monkeypatch, tmp_path):
    """Resolution is pure: no mkdir, so a probe cannot leave an empty store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "probe"))
    _ = default_store_path()
    assert not (tmp_path / "probe").exists()
