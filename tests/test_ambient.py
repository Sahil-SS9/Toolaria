"""Tests for the ambient-toolset self-registration guard.

The rescue hook fires ungated in every session, so rescuer_fetch must be made
ambient (always reachable) or restricted sessions get unredeemable handles.
``_mark_rescuer_ambient`` performs that registration host-agnostically: it marks
the toolset ambient when the host supports it, and fails LOUD (never silent)
when it does not.
"""
import sys
import types

import pytest


def _install_fake_registry(monkeypatch, registry_obj):
    mod_tools = types.ModuleType("tools")
    mod_reg = types.ModuleType("tools.registry")
    mod_reg.registry = registry_obj
    monkeypatch.setitem(sys.modules, "tools", mod_tools)
    monkeypatch.setitem(sys.modules, "tools.registry", mod_reg)


class _RecordingRegistry:
    def __init__(self):
        self.calls = []

    def mark_ambient(self, toolset):
        self.calls.append(toolset)


def test_marks_rescuer_ambient_when_supported(toolaria, monkeypatch):
    reg = _RecordingRegistry()
    _install_fake_registry(monkeypatch, reg)
    toolaria._mark_rescuer_ambient()
    assert reg.calls == ["rescuer"], "must mark exactly the rescuer toolset ambient"


def test_warns_loud_when_host_lacks_ambient(toolaria, monkeypatch, caplog):
    class _NoAmbient:  # no mark_ambient attribute
        pass
    _install_fake_registry(monkeypatch, _NoAmbient())
    with caplog.at_level("WARNING"):
        toolaria._mark_rescuer_ambient()  # must not raise
    assert "ambient" in caplog.text.lower()
    assert "unavailable" in caplog.text.lower() or "restricted" in caplog.text.lower()


def test_safe_when_registry_unimportable(toolaria, monkeypatch, caplog):
    mod_tools = types.ModuleType("tools")
    mod_reg = types.ModuleType("tools.registry")  # deliberately no `registry` attr
    monkeypatch.setitem(sys.modules, "tools", mod_tools)
    monkeypatch.setitem(sys.modules, "tools.registry", mod_reg)
    with caplog.at_level("WARNING"):
        toolaria._mark_rescuer_ambient()  # must not raise
    assert "unavailable" in caplog.text.lower()


def test_register_invokes_ambient_marking(toolaria, monkeypatch, base_cfg, fake_ctx_cls):
    reg = _RecordingRegistry()
    _install_fake_registry(monkeypatch, reg)
    fc = fake_ctx_cls({"toolaria": base_cfg})
    toolaria.register(fc)
    assert reg.calls == ["rescuer"], "register() must mark the rescuer toolset ambient"
    # And the tool it depends on is actually registered under that toolset.
    assert "rescuer_fetch" in fc.tools
