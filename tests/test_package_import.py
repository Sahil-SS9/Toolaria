"""Regression coverage for package-style Toolaria loading."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_PACKAGE_PARENT = _PLUGIN_DIR.parent

# ``cryptography`` is an optional extra (see requirements.txt), so only the
# rotation half of this coverage can assume it. Mirrors the ``_HAVE_FERNET``
# gates in test_t4_2.py / test_reviewer_fixes.py; probed with find_spec so
# this module still imports nothing from the plugin at collection time.
_HAVE_FERNET = importlib.util.find_spec("cryptography") is not None

needs_crypto = pytest.mark.skipif(
    not _HAVE_FERNET,
    reason="requires the cryptography package (encryption tier)")


# Load the plugin the way Hermes does — spec_from_file_location under the
# explicit name "toolaria" with submodule_search_locations — while keeping
# both the plugin dir and its parent off sys.path. Naming the package
# explicitly (rather than ``import <dirname>``) keeps this independent of
# what the checkout directory happens to be called.
_PACKAGE_SETUP = [
    "import importlib",
    "import importlib.util",
    "import sys",
    "from pathlib import Path",
    f"plugin_dir = Path({str(_PLUGIN_DIR)!r})",
    f"init_path = Path({str(_PLUGIN_DIR / '__init__.py')!r})",
    f"blocked = {{{str(_PLUGIN_DIR)!r}, {str(_PACKAGE_PARENT)!r}}}",
    "sys.path = [p for p in sys.path if p not in blocked]",
]

_PACKAGE_LOAD = [
    "spec = importlib.util.spec_from_file_location(",
    "    'toolaria', init_path, submodule_search_locations=[str(plugin_dir)]",
    ")",
    "assert spec is not None and spec.loader is not None",
    "toolaria = importlib.util.module_from_spec(spec)",
    "sys.modules['toolaria'] = toolaria",
    "spec.loader.exec_module(toolaria)",
    "assert toolaria.__package__ == 'toolaria'",
    "assert Path(toolaria.__file__).resolve() == init_path",
]


def _run_in_package_mode(body, tmp_path, *, setup=(), expect_success=True):
    """Run *body* in a bounded, minimal-env package subprocess."""
    code = "\n".join(_PACKAGE_SETUP + list(setup) + _PACKAGE_LOAD + body)
    env = {
        "HOME": str(tmp_path / "home"),
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
    }

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    if expect_success:
        assert result.returncode == 0, result.stderr
    return result


def _write_conflicting_module(tmp_path, module_name):
    (tmp_path / f"{module_name}.py").write_text(
        f"raise AssertionError('conflicting top-level {module_name} imported')\n",
        encoding="utf-8",
    )


def _block_package_module(module_name):
    return [
        "class _BlockRelativeImport:",
        "    def find_spec(self, fullname, path=None, target=None):",
        f"        if fullname == 'toolaria.{module_name}':",
        "            raise ImportError('blocked internal relative import')",
        "        return None",
        "sys.meta_path.insert(0, _BlockRelativeImport())",
    ]


def test_package_import_without_plugin_dir_on_sys_path(tmp_path):
    """Package loading and lazy imports must use package-relative siblings.

    Covers the import-time chain plus the function-level imports that only
    fire at request time: the passref enforcement/confirmation gates, the
    credential-refusal markers pulled inside ``BlobStore.fetch``, and the
    entity-binding imports at the top of the passref middleware.
    """
    _run_in_package_mode([
        "toolaria._merge_cfg({})",
        "passref = importlib.import_module('toolaria.passref')",
        "assert passref._credential_enforcement_active({'enforcement_enabled': True})",
        "assert passref._confirmation_required({'confirmation_required': True})",
        f"store_path = Path({str(tmp_path / 'toolaria-store')!r})",
        "store = toolaria.BlobStore({",
        "    'store_path': str(store_path),",
        "    'enforcement_enabled': True,",
        "})",
        "bid = store.put('credential data', tool_name='send_email',",
        "                 session_id='package-session', label='credential')",
        "refused = store.fetch(bid, 'full', session_id='package-session')",
        "assert refused == (passref.CREDENTIAL_REFUSE_MARKER_PREFIX + bid",
        "                  + passref.CREDENTIAL_REFUSE_MARKER_SUFFIX)",
        "middleware = passref.make_middleware(",
        "    lambda: store, {'passref_enabled': True}, frozenset())",
        "assert middleware(tool_name='summarise', args={},",
        "                  session_id='package-session') is None",
    ], tmp_path)


def test_package_import_does_not_redirect_to_conflicting_top_level_module(tmp_path):
    """A package import failure must not fall back to another checkout."""
    _write_conflicting_module(tmp_path, "blobstore")
    result = _run_in_package_mode(
        [], tmp_path, setup=_block_package_module("blobstore"),
        expect_success=False,
    )
    assert result.returncode != 0
    assert "blocked internal relative import" in result.stderr
    assert "conflicting top-level blobstore imported" not in result.stderr


def test_lazy_package_import_error_is_not_redirected(tmp_path):
    """Lazy package imports must surface internal errors, not loose siblings."""
    _write_conflicting_module(tmp_path, "passref")
    result = _run_in_package_mode(
        [
            "sys.modules.pop('toolaria.passref', None)",
            "delattr(toolaria, 'passref') if hasattr(toolaria, 'passref') else None",
            f"store = toolaria.BlobStore({{'store_path': {str(tmp_path / 'store')!r},",
            "    'enforcement_enabled': True})",
            "bid = store.put('credential data', 'send_email',",
            "                 session_id='lazy-session', label='credential')",
            "store.fetch(bid, 'full', session_id='lazy-session')",
        ],
        tmp_path,
        setup=_block_package_module("passref"),
        expect_success=False,
    )
    assert result.returncode != 0
    assert "blocked internal relative import" in result.stderr
    assert "conflicting top-level passref imported" not in result.stderr


def test_merge_cfg_import_error_is_not_redirected(tmp_path):
    """Config validation must surface blocked package imports, not redirect."""
    _write_conflicting_module(tmp_path, "labels")
    result = _run_in_package_mode(
        [
            "sys.modules.pop('toolaria.labels', None)",
            "delattr(toolaria, 'labels') if hasattr(toolaria, 'labels') else None",
            "toolaria._merge_cfg({})",
        ],
        tmp_path,
        setup=_block_package_module("labels"),
        expect_success=False,
    )
    assert result.returncode != 0
    assert "blocked internal relative import" in result.stderr
    assert "conflicting top-level labels imported" not in result.stderr


@needs_crypto
def test_package_key_rotation_ledger_import(tmp_path):
    """``rotate_key`` writes its ledger row through a package-relative import.

    Split from the main case because ``rotate_key`` hard-requires Fernet,
    so this is the one lazy import that cannot be exercised without the
    optional ``cryptography`` extra installed.
    """
    _run_in_package_mode([
        f"store_path = Path({str(tmp_path / 'toolaria-store')!r})",
        f"old_key_path = Path({str(tmp_path / 'old.key')!r})",
        f"new_key_path = Path({str(tmp_path / 'new.key')!r})",
        "from cryptography.fernet import Fernet",
        "old_key_path.write_bytes(Fernet.generate_key())",
        "store = toolaria.BlobStore({",
        "    'store_path': str(store_path),",
        "    'toolaria_key_file': str(old_key_path),",
        "    'enforcement_enabled': True,",
        "})",
        "assert store.rotate_key(str(new_key_path)) == 0",
        "rotation_ledger = store_path / 'ledger' / 'key_rotations.jsonl'",
        "assert rotation_ledger.exists()",
        "assert 'key_rotated' in rotation_ledger.read_text()",
    ], tmp_path)
