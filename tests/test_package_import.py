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
_PRELUDE = [
    "import importlib",
    "import importlib.util",
    "import sys",
    "from pathlib import Path",
    f"plugin_dir = Path({str(_PLUGIN_DIR)!r})",
    f"init_path = Path({str(_PLUGIN_DIR / '__init__.py')!r})",
    f"blocked = {{{str(_PLUGIN_DIR)!r}, {str(_PACKAGE_PARENT)!r}}}",
    "sys.path = [p for p in sys.path if p not in blocked]",
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


def _run_in_package_mode(body, tmp_path):
    """Run *body* in a subprocess with Toolaria loaded as a package."""
    code = "\n".join(_PRELUDE + body)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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
