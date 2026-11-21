"""Resolve the Hermes home without inventing a second store.

``HERMES_HOME`` is authoritative whenever the runtime exports it. The failure
this module exists to stop: profile-scoped plugin loads sometimes ran without
``HERMES_HOME`` set, so the old ``HERMES_HOME or ~/.hermes`` fallback resolved
to the POSIX dot-directory and created an empty parallel store beside the real
install (on Windows, ``C:\\Users\\<user>\\.hermes\\toolaria``).

Resolution order:

1. ``HERMES_HOME`` if set and non-empty.
2. The OS launch-scope location: ``%LOCALAPPDATA%\\hermes`` on Windows.
3. The historic ``~/.hermes``, logged as a warning — a store created there is
   invisible to the launch scope.

Never creates anything; callers mkdir as they already did.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["hermes_home", "default_store_path"]


def hermes_home() -> Path:
    """Return the Hermes home directory for this process."""
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env).expanduser()

    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidate = Path(local) / "hermes"
            if candidate.is_dir():
                return candidate

    legacy = Path(os.path.expanduser("~/.hermes"))
    logger.warning(
        "toolaria: HERMES_HOME is unset and no launch-scope home was found; "
        "falling back to %s. A store created there is not shared with the "
        "launch scope — set HERMES_HOME in this process's environment.",
        legacy,
    )
    return legacy


def default_store_path() -> Path:
    """Return the blob store path under :func:`hermes_home`."""
    return hermes_home() / "toolaria"
