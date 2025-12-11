"""Utilities shared by the local AffectGPT-compatible modules."""

from __future__ import annotations

import sys
from types import ModuleType


def get_active_config() -> ModuleType:
    """Return the config module populated by `load_config`."""
    cfg = sys.modules.get("config")
    if cfg is None:
        raise RuntimeError(
            "Affect config has not been loaded. "
            "Call `load_config` before importing local AffectGPT modules."
        )
    return cfg

