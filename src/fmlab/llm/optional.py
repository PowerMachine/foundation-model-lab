"""Helpers for actionable optional-dependency errors."""

from __future__ import annotations

import importlib
from types import ModuleType


class OptionalDependencyError(RuntimeError):
    """Raised when a non-smoke workflow needs an uninstalled extra."""


def require_optional(module_name: str, *, extra: str, purpose: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalDependencyError(
            f"{purpose} requires optional package '{module_name}'. "
            f"Install it with: pip install -e '.[{extra}]'"
        ) from exc
