from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .paths import LabPaths


_ENV_REFERENCE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})")


def _expand_config_value(value: Any, *, defaults: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _expand_config_value(item, defaults=defaults) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_value(item, defaults=defaults) for item in value]
    if not isinstance(value, str):
        return value

    expanded = value
    for name, replacement in defaults.items():
        expanded = expanded.replace(f"${{{name}}}", replacement)
        expanded = re.sub(rf"\${re.escape(name)}\b", lambda _: replacement, expanded)
    expanded = os.path.expanduser(os.path.expandvars(expanded))
    unresolved = _ENV_REFERENCE.findall(expanded)
    if unresolved:
        raise ValueError(f"Unresolved environment variables in config value: {unresolved}")
    return expanded


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {config_path}")
    paths = LabPaths.from_env()
    defaults = {
        "FMLAB_DATA_ROOT": str(paths.data_root),
        "FMLAB_MODEL_ROOT": str(paths.model_root),
    }
    return _expand_config_value(value, defaults=defaults)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
