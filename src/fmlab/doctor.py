from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .artifacts import system_snapshot
from .paths import LabPaths


PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "trl",
    "bitsandbytes",
    "sentencepiece",
    "numpy",
    "matplotlib",
    "Pillow",
    "scikit-learn",
)

KNOWN_MODELS = {
    "qwen_base_05b": "Qwen/Qwen2.5-0.5B",
    "qwen_instruct_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen_instruct_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_instruct_32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen_coder_3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen_coder_7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen_vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen_vl_30b_fp8": "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
    "qwen_vl_32b_fp8": "Qwen/Qwen3-VL-32B-Instruct-FP8",
}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _disk(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": str(path),
            "total_gib": round(usage.total / 2**30, 2),
            "used_gib": round(usage.used / 2**30, 2),
            "free_gib": round(usage.free / 2**30, 2),
        }
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}


def _bwrap_probe() -> dict[str, Any]:
    executable = shutil.which("bwrap")
    if executable is None:
        return {"available": False, "detail": "bwrap executable was not found"}
    command = [
        executable,
        "--unshare-all",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "/bin/true",
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "detail": str(exc)}
    detail = (result.stderr or result.stdout).strip()[:500]
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "detail": detail,
    }


def _podman_probe(workspace: Path) -> dict[str, Any]:
    try:
        from .agent.container_runner import PodmanRunner

        return PodmanRunner(workspace).preflight().to_dict()
    except Exception as exc:  # diagnostics must still report the remaining system
        return {"available": False, "detail": f"{type(exc).__name__}: {exc}"}


def run_doctor(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.from_env()
    models = {
        name: {
            "relative_path": relative,
            "resolved_path": str(paths.model_root / relative),
            "available": (paths.model_root / relative / "config.json").is_file(),
        }
        for name, relative in KNOWN_MODELS.items()
    }
    return {
        "system": system_snapshot(),
        "paths": {
            "data_root": str(paths.data_root),
            "data_root_exists": paths.data_root.is_dir(),
            "data_root_writable": os.access(paths.data_root, os.W_OK),
            "model_root": str(paths.model_root),
            "model_root_exists": paths.model_root.is_dir(),
        },
        "disk": {
            "data": _disk(paths.data_root if paths.data_root.exists() else paths.data_root.parent),
            "home": _disk(Path.home()),
        },
        "tools": {name: shutil.which(name) for name in ("git", "bwrap", "podman", "docker")},
        "isolation": {
            "bwrap": _bwrap_probe(),
            "podman": _podman_probe(paths.tmp / "doctor-podman-preflight"),
        },
        "packages": _package_versions(),
        "models": models,
    }


def write_doctor_report(output: str | Path, paths: LabPaths | None = None) -> Path:
    value = run_doctor(paths)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
