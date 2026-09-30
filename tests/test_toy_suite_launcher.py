from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

from fmlab.artifacts import ExperimentResult


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_toy_suite.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("fmlab_test_toy_suite", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_runs_all_offline_tracks_and_writes_summary(tmp_path, monkeypatch):
    launcher = _load_launcher()
    calls: list[str] = []

    def completed(name: str, output_dir: str | Path) -> ExperimentResult:
        now = time.time()
        result = ExperimentResult(name, "completed", now, now, metrics={"smoke": True})
        result.write(Path(output_dir))
        calls.append(name)
        return result

    monkeypatch.setattr(
        launcher,
        "run_offline_llm_smoke_suite",
        lambda output, **kwargs: completed("llm_offline", output),
    )
    monkeypatch.setattr(
        launcher,
        "run_tiny_from_scratch",
        lambda output, config: completed("llm_tiny", output),
    )
    monkeypatch.setattr(
        launcher,
        "train_and_compare_tokenizers",
        lambda output: completed("llm_tokenizer", output),
    )
    monkeypatch.setattr(
        launcher,
        "run_offline_demo",
        lambda output, **kwargs: completed("vlm", output),
    )
    monkeypatch.setattr(
        launcher,
        "run_experiment",
        lambda name, config, output: completed(f"ml_{name}", output),
    )

    def fake_dashboard(results_root: Path, output: Path) -> Path:
        output.write_text("<html>dashboard</html>", encoding="utf-8")
        return output

    monkeypatch.setattr(launcher, "build_dashboard", fake_dashboard)
    monkeypatch.setattr(launcher, "system_snapshot", lambda: {"python": "test"})
    monkeypatch.setenv("FMLAB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("FMLAB_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    output_root = tmp_path / "artifacts"
    exit_code = launcher.main(
        [
            "--output-root",
            str(output_root),
            "--config-dir",
            str(tmp_path / "missing-configs"),
            "--device",
            "cpu",
            "--llm-steps",
            "1",
            "--tiny-steps",
            "1",
            "--vlm-samples",
            "1",
        ]
    )

    assert exit_code == 0
    assert calls[:4] == ["llm_offline", "llm_tiny", "llm_tokenizer", "vlm"]
    assert calls[4:] == [f"ml_{name}" for name in launcher.EXPERIMENTS]
    assert not any("agent" in name for name in calls)
    summary = json.loads((output_root / "toy_suite_summary.json").read_text(encoding="utf-8"))
    assert summary["offline"] is True
    assert summary["status"] == "completed"
    assert summary["experiments_total"] == 4 + len(launcher.EXPERIMENTS)
    assert summary["experiments_failed"] == 0
    assert (output_root / "dashboard.html").exists()
