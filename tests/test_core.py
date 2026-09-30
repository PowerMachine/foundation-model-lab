from __future__ import annotations

import json
from pathlib import Path

from fmlab.config import deep_merge, load_config
from fmlab.paths import LabPaths
from fmlab.reporting import build_dashboard, discover_results


def test_paths_and_model_escape(tmp_path: Path) -> None:
    paths = LabPaths(tmp_path / "data", tmp_path / "models")
    paths.ensure()
    assert paths.artifacts.is_dir()
    try:
        paths.model("../../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("model path traversal must fail")


def test_deep_merge() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_load_config_expands_portable_lab_paths(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "external data"
    model_root = tmp_path / "models"
    monkeypatch.setenv("FMLAB_DATA_ROOT", str(data_root))
    monkeypatch.setenv("FMLAB_MODEL_ROOT", str(model_root))
    config = tmp_path / "portable.yaml"
    config.write_text(
        "output: ${FMLAB_DATA_ROOT}/artifacts/run\n"
        "model: $FMLAB_MODEL_ROOT/Qwen/model\n"
        "nested:\n  - ${FMLAB_DATA_ROOT}/datasets/input.jsonl\n",
        encoding="utf-8",
    )

    loaded = load_config(config)

    assert loaded["output"] == str(data_root / "artifacts" / "run")
    assert loaded["model"] == str(model_root / "Qwen" / "model")
    assert loaded["nested"] == [str(data_root / "datasets" / "input.jsonl")]


def test_load_config_rejects_unresolved_environment_variable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FMLAB_REQUIRED_BUT_UNSET", raising=False)
    config = tmp_path / "bad.yaml"
    config.write_text("path: ${FMLAB_REQUIRED_BUT_UNSET}/x\n", encoding="utf-8")

    try:
        load_config(config)
    except ValueError as error:
        assert "Unresolved environment" in str(error)
    else:
        raise AssertionError("unresolved variables must fail closed")


def test_committed_yaml_configs_are_portable_and_parseable() -> None:
    repository = Path(__file__).resolve().parents[1]
    configs = sorted((repository / "configs").rglob("*.yaml"))
    assert configs
    for config in configs:
        text = config.read_text(encoding="utf-8")
        assert "/home/" not in text, config
        assert "/data/" not in text, config
        loaded = load_config(config)
        assert isinstance(loaded, dict), config


def test_dashboard_discovers_metrics_and_images(tmp_path: Path) -> None:
    run = tmp_path / "artifacts" / "toy" / "run"
    run.mkdir(parents=True)
    (run / "result.json").write_text(
        json.dumps({"experiment": "toy", "status": "completed", "metrics": {"loss": 0.2}})
    )
    (run / "curve.png").write_bytes(b"not-a-real-png")
    documents = discover_results(tmp_path / "artifacts")
    assert len(documents) == 1
    output = build_dashboard(tmp_path / "artifacts", tmp_path / "dashboard.html")
    html = output.read_text()
    assert "toy" in html and "curve.png" in html and "loss" in html
