from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from fmlab.artifacts import ExperimentResult
from fmlab.config import load_config

from .anomaly import run_anomaly
from .calibration import run_calibration
from .contrastive import run_contrastive
from .diffusion import run_diffusion
from .graph import run_graph
from .timeseries import run_timeseries


ExperimentFn = Callable[[dict[str, Any] | None, str | Path], ExperimentResult]

EXPERIMENTS: dict[str, ExperimentFn] = {
    "diffusion": run_diffusion,
    "contrastive": run_contrastive,
    "anomaly": run_anomaly,
    "timeseries": run_timeseries,
    "graph": run_graph,
    "calibration": run_calibration,
}


def run_experiment(
    name: str, config: dict[str, Any] | None, output_dir: str | Path
) -> ExperimentResult:
    try:
        function = EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ML experiment {name!r}; choose from {sorted(EXPERIMENTS)}"
        ) from exc
    return function(config, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the visual, offline ML toy experiments")
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--config", type=Path, help="YAML config (for one experiment only)")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/ml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.experiment != "all":
        config = load_config(args.config) if args.config else None
        result = run_experiment(args.experiment, config, args.output_dir)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.config:
        parser.error("--config cannot be used with 'all'; place configs in --config-dir")
    summary: dict[str, Any] = {}
    for name in EXPERIMENTS:
        config_path = args.config_dir / f"{name}.yaml"
        config = load_config(config_path) if config_path.exists() else None
        result = run_experiment(name, config, args.output_dir / name)
        summary[name] = result.to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
