from __future__ import annotations

import json
from pathlib import Path

from fmlab.reporting import build_dashboard


def test_dashboard_links_declared_html_reports(tmp_path: Path) -> None:
    run = tmp_path / "artifacts" / "track" / "run"
    run.mkdir(parents=True)
    report = run / "report.html"
    report.write_text("<html>detail</html>", encoding="utf-8")
    (run / "result.json").write_text(
        json.dumps(
            {
                "experiment": "linked-report",
                "status": "completed",
                "metrics": {"score": 1.0},
                "artifacts": [str(report)],
            }
        ),
        encoding="utf-8",
    )

    dashboard = build_dashboard(tmp_path / "artifacts", tmp_path / "dashboard.html")
    rendered = dashboard.read_text(encoding="utf-8")

    assert "Open report.html" in rendered
    assert "artifacts/track/run/report.html" in rendered
