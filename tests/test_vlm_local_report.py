from __future__ import annotations

import json
from pathlib import Path

from fmlab.vlm.local_report import render_local_inference_report


def test_render_local_inference_report_registers_image_and_html(tmp_path: Path) -> None:
    image = tmp_path / "invoice.png"
    image.write_bytes(b"png")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "metrics": {
                    "expected": "297.00",
                    "prediction": "297.00",
                    "exact_match": True,
                    "generation_latency_seconds": 1.2,
                },
                "parameters": {"image": str(image), "model": "/models/vlm"},
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    report = render_local_inference_report(result)
    rendered = report.read_text(encoding="utf-8")
    updated = json.loads(result.read_text(encoding="utf-8"))

    assert "297.00" in rendered and "invoice.png" in rendered
    assert updated["artifacts"] == [str(image), str(report)]
