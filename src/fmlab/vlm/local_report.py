from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any


def render_local_inference_report(result_json: str | Path) -> Path:
    """Render one real local VLM smoke result and register its visual artifacts."""

    source = Path(result_json)
    value: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    metrics = value.get("metrics", {})
    parameters = value.get("parameters", {})
    image_path = Path(str(parameters.get("image", "")))
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    destination = source.with_name("report.html")
    relative_image = os.path.relpath(image_path, destination.parent)
    expected = html.escape(str(metrics.get("expected", "")))
    prediction = html.escape(str(metrics.get("prediction", "")))
    latency = html.escape(str(metrics.get("generation_latency_seconds", "")))
    exact = bool(metrics.get("exact_match", False))
    status_class = "pass" if exact else "fail"
    document = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Real local VLM inference</title><style>
body{{margin:0;background:#0b1020;color:#e8edf7;font:16px system-ui}}main{{max-width:1100px;margin:auto;padding:2rem}}
.grid{{display:grid;grid-template-columns:minmax(320px,1fr) minmax(280px,.7fr);gap:1.5rem}}img{{width:100%;max-height:720px;object-fit:contain;background:white;border-radius:12px}}
.card{{background:#151d33;border:1px solid #2c385b;border-radius:12px;padding:1.2rem}}dt{{color:#9ba8c6}}dd{{font-size:1.3rem;margin:0 0 1rem}}.pass{{color:#71e3a1}}.fail{{color:#ff8f9c}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Qwen3-VL 실제 로컬 추론</h1><div class='grid'>
<a href='{html.escape(relative_image)}'><img src='{html.escape(relative_image)}' alt='input document'></a>
<section class='card'><dl><dt>Expected</dt><dd>{expected}</dd><dt>Prediction</dt><dd>{prediction}</dd>
<dt>Exact match</dt><dd class='{status_class}'>{str(exact).lower()}</dd>
<dt>첫 load 포함 latency</dt><dd>{latency} s</dd><dt>Model</dt><dd>{html.escape(str(parameters.get("model", "")))}</dd></dl></section>
</div></main></body></html>"""
    destination.write_text(document, encoding="utf-8")

    artifacts = list(value.get("artifacts") or [])
    for artifact in (str(image_path), str(destination)):
        if artifact not in artifacts:
            artifacts.append(artifact)
    value["artifacts"] = artifacts
    source.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
