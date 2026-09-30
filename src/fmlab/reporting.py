from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResultDocument:
    path: Path
    value: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.value.get("experiment") or self.path.parent.name)

    @property
    def status(self) -> str:
        return str(self.value.get("status", "recorded"))


def discover_results(root: str | Path) -> list[ResultDocument]:
    base = Path(root)
    found: list[ResultDocument] = []
    if not base.exists():
        return found
    candidates = set(base.rglob("result.json"))
    candidates.update(base.rglob("metrics.json"))
    for path in sorted(candidates):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            found.append(ResultDocument(path, value))
    return found


def _flatten(value: Any, prefix: str = "", limit: int = 30) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(item, next_prefix, limit))
            if len(rows) >= limit:
                break
    elif isinstance(value, (list, tuple)):
        rows.append((prefix, json.dumps(value, ensure_ascii=False)[:300]))
    else:
        rows.append((prefix, str(value)))
    return rows[:limit]


def _artifact_links(document: ResultDocument, output_parent: Path) -> str:
    image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".gif"}
    images = [
        path
        for path in document.path.parent.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ]
    explicit = document.value.get("artifacts", [])
    if isinstance(explicit, list):
        for raw in explicit:
            path = Path(str(raw))
            if not path.is_absolute():
                path = document.path.parent / path
            if path.is_file() and path.suffix.lower() in image_extensions and path not in images:
                images.append(path)
    cards = []
    for path in images[:12]:
        relative = os.path.relpath(path, output_parent)
        escaped = html.escape(relative)
        cards.append(
            f"<a class='image' href='{escaped}'><img loading='lazy' src='{escaped}' "
            f"alt='{html.escape(path.name)}'><span>{html.escape(path.name)}</span></a>"
        )
    return "".join(cards)


def _report_links(document: ResultDocument, output_parent: Path) -> str:
    reports: list[Path] = []
    for name in ("report.html", "index.html", "trace.html", "gallery.html"):
        path = document.path.parent / name
        if path.is_file():
            reports.append(path)
    explicit = document.value.get("artifacts", [])
    if isinstance(explicit, list):
        for raw in explicit:
            path = Path(str(raw))
            if not path.is_absolute():
                path = document.path.parent / path
            if path.is_file() and path.suffix.lower() == ".html" and path not in reports:
                reports.append(path)
    links = []
    for path in reports:
        relative = os.path.relpath(path, output_parent)
        escaped = html.escape(relative)
        links.append(f"<a class='report-link' href='{escaped}'>Open {html.escape(path.name)}</a>")
    return "".join(links)


def build_dashboard(results_root: str | Path, output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    documents = discover_results(results_root)
    sections: list[str] = []
    for document in documents:
        rows = _flatten(document.value.get("metrics", document.value))
        table = "".join(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
            for key, value in rows
        )
        source = os.path.relpath(document.path, destination.parent)
        sections.append(
            f"<article data-status='{html.escape(document.status)}'>"
            f"<header><h2>{html.escape(document.name)}</h2>"
            f"<span class='status'>{html.escape(document.status)}</span></header>"
            f"<p><a href='{html.escape(source)}'>{html.escape(str(document.path))}</a></p>"
            f"<nav class='reports'>{_report_links(document, destination.parent)}</nav>"
            f"<table>{table}</table><div class='gallery'>"
            f"{_artifact_links(document, destination.parent)}</div></article>"
        )
    empty = "<p class='empty'>아직 실행 결과가 없습니다. smoke suite를 먼저 실행하세요.</p>"
    document_html = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Foundation Model Lab</title><style>
:root{{--bg:#0b1020;--card:#151d33;--line:#2c385b;--text:#e8edf7;--muted:#9ba8c6;--accent:#63d4ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px system-ui}}
main{{max-width:1400px;margin:auto;padding:2rem}} h1{{font-size:2rem}} .lead{{color:var(--muted)}}
.summary{{display:flex;gap:1rem;margin:1.5rem 0}} .pill,.status{{padding:.35rem .7rem;border:1px solid var(--line);border-radius:999px}}
article{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.2rem;margin:1rem 0}}
header{{display:flex;justify-content:space-between;align-items:center;gap:1rem}} h2{{margin:.2rem 0}}
a{{color:var(--accent)}} table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{text-align:left;border-bottom:1px solid var(--line);padding:.5rem}} th{{width:35%;color:var(--muted)}}
.reports{{display:flex;flex-wrap:wrap;gap:.6rem}} .report-link{{padding:.45rem .7rem;border:1px solid var(--line);border-radius:8px;text-decoration:none}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1rem}} .image{{display:flex;flex-direction:column;gap:.4rem}}
.image img{{width:100%;height:190px;object-fit:contain;background:#080c17;border-radius:8px}} .empty{{padding:3rem;text-align:center;color:var(--muted)}}
</style></head><body><main><h1>Foundation Model Lab</h1>
<p class='lead'>작은 실험의 학습 과정, 정량 결과, 시각 자료를 한곳에서 비교합니다.</p>
<div class='summary'><span class='pill'>실험 결과 {len(documents)}개</span><span class='pill'>root: {html.escape(str(results_root))}</span></div>
{"".join(sections) if sections else empty}</main></body></html>"""
    destination.write_text(document_html, encoding="utf-8")
    return destination
