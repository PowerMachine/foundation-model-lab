from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def write_score_bars(
    path: Path,
    scores: dict[str, float],
    *,
    title: str,
    maximum: float = 1.0,
) -> Path:
    """Write a browser-viewable SVG without requiring plotting libraries."""

    width = 760
    row_height = 62
    height = 95 + len(scores) * row_height
    items: list[str] = []
    for index, (name, value) in enumerate(scores.items()):
        y = 72 + index * row_height
        ratio = max(0.0, min(1.0, value / maximum if maximum else 0.0))
        bar_width = 480 * ratio
        items.append(
            f'<text x="20" y="{y + 20}" font-size="16">{html.escape(name)}</text>'
            f'<rect x="210" y="{y}" width="480" height="28" rx="4" fill="#e7edf4"/>'
            f'<rect x="210" y="{y}" width="{bar_width:.2f}" height="28" rx="4" fill="#3575b8"/>'
            f'<text x="700" y="{y + 20}" font-size="15">{value:.3f}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="20" y="36" font-size="24" font-weight="bold">{html.escape(title)}</text>'
        + "".join(items)
        + "</svg>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path


def write_evaluation_html(
    path: Path,
    *,
    title: str,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    image_key: str = "image_path",
    score_chart: Path | None = None,
    notice: str | None = None,
) -> Path:
    columns = list(rows[0]) if rows else []
    table_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if column == image_key and value:
                target = _relative_href(path.parent, Path(str(value)))
                cell = f'<a href="{html.escape(target)}"><img src="{html.escape(target)}" alt="sample"></a>'
            else:
                rendered = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else str(value)
                )
                cell = html.escape(rendered)
            cells.append(f"<td>{cell}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")
    summary_cards = "".join(
        f'<div class="card"><span>{html.escape(str(key))}</span><strong>{html.escape(_pretty(value))}</strong></div>'
        for key, value in summary.items()
    )
    chart_html = ""
    if score_chart is not None:
        chart_html = f'<img class="chart" src="{html.escape(_relative_href(path.parent, score_chart))}" alt="scores">'
    notice_html = f'<p class="notice">{html.escape(notice)}</p>' if notice else ""
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#152333;background:#f7f9fb}}
h1{{margin-bottom:.4rem}} .summary{{display:flex;flex-wrap:wrap;gap:.8rem;margin:1rem 0}}
.card{{background:white;border:1px solid #dce3eb;border-radius:8px;padding:.8rem 1rem;min-width:130px}}
.card span{{display:block;color:#64748b;font-size:.8rem}} .card strong{{font-size:1.25rem}}
.notice{{background:#fff4cc;border-left:4px solid #e0a800;padding:.8rem}}
.chart{{max-width:760px;width:100%;background:white;border:1px solid #dce3eb}}
table{{border-collapse:collapse;background:white;font-size:.86rem;width:100%}}
th,td{{border:1px solid #dce3eb;padding:.45rem;text-align:left;vertical-align:top}}
th{{position:sticky;top:0;background:#eaf1f7}} td img{{width:110px;max-height:90px;object-fit:contain}}
</style></head><body><h1>{html.escape(title)}</h1>{notice_html}
<div class="summary">{summary_cards}</div>{chart_html}
<h2>Per-example evidence</h2><div style="overflow:auto"><table><thead><tr>
{"".join(f"<th>{html.escape(column)}</th>" for column in columns)}</tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></div></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


def write_gallery(path: Path, *, title: str, images: Iterable[Path]) -> Path:
    cards = []
    for image_path in images:
        href = _relative_href(path.parent, image_path)
        cards.append(
            f'<figure><a href="{html.escape(href)}"><img src="{html.escape(href)}"></a>'
            f"<figcaption>{html.escape(image_path.name)}</figcaption></figure>"
        )
    content = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui;margin:2rem}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}}
figure{{margin:0;padding:1rem;border:1px solid #ddd}}img{{width:100%;height:260px;object-fit:contain}}</style>
</head><body><h1>{html.escape(title)}</h1><main>{"".join(cards)}</main></body></html>"""
    path.write_text(content, encoding="utf-8")
    return path


def _pretty(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _relative_href(base: Path, target: Path) -> str:
    import os

    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()
