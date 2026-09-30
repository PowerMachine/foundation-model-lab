from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _decimal(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def write_sharding_svg(path: Path, sharding: Mapping[str, Any], dataset_size: int) -> Path:
    cell_width = 35
    row_height = 46
    margin_left = 92
    margin_top = 42
    epochs = list(sharding["epochs"])
    rows = len(epochs) * 2
    width = margin_left + dataset_size * cell_width + 28
    height = margin_top + rows * row_height + 45
    colors = ("#2563eb", "#db2777")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        '<text x="20" y="26" fill="#f8fafc" font-size="17" font-family="sans-serif">'
        "DistributedSampler assignment (cell text = sample id)</text>",
    ]
    row_index = 0
    for epoch in epochs:
        for rank_text, sample_ids in sorted(epoch["rank_indices"].items()):
            y = margin_top + row_index * row_height
            elements.append(
                f'<text x="18" y="{y + 25}" fill="#cbd5e1" font-size="13" '
                f'font-family="sans-serif">e{epoch["epoch"]} / r{rank_text}</text>'
            )
            rank = int(rank_text)
            for position, sample_id in enumerate(sample_ids):
                x = margin_left + position * cell_width
                elements.extend(
                    [
                        f'<rect x="{x}" y="{y + 5}" width="29" height="29" rx="5" '
                        f'fill="{colors[rank]}" opacity="0.88"/>',
                        f'<text x="{x + 14.5}" y="{y + 24}" text-anchor="middle" '
                        f'fill="white" font-size="11" font-family="monospace">{sample_id}</text>',
                    ]
                )
            row_index += 1
    footer = (
        f"coverage={sharding['all_epochs_complete']}; "
        f"set_epoch_changes_order={sharding['epoch_permutation_changed']}"
    )
    elements.append(
        f'<text x="18" y="{height - 16}" fill="#94a3b8" font-size="12" '
        f'font-family="monospace">{footer}</text>'
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")
    return path


def write_resume_svg(
    path: Path,
    uninterrupted: Sequence[Mapping[str, Any]],
    resumed: Sequence[Mapping[str, Any]],
) -> Path:
    width, height = 700, 330
    left, top, plot_width, plot_height = 76, 52, 570, 210
    all_values = [float(row["loss"]) for row in (*uninterrupted, *resumed)]
    low = min(all_values)
    high = max(all_values)
    span = max(high - low, 1e-12)

    def points(rows: Sequence[Mapping[str, Any]]) -> str:
        maximum_step = max(len(uninterrupted) - 1, 1)
        values: list[str] = []
        for row in rows:
            x = left + float(row["optimizer_step"]) / maximum_step * plot_width
            y = top + (high - float(row["loss"])) / span * plot_height
            values.append(f"{x:.2f},{y:.2f}")
        return " ".join(values)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#0b1220"/>
<text x="24" y="28" fill="#f8fafc" font-size="17" font-family="sans-serif">Uninterrupted vs checkpoint-resumed loss</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#64748b"/>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#64748b"/>
<polyline points="{points(uninterrupted)}" fill="none" stroke="#38bdf8" stroke-width="4"/>
<polyline points="{points(resumed)}" fill="none" stroke="#f472b6" stroke-width="2" stroke-dasharray="7 5"/>
<text x="{left}" y="{top + plot_height + 27}" fill="#94a3b8" font-size="12" font-family="sans-serif">optimizer step</text>
<text x="{left + 230}" y="{height - 20}" fill="#38bdf8" font-size="13" font-family="sans-serif">uninterrupted</text>
<text x="{left + 360}" y="{height - 20}" fill="#f472b6" font-size="13" font-family="sans-serif">resumed</text>
<text x="14" y="{top + 8}" fill="#94a3b8" font-size="11" font-family="monospace">{high:.6f}</text>
<text x="14" y="{top + plot_height}" fill="#94a3b8" font-size="11" font-family="monospace">{low:.6f}</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return path


def _metric_rows(values: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, value in values.items():
        label = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            rows.extend(_metric_rows(value, label))
        elif not isinstance(value, (list, tuple)):
            rows.append((label, value))
    return rows


def write_html_report(path: Path, result: Mapping[str, Any]) -> Path:
    metrics = result["metrics"]
    parity = metrics["parity"]
    resume = metrics["resume"]
    checkpoint = metrics["checkpoint"]
    cards = [
        ("Gradient max abs", parity["gradient"]["max_absolute_error"]),
        ("Final-weight max abs", parity["final_weight"]["max_absolute_error"]),
        ("Resume max abs", resume["state_error"]["max_absolute_error"]),
        ("Effective global batch", metrics["training"]["effective_global_batch"]),
        ("Coverage", metrics["sharding"]["all_epochs_complete"]),
        ("Corrupt rejected", checkpoint["fault_audit"]["corrupt_checkpoint_rejected"]),
    ]
    card_html = "".join(
        '<div class="card"><div class="label">'
        + html.escape(label)
        + '</div><div class="value">'
        + html.escape(_decimal(value))
        + "</div></div>"
        for label, value in cards
    )
    rows = _metric_rows(
        {
            "parity": {
                "gradient_abs": parity["gradient"]["max_absolute_error"],
                "gradient_rel": parity["gradient"]["max_relative_error"],
                "first_update_abs": parity["first_update"]["max_absolute_error"],
                "final_weight_abs": parity["final_weight"]["max_absolute_error"],
            },
            "resume": {
                "state_abs": resume["state_error"]["max_absolute_error"],
                "state_rel": resume["state_error"]["max_relative_error"],
                "loss_abs": resume["max_loss_absolute_error"],
            },
            "training": metrics["training"],
            "checkpoint": {
                "bytes": checkpoint["write"]["bytes"],
                "save_seconds": checkpoint["write"]["save_seconds"],
                "load_seconds": checkpoint["load"]["load_seconds"],
            },
        }
    )
    table = "".join(
        f"<tr><td>{html.escape(label)}</td><td><code>{html.escape(_decimal(value, 6))}</code></td></tr>"
        for label, value in rows
    )
    boundary = result["claim_boundary"]
    excluded = "".join(f"<li>{html.escape(item)}</li>" for item in boundary["not_evidenced"])
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CPU DDP correctness report</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;background:#08111f;color:#e2e8f0;margin:0;padding:28px;line-height:1.5}}
main{{max-width:1120px;margin:auto}} h1,h2{{color:#f8fafc}} .sub{{color:#94a3b8}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}}
.card{{background:#111c2e;border:1px solid #26364f;border-radius:12px;padding:15px}}
.label{{font-size:12px;color:#94a3b8}} .value{{font-size:23px;font-weight:700;margin-top:5px}}
.panel{{background:#111c2e;border:1px solid #26364f;border-radius:12px;padding:18px;margin:16px 0;overflow:auto}}
table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #26364f;padding:8px;text-align:left}}
img{{max-width:100%;height:auto;border-radius:8px}} code{{color:#7dd3fc}}
.pass{{color:#86efac}} .warning{{color:#fbbf24}}
</style></head><body><main>
<h1>CPU DDP Correctness &amp; Exact-Resume Lab</h1>
<p class="sub">Actual {result["parameters"]["world_size"]}-process CPU/Gloo execution · status <strong class="pass">{html.escape(str(result["status"]))}</strong> · claim level <code>{html.escape(str(result["claim_level"]))}</code></p>
<div class="cards">{card_html}</div>
<div class="panel"><h2>Deterministic sharding</h2><img src="sample_sharding.svg" alt="rank sample assignment"></div>
<div class="panel"><h2>Resume equivalence</h2><img src="resume_equivalence.svg" alt="uninterrupted and resumed loss"></div>
<div class="panel"><h2>Measured and estimated metrics</h2><table>{table}</table></div>
<div class="panel"><h2>Claim boundary</h2><p>{html.escape(boundary["evidenced"])}</p><p class="warning">This run does not evidence:</p><ul>{excluded}</ul></div>
<div class="panel"><h2>Reproduction</h2><pre><code>{html.escape(result["reproduction_command"])}</code></pre></div>
<div class="panel"><h2>Machine-readable contract</h2><pre><code>{html.escape(json.dumps(result["contracts"], indent=2))}</code></pre></div>
</main></body></html>"""
    path.write_text(report, encoding="utf-8")
    return path
