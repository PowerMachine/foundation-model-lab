"""Dependency-light artifacts for the Inference Dynamics Lab."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

from .simulator import SimulationOutput


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706")


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _svg_document(*, title: str, width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<text x="32" y="38" font-family="system-ui" font-size="22" '
        f'font-weight="700">{html.escape(title)}</text>'
        f"{body}</svg>"
    )


def write_pareto_svg(path: Path, outputs: list[SimulationOutput]) -> Path:
    width, height = 820, 500
    left, right, top, bottom = 90, 35, 75, 75
    points = []
    for output in outputs:
        p99 = output.metrics["latency_ms"]["e2e"]["p99"]
        if p99 is None:
            continue
        points.append(
            (
                output.policy["name"],
                float(p99),
                float(output.metrics["slo_goodput_tokens_per_second"]),
            )
        )
    maximum_x = max((point[1] for point in points), default=1.0) * 1.12
    maximum_y = max((point[2] for point in points), default=1.0) * 1.18 or 1.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(value: float) -> float:
        return left + value / maximum_x * plot_width

    def sy(value: float) -> float:
        return top + plot_height - value / maximum_y * plot_height

    pieces = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#64748b"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#64748b"/>',
        f'<text x="{width / 2 - 85}" y="{height - 22}" font-family="system-ui" '
        'font-size="15">E2E p99 latency (ms, lower is better)</text>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" '
        'font-family="system-ui" font-size="15">SLO goodput (output tokens/s)</text>',
    ]
    for tick in range(6):
        value_x = maximum_x * tick / 5
        x = sx(value_x)
        pieces.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            'stroke="#e2e8f0"/>'
            f'<text x="{x - 12:.2f}" y="{top + plot_height + 24}" font-family="system-ui" '
            f'font-size="12">{value_x:.0f}</text>'
        )
        value_y = maximum_y * tick / 5
        y = sy(value_y)
        pieces.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#e2e8f0"/>'
            f'<text x="{left - 55}" y="{y + 4:.2f}" font-family="system-ui" '
            f'font-size="12">{value_y:.1f}</text>'
        )
    for index, (name, p99, goodput) in enumerate(points):
        color = COLORS[index % len(COLORS)]
        x, y = sx(p99), sy(goodput)
        pieces.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="{color}"/>'
            f'<text x="{x + 12:.2f}" y="{y - 8:.2f}" font-family="system-ui" '
            f'font-size="13" fill="{color}">{html.escape(name)}</text>'
            f'<text x="{x + 12:.2f}" y="{y + 9:.2f}" font-family="system-ui" '
            f'font-size="11">p99={p99:.1f}, goodput={goodput:.1f}</text>'
        )
    path.write_text(
        _svg_document(
            title="Latency / SLO-goodput policy frontier",
            width=width,
            height=height,
            body="".join(pieces),
        ),
        encoding="utf-8",
    )
    return path


def write_slo_svg(path: Path, outputs: list[SimulationOutput]) -> Path:
    width = 860
    row_height = 92
    height = 105 + row_height * len(outputs)
    statuses = ("slo_met", "late", "rejected", "timed_out", "cancelled", "oom", "aborted")
    colors = {
        "slo_met": "#059669",
        "late": "#f59e0b",
        "rejected": "#dc2626",
        "timed_out": "#9333ea",
        "cancelled": "#64748b",
        "oom": "#be123c",
        "aborted": "#111827",
    }
    pieces: list[str] = []
    for index, output in enumerate(outputs):
        metrics = output.metrics
        counts = metrics["status_counts"]
        completed = int(counts["completed"])
        slo_met = round(float(metrics["slo_attainment"]) * completed)
        values = {
            "slo_met": slo_met,
            "late": completed - slo_met,
            "rejected": int(counts["rejected"]),
            "timed_out": int(counts["timed_out"]),
            "cancelled": int(counts["cancelled"]),
            "oom": int(counts["oom"]),
            "aborted": int(counts["aborted"]),
        }
        total = max(1, int(metrics["request_count"]))
        y = 78 + index * row_height
        pieces.append(
            f'<text x="24" y="{y + 20}" font-family="system-ui" font-size="14" '
            f'font-weight="700">{html.escape(output.policy["name"])}</text>'
        )
        cursor = 180.0
        for status in statuses:
            bar = 610.0 * values[status] / total
            pieces.append(
                f'<rect x="{cursor:.2f}" y="{y}" width="{bar:.2f}" height="28" '
                f'fill="{colors[status]}"><title>{status}: {values[status]}</title></rect>'
            )
            cursor += bar
        pieces.append(
            f'<text x="180" y="{y + 50}" font-family="system-ui" font-size="12">'
            f"SLO {metrics['slo_e2e_ms']:.0f} ms · attainment {metrics['slo_attainment']:.1%} · "
            f"goodput {metrics['slo_goodput_requests_per_second']:.2f} req/s</text>"
        )
    legend = []
    cursor = 25
    legend_y = height - 24
    for status in statuses:
        legend.append(
            f'<rect x="{cursor}" y="{legend_y - 12}" width="12" height="12" '
            f'fill="{colors[status]}"/><text x="{cursor + 17}" y="{legend_y}" '
            f'font-family="system-ui" font-size="11">{status}</text>'
        )
        cursor += 105
    path.write_text(
        _svg_document(
            title="SLO outcome and backpressure accounting",
            width=width,
            height=height,
            body="".join(pieces + legend),
        ),
        encoding="utf-8",
    )
    return path


def write_kv_svg(path: Path, outputs: list[SimulationOutput]) -> Path:
    width, height = 860, 480
    left, right, top, bottom = 80, 35, 70, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum_time = max(
        (float(sample["time_ms"]) for output in outputs for sample in output.kv_samples),
        default=1.0,
    )
    maximum_time = max(maximum_time, 1.0)
    pieces = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#64748b"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#64748b"/>',
        f'<text x="{width / 2 - 50}" y="{height - 18}" font-family="system-ui" '
        'font-size="14">simulation time (ms)</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" '
        'font-family="system-ui" font-size="14">KV block utilization</text>',
    ]
    for tick in range(6):
        ratio = tick / 5
        y = top + plot_height - ratio * plot_height
        pieces.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#e2e8f0"/>'
            f'<text x="{left - 48}" y="{y + 4:.2f}" font-family="system-ui" '
            f'font-size="12">{ratio:.0%}</text>'
        )
    for index, output in enumerate(outputs):
        color = COLORS[index % len(COLORS)]
        samples = list(output.kv_samples)
        if len(samples) > 300:
            stride = max(1, len(samples) // 300)
            samples = samples[::stride]
            if samples[-1] != output.kv_samples[-1]:
                samples.append(output.kv_samples[-1])
        coordinates = " ".join(
            f"{left + float(sample['time_ms']) / maximum_time * plot_width:.2f},"
            f"{top + plot_height - float(sample['utilization']) * plot_height:.2f}"
            for sample in samples
        )
        pieces.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            f'stroke-width="2"><title>{html.escape(output.policy["name"])}</title></polyline>'
            f'<rect x="{width - 220}" y="{58 + index * 22}" width="12" height="12" '
            f'fill="{color}"/><text x="{width - 202}" y="{69 + index * 22}" '
            f'font-family="system-ui" font-size="12">{html.escape(output.policy["name"])}</text>'
        )
    path.write_text(
        _svg_document(
            title="Paged-KV utilization over simulated time",
            width=width,
            height=height,
            body="".join(pieces),
        ),
        encoding="utf-8",
    )
    return path


def write_capacity_svg(path: Path, capacity_sweep: dict[str, Any]) -> Path:
    """Plot modeled offered/completed load and SLO attainment with seed ranges."""

    width, height = 1120, 570
    if not capacity_sweep.get("enabled") or not capacity_sweep.get("aggregates"):
        body = (
            '<text x="42" y="100" font-family="system-ui" font-size="16" '
            'fill="#64748b">Capacity sweep disabled; no modeled capacity claim produced.</text>'
        )
        path.write_text(
            _svg_document(title="Modeled capacity sweep", width=width, height=height, body=body),
            encoding="utf-8",
        )
        return path

    aggregates = capacity_sweep["aggregates"]
    policy_names = list(dict.fromkeys(row["policy"] for row in aggregates))
    panels = (
        ("completed_load_requests_per_second", "completed load (requests/s)"),
        ("slo_attainment", "SLO attainment among completions"),
    )
    plot_top, plot_height, plot_width = 92.0, 350.0, 440.0
    panel_lefts = (82.0, 640.0)
    maximum_x = max(
        float(row["metrics"]["offered_load_requests_per_second"]["max"] or 0.0)
        for row in aggregates
    )
    maximum_x = max(maximum_x * 1.08, 1.0)
    maximum_completed = max(
        float(row["metrics"]["completed_load_requests_per_second"]["max"] or 0.0)
        for row in aggregates
    )
    maximum_completed = max(maximum_completed * 1.12, 1.0)
    pieces: list[str] = []

    def sx(value: float, left: float) -> float:
        return left + value / maximum_x * plot_width

    for panel_index, (metric_name, label) in enumerate(panels):
        left = panel_lefts[panel_index]
        maximum_y = maximum_completed if metric_name.startswith("completed") else 1.0

        def sy(value: float) -> float:
            return plot_top + plot_height - value / maximum_y * plot_height

        pieces.extend(
            [
                f'<line x1="{left}" y1="{plot_top}" x2="{left}" '
                f'y2="{plot_top + plot_height}" stroke="#64748b"/>',
                f'<line x1="{left}" y1="{plot_top + plot_height}" '
                f'x2="{left + plot_width}" y2="{plot_top + plot_height}" stroke="#64748b"/>',
                f'<text x="{left + 100}" y="{height - 70}" font-family="system-ui" '
                'font-size="13">realized offered load (requests/s)</text>',
                f'<text x="{left + 85}" y="70" font-family="system-ui" font-size="15" '
                f'font-weight="700">{html.escape(label)}</text>',
            ]
        )
        for tick in range(6):
            x_value = maximum_x * tick / 5
            x = sx(x_value, left)
            y_value = maximum_y * tick / 5
            y = sy(y_value)
            y_label = f"{y_value:.0%}" if maximum_y == 1.0 else f"{y_value:.0f}"
            pieces.append(
                f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" '
                f'y2="{plot_top + plot_height}" stroke="#eef2f7"/>'
                f'<text x="{x - 12:.2f}" y="{plot_top + plot_height + 22}" '
                f'font-family="system-ui" font-size="11">{x_value:.0f}</text>'
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
                f'y2="{y:.2f}" stroke="#eef2f7"/>'
                f'<text x="{left - 45}" y="{y + 4:.2f}" font-family="system-ui" '
                f'font-size="11">{y_label}</text>'
            )
        if metric_name == "slo_attainment":
            threshold = float(
                capacity_sweep["design"]["knee_rule"]["criteria"]["slo_attainment_mean_gte"]
            )
            threshold_y = sy(threshold)
            pieces.append(
                f'<line x1="{left}" y1="{threshold_y:.2f}" x2="{left + plot_width}" '
                f'y2="{threshold_y:.2f}" stroke="#d97706" stroke-dasharray="6 5"/>'
                f'<text x="{left + 5}" y="{threshold_y - 7:.2f}" font-family="system-ui" '
                f'font-size="11" fill="#b45309">threshold {threshold:.0%}</text>'
            )
        for policy_index, policy_name in enumerate(policy_names):
            color = COLORS[policy_index % len(COLORS)]
            rows = [row for row in aggregates if row["policy"] == policy_name]
            coordinates: list[str] = []
            for row in rows:
                x_summary = row["metrics"]["offered_load_requests_per_second"]
                y_summary = row["metrics"][metric_name]
                x_value = float(x_summary["mean"] or 0.0)
                y_value = float(y_summary["mean"] or 0.0)
                x, y = sx(x_value, left), sy(y_value)
                coordinates.append(f"{x:.2f},{y:.2f}")
                y_min, y_max = float(y_summary["min"] or 0.0), float(y_summary["max"] or 0.0)
                pieces.append(
                    f'<line x1="{x:.2f}" y1="{sy(y_min):.2f}" x2="{x:.2f}" '
                    f'y2="{sy(y_max):.2f}" stroke="{color}" opacity=".45"/>'
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" '
                    f'fill="{color if row["sustainable"] else "#ffffff"}" '
                    f'stroke="{color}" stroke-width="2"><title>{html.escape(policy_name)} · '
                    f"configured={row['configured_poisson_rate_rps']} rps · "
                    f"{metric_name}={y_value:.4f} · sustainable={row['sustainable']}</title></circle>"
                )
            pieces.append(
                f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{color}" '
                'stroke-width="2"/>'
            )

    legend_x = 88
    for policy_index, policy_name in enumerate(policy_names):
        color = COLORS[policy_index % len(COLORS)]
        knee = capacity_sweep["knees"][policy_name]
        knee_text = (
            f"knee {knee['last_sustainable_configured_rate_rps']}–"
            f"{knee['first_unsustainable_configured_rate_rps']} configured rps"
            if knee["classification"] == "observed_interval"
            else knee["classification"].replace("_", " ")
        )
        pieces.append(
            f'<line x1="{legend_x}" y1="{height - 28}" x2="{legend_x + 24}" '
            f'y2="{height - 28}" stroke="{color}" stroke-width="3"/>'
            f'<text x="{legend_x + 31}" y="{height - 23}" font-family="system-ui" '
            f'font-size="11">{html.escape(policy_name)} · {html.escape(knee_text)}</text>'
        )
        legend_x += 335
    path.write_text(
        _svg_document(
            title="Modeled capacity and sustainable-load knees (mean + seed range)",
            width=width,
            height=height,
            body="".join(pieces),
        ),
        encoding="utf-8",
    )
    return path


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_html_report(
    path: Path,
    *,
    outputs: list[SimulationOutput],
    probe: dict[str, Any],
    comparison: dict[str, Any],
    claim_boundary: list[str],
    capacity_sweep: dict[str, Any],
    artifacts: list[str],
) -> Path:
    policy_rows = []
    for output in outputs:
        metrics = output.metrics
        counts = metrics["status_counts"]
        policy_rows.append(
            "<tr>"
            f"<td>{html.escape(output.policy['name'])}</td>"
            f"<td>{html.escape(output.policy['kind'])}</td>"
            f"<td>{counts['completed']}/{metrics['request_count']}</td>"
            f"<td>{_format(metrics['latency_ms']['ttft']['p99'])}</td>"
            f"<td>{_format(metrics['latency_ms']['e2e']['p99'])}</td>"
            f"<td>{_format(metrics['output_token_throughput_per_second'])}</td>"
            f"<td>{_format(metrics['slo_goodput_tokens_per_second'])}</td>"
            f"<td>{counts['rejected']}/{counts['timed_out']}/{counts['oom']}</td>"
            f"<td>{metrics['kv']['peak_utilization']:.1%}</td>"
            f"<td>{metrics['theoretical_vs_achieved_efficiency']:.1%}</td>"
            "</tr>"
        )
    root_cards = []
    for output in outputs:
        causes = "".join(
            f"<li><strong>{html.escape(str(item['cause']))}</strong> "
            f"({html.escape(str(item['confidence']))})"
            f"<pre>{html.escape(json.dumps(item['evidence'], ensure_ascii=False, indent=2))}</pre></li>"
            for item in output.root_causes
        )
        root_cards.append(
            f"<section><h3>{html.escape(output.policy['name'])}</h3><ul>{causes}</ul></section>"
        )
    probe_rows = []
    if probe.get("enabled"):
        for name, value in probe["variants"].items():
            fidelity = value["fidelity_vs_fp32"]
            probe_rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{_format(fidelity.get('top1_agreement'))}</td>"
                f"<td>{_format(fidelity.get('cosine_similarity'))}</td>"
                f"<td>{_format(fidelity.get('kl_divergence_reference_to_candidate'))}</td>"
                f"<td>{_format(value['latency']['p50_ms'])}</td>"
                f"<td>{_format(value['latency_ratio_vs_fp32'])}</td>"
                f"<td>{html.escape(str(value['gate_pass']))}</td>"
                "</tr>"
            )
    capacity_rows: list[str] = []
    if capacity_sweep.get("enabled"):
        for aggregate in capacity_sweep.get("aggregates", []):
            metrics = aggregate["metrics"]

            def summary_cell(name: str, *, percent: bool = False) -> str:
                summary = metrics[name]
                if summary["mean"] is None:
                    return "n/a"
                if percent:
                    return (
                        f"{float(summary['mean']):.1%} "
                        f"[{float(summary['min']):.1%}, {float(summary['max']):.1%}]"
                    )
                return (
                    f"{float(summary['mean']):.2f} "
                    f"[{float(summary['min']):.2f}, {float(summary['max']):.2f}]"
                )

            capacity_rows.append(
                "<tr>"
                f"<td>{html.escape(aggregate['policy'])}</td>"
                f"<td>{aggregate['configured_poisson_rate_rps']}</td>"
                f"<td>{summary_cell('offered_load_requests_per_second')}</td>"
                f"<td>{summary_cell('completed_load_requests_per_second')}</td>"
                f"<td>{summary_cell('e2e_p99_ms')}</td>"
                f"<td>{summary_cell('slo_attainment', percent=True)}</td>"
                f"<td>{summary_cell('completion_ratio', percent=True)}</td>"
                f"<td>{summary_cell('slo_goodput_tokens_per_second')}</td>"
                f"<td>{summary_cell('terminal_failure_ratio', percent=True)}</td>"
                f"<td>{summary_cell('kv_peak_utilization', percent=True)}</td>"
                f"<td>{html.escape(str(aggregate['sustainable']))}</td>"
                "</tr>"
            )
        knee_json = html.escape(
            json.dumps(capacity_sweep.get("knees", {}), ensure_ascii=False, indent=2)
        )
        knee_rule = html.escape(
            json.dumps(capacity_sweep["design"]["knee_rule"], ensure_ascii=False, indent=2)
        )
        capacity_section = (
            "<section><h2>Sustainable-load capacity study (simulator)</h2>"
            f"<p>{html.escape(str(capacity_sweep['claim_boundary']))}</p>"
            "<p>Cells show mean [minimum, maximum] across fixed seeds; every policy at a "
            "rate uses the same request trace.</p>"
            "<table><thead><tr><th>policy</th><th>configured rps</th><th>offered rps</th>"
            "<th>completed rps</th><th>E2E p99 ms</th><th>SLO attainment</th>"
            "<th>completion ratio</th><th>SLO goodput tok/s</th><th>terminal failures</th>"
            "<th>KV peak</th><th>sustainable</th></tr></thead>"
            f"<tbody>{''.join(capacity_rows)}</tbody></table>"
            f"<h3>Explicit knee rule</h3><pre>{knee_rule}</pre>"
            f"<h3>Detected knees</h3><pre>{knee_json}</pre></section>"
        )
    else:
        capacity_section = (
            "<section><h2>Sustainable-load capacity study (simulator)</h2>"
            "<p>Disabled by configuration.</p></section>"
        )
    boundary = "".join(f"<li>{html.escape(item)}</li>" for item in claim_boundary)
    artifact_links = " · ".join(
        f'<a href="{html.escape(name)}">{html.escape(name)}</a>' for name in artifacts
    )
    chart_cards = "".join(
        f'<figure><a href="{name}"><img src="{name}" alt="{name}"></a>'
        f"<figcaption>{name}</figcaption></figure>"
        for name in ("capacity.svg", "pareto.svg", "slo.svg", "kv.svg")
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inference Dynamics Lab</title><style>
:root{{--ink:#172033;--muted:#5f6b7a;--line:#dbe3ec;--panel:#fff;--bg:#f4f7fa;--accent:#1456b8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px system-ui}}
main{{max-width:1280px;margin:auto;padding:2rem}} h1{{margin-bottom:.25rem}} .lead{{color:var(--muted)}}
.boundary{{background:#fff4d6;border-left:5px solid #e29b13;padding:.8rem 1.1rem}}
section,figure{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem;margin:1rem 0}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}
.charts figure{{margin:0}} img{{width:100%;height:auto}} table{{width:100%;border-collapse:collapse;background:white}}
th,td{{padding:.55rem;border:1px solid var(--line);text-align:left}} th{{background:#eaf1f8}}
pre{{white-space:pre-wrap;font-size:.78rem;background:#f7f9fb;padding:.6rem}} a{{color:var(--accent)}}
.artifacts{{line-height:2}}</style></head><body><main><h1>Inference Dynamics Lab</h1>
<p class="lead">Deterministic scheduler/KV experiments plus a separately labeled actual CPU TinyDecoder probe.</p>
<div class="boundary"><strong>Claim boundary</strong><ul>{boundary}</ul></div>
<section><h2>Policy comparison</h2><table><thead><tr><th>policy</th><th>kind</th><th>completed</th>
<th>TTFT p99 ms</th><th>E2E p99 ms</th><th>output tok/s</th><th>SLO goodput tok/s</th>
<th>reject/timeout/OOM</th><th>KV peak</th><th>modeled efficiency</th></tr></thead>
<tbody>{"".join(policy_rows)}</tbody></table><pre>{html.escape(json.dumps(comparison, indent=2))}</pre></section>
{capacity_section}
<div class="charts">{chart_cards}</div><section><h2>Root-cause attribution</h2>{"".join(root_cards)}</section>
<section><h2>Actual CPU TinyDecoder correctness gate</h2><p>{html.escape(str(probe.get("claim_boundary", "")))}</p>
<table><thead><tr><th>variant</th><th>top-1</th><th>cosine</th><th>KL</th><th>p50 ms</th>
<th>latency ratio</th><th>gate</th></tr></thead><tbody>{"".join(probe_rows)}</tbody></table></section>
<section class="artifacts"><h2>Machine-readable evidence</h2>{artifact_links}</section>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")
    return path
