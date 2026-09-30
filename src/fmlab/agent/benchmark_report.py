from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_metrics_svg(policy_metrics: dict[str, dict[str, Any]], output: Path) -> Path:
    width = 900
    height = 150 + 150 * len(policy_metrics)
    chart_left = 300
    chart_width = 500
    rows: list[str] = []
    for index, (policy, metrics) in enumerate(sorted(policy_metrics.items())):
        y = 105 + index * 150
        rows.append(f'<text x="24" y="{y}" class="label">{html.escape(policy)}</text>')
        values = (
            ("integrity pass@1", float(metrics["pass_at_1"]), "#41d69f"),
            ("outcome pass", float(metrics["outcome_pass_rate"]), "#63a8ff"),
            ("reward-hack", float(metrics["reward_hack_rate"]), "#ff6b7d"),
        )
        for offset, (label, value, color) in enumerate(values):
            bar_y = y - 24 + offset * 32
            bar_width = round(chart_width * min(max(value, 0.0), 1.0), 2)
            rows.extend(
                [
                    f'<text x="160" y="{bar_y + 14}" class="small">{label}</text>',
                    f'<rect x="{chart_left}" y="{bar_y}" width="{chart_width}" height="18" '
                    'rx="4" fill="#17213a"/>',
                    f'<rect x="{chart_left}" y="{bar_y}" width="{bar_width}" height="18" '
                    f'rx="4" fill="{color}"/>',
                    f'<text x="{chart_left + chart_width + 12}" y="{bar_y + 14}" '
                    f'class="value">{value:.1%}</text>',
                ]
            )
    document = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
 viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">Reliable Agent Benchmark policy metrics</title>
<desc id="desc">Integrity-qualified pass, raw hidden outcome pass, and reward-hack rate.</desc>
<rect width="100%" height="100%" rx="18" fill="#0b1020"/>
<style>
.title{{font:700 25px system-ui;fill:#f4f7ff}} .label{{font:700 16px system-ui;fill:#e7edff}}
.small{{font:13px system-ui;fill:#b8c4df}} .value{{font:700 13px ui-monospace;fill:#f4f7ff}}
</style>
<text x="24" y="42" class="title">Reliable Agent Eval · policy controls</text>
{"".join(rows)}
</svg>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def _metric_card(label: str, value: str, note: str) -> str:
    return (
        "<section class='card'><span>"
        + html.escape(label)
        + "</span><strong>"
        + html.escape(value)
        + "</strong><small>"
        + html.escape(note)
        + "</small></section>"
    )


def render_benchmark_html(
    result: dict[str, Any],
    episodes: list[dict[str, Any]],
    task_manifest: list[dict[str, Any]],
    output: Path,
) -> Path:
    metrics = result["metrics"]
    cards = [
        _metric_card(
            "Integrity pass@1",
            f"{metrics['pass_at_1']:.1%}",
            "Hidden outcome + process-integrity gate",
        ),
        _metric_card(
            "Raw outcome pass",
            f"{metrics['outcome_pass_rate']:.1%}",
            "Before reward-hack and validation gates",
        ),
        _metric_card(
            "Reward-hack episodes",
            f"{metrics['reward_hack_rate']:.1%}",
            "Any detected integrity violation",
        ),
        _metric_card(
            "Episodes",
            str(metrics["episodes"]),
            f"{metrics['tasks']} tasks · {metrics['policies']} scripted controls",
        ),
    ]

    policy_rows = []
    for policy, values in sorted(metrics["by_policy"].items()):
        ci = values["pass_at_1_ci_95"]
        policy_rows.append(
            "<tr>"
            f"<td>{html.escape(policy)}</td>"
            f"<td>{values['episodes']}</td>"
            f"<td>{values['pass_at_1']:.1%}</td>"
            f"<td>[{ci[0]:.1%}, {ci[1]:.1%}]</td>"
            f"<td>{values['outcome_pass_rate']:.1%}</td>"
            f"<td>{values['reward_hack_rate']:.1%}</td>"
            f"<td>{values['mean_steps']:.2f}</td>"
            f"<td>{values['p50_latency_seconds']:.4f}s</td>"
            "</tr>"
        )

    failure_rows = "".join(
        f"<tr><td>{html.escape(category)}</td><td>{count}</td></tr>"
        for category, count in sorted(metrics["failure_taxonomy"].items())
    )
    task_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['task_id'])}</td>"
        f"<td>{html.escape(item['perturbation_id'])}</td>"
        f"<td><code>{html.escape(item['fingerprint_sha256'][:16])}</code></td>"
        f"<td>{item['hidden_case_count']}</td>"
        "</tr>"
        for item in task_manifest
    )
    failed = [episode for episode in episodes if not episode.get("passed")]
    failure_details = "".join(
        "<details><summary>"
        + html.escape(
            f"{episode['policy_id']} · {episode['task_uid']} · {episode['failure_category']}"
        )
        + "</summary><pre>"
        + html.escape(json.dumps(episode, ensure_ascii=False, indent=2))
        + "</pre></details>"
        for episode in failed
    )
    claim = result["claim"]
    isolation_note = (
        "Network namespace isolation was active."
        if claim["network_isolated"]
        else "Process mode was used: resource/path bounds are measured, but network and kernel "
        "isolation are not provided. Do not run untrusted model code in this mode."
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Reliable Agent Eval Environment</title>
<style>
:root{{--bg:#080d19;--panel:#111a2d;--line:#273654;--text:#edf3ff;--muted:#a9b6d1;
--green:#41d69f;--red:#ff6b7d;--blue:#63a8ff}} *{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:36px 22px 80px}} h1{{font-size:34px;margin-bottom:4px}}
h2{{margin-top:38px}} p,small{{color:var(--muted)}} .claim{{border-left:5px solid var(--blue);
background:var(--panel);padding:16px 20px;border-radius:10px}} .grid{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:22px 0}}
.card{{background:var(--panel);border:1px solid var(--line);padding:16px;border-radius:12px}}
.card span,.card small{{display:block}} .card strong{{font-size:30px;display:block;margin:6px 0}}
table{{width:100%;border-collapse:collapse;background:var(--panel)}} th,td{{padding:10px 12px;
border:1px solid var(--line);text-align:left}} th{{color:#cbd8f2}} code,pre{{font-family:ui-monospace,monospace}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#070b14;padding:14px;border-radius:8px}}
details{{background:var(--panel);border:1px solid var(--line);padding:10px 14px;margin:8px 0;
border-radius:9px}} a{{color:#8ec5ff}} img{{max-width:100%;height:auto}}
</style></head><body><main>
<h1>Reliable Agent Eval Environment</h1>
<p>Deterministic local benchmark for environment and grader integrity.</p>
<section class="claim"><strong>Claim level: {html.escape(result["claim_level"])}</strong>
<p>{html.escape(claim["scope"])}</p><p>{html.escape(isolation_note)}</p></section>
<div class="grid">{"".join(cards)}</div>
<img src="metrics.svg" alt="Policy metric bars">
<h2>Policy controls</h2>
<table><thead><tr><th>Policy</th><th>N</th><th>Pass@1</th><th>95% bootstrap CI</th>
<th>Outcome pass</th><th>Reward hack</th><th>Mean steps</th><th>P50 latency</th></tr></thead>
<tbody>{"".join(policy_rows)}</tbody></table>
<h2>Failure taxonomy</h2><table><thead><tr><th>Category</th><th>Episodes</th></tr></thead>
<tbody>{failure_rows}</tbody></table>
<h2>Versioned task commitments</h2><table><thead><tr><th>Task</th><th>Perturbation</th>
<th>SHA-256 prefix</th><th>Hidden cases</th></tr></thead><tbody>{task_rows}</tbody></table>
<h2>Failure evidence</h2>{failure_details or "<p>No failures.</p>"}
<h2>Machine-readable artifacts</h2><ul><li><a href="result.json">result.json</a></li>
<li><a href="ledger.jsonl">atomic resume ledger</a></li><li><a href="traces.jsonl">full traces</a></li>
<li><a href="task_manifest.json">task manifest</a></li></ul>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
