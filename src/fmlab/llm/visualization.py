"""Portable visual artifacts for losses, attention, and comparisons."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


def save_loss_curve(history: Sequence[Mapping[str, float | int]], path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(steps, losses, color="#2563eb", linewidth=2)
    axis.set(title="Training loss", xlabel="Step", ylabel="Cross-entropy")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def save_attention_svg(
    attention: torch.Tensor | np.ndarray,
    token_labels: Sequence[str],
    path: str | Path,
    *,
    title: str = "Causal self-attention",
) -> Path:
    """Write a dependency-free attention heatmap as an SVG."""

    matrix = torch.as_tensor(attention).detach().cpu().float().numpy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("attention must be a square [query, key] matrix")
    length = matrix.shape[0]
    labels = list(token_labels)[:length]
    labels.extend([f"{index}" for index in range(len(labels), length)])
    cell = max(18, min(34, 620 // max(length, 1)))
    margin_left = 95
    margin_top = 110
    width = margin_left + length * cell + 30
    height = margin_top + length * cell + 35
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="16" y="28" font-family="sans-serif" font-size="18" '
        f'font-weight="600">{html.escape(title)}</text>',
        '<text x="16" y="50" font-family="sans-serif" font-size="11" fill="#475569">'
        "rows=query, columns=key; darker means more attention</text>",
    ]
    for index, label in enumerate(labels):
        x = margin_left + index * cell + cell / 2
        y = margin_top - 7
        escaped = html.escape(label[:10])
        elements.append(
            f'<text x="{x:.1f}" y="{y}" transform="rotate(-55 {x:.1f} {y})" '
            f'text-anchor="start" font-family="monospace" font-size="9">{escaped}</text>'
        )
        elements.append(
            f'<text x="{margin_left - 7}" y="{margin_top + index * cell + cell * 0.68:.1f}" '
            f'text-anchor="end" font-family="monospace" font-size="9">{escaped}</text>'
        )
    for row in range(length):
        for column in range(length):
            value = float(np.clip(matrix[row, column], 0.0, 1.0))
            lightness = 97 - value * 62
            x = margin_left + column * cell
            y = margin_top + row * cell
            elements.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="hsl(217 91% {lightness:.1f}%)" stroke="#e2e8f0" stroke-width="0.5">'
                f"<title>query={html.escape(labels[row])}, key={html.escape(labels[column])}, "
                f"weight={value:.4f}</title></rect>"
            )
    elements.append("</svg>")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(elements), encoding="utf-8")
    return destination


def save_metrics_html(
    title: str,
    metrics: Mapping[str, object],
    artifacts: Sequence[str],
    path: str | Path,
) -> Path:
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metrics.items()
    )
    links = "".join(
        f'<li><a href="{html.escape(item)}">{html.escape(item)}</a></li>' for item in artifacts
    )
    payload = html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:15px system-ui;max-width:900px;margin:40px auto;padding:0 20px;color:#172033}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dbe2ea;padding:8px;text-align:left}}
th{{background:#f4f7fb;width:42%}}code,pre{{background:#f4f7fb;padding:12px;overflow:auto}}</style>
</head><body><h1>{html.escape(title)}</h1><table>{rows}</table>
<h2>Artifacts</h2><ul>{links}</ul><details><summary>JSON</summary><pre>{payload}</pre></details>
</body></html>"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination
