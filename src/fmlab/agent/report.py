from __future__ import annotations

import html
import json
from pathlib import Path

from .loop import CodingAgentResult


def render_trace_html(result: CodingAgentResult, output: str | Path) -> Path:
    cards: list[str] = []
    for event in result.trace:
        payload = html.escape(json.dumps(event.payload, ensure_ascii=False, indent=2))
        cards.append(
            f"<section class='event {html.escape(event.kind)}'>"
            f"<h2>Step {event.step} · {html.escape(event.kind)}</h2>"
            f"<p>{event.elapsed_seconds:.3f}s</p><pre>{payload}</pre></section>"
        )
    document = f"""<!doctype html>
<html lang='ko'><head><meta charset='utf-8'><title>Coding Agent Trace</title>
<style>
body{{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#0b1020;color:#e8edf7}}
.summary,.event{{background:#151d33;border:1px solid #2a3658;border-radius:12px;padding:1rem;margin:1rem 0}}
.tool{{border-left:5px solid #55c2ff}} .model{{border-left:5px solid #b895ff}} .final{{border-left:5px solid #65d995}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#090d18;padding:1rem;border-radius:8px}}
</style></head><body><h1>Codex-style Coding Agent Trace</h1>
<section class='summary'><b>Status:</b> {html.escape(result.status)} · <b>Steps:</b> {result.steps}<br>
<b>Workspace:</b> {html.escape(result.workspace)}<p>{html.escape(result.summary)}</p></section>
{"".join(cards)}</body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
