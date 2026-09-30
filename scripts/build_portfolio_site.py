"""Build a deterministic GitHub Pages portfolio from checked-in public evidence.

The public-evidence bundles are the source of truth. Headline numbers are derived here rather than
duplicated by hand, and only sanitized review artifacts are packaged with the site.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EVIDENCE = ROOT / "public-evidence"
SITE_SOURCE = ROOT / "site"
SCORECARD_SOURCE = PUBLIC_EVIDENCE / "meta" / "portfolio-scorecard"


@dataclass(frozen=True)
class Track:
    """Presentation metadata for one public evidence bundle."""

    slug: str
    source_dir: Path
    chart_name: str
    chart_source: str | None
    source_code: str
    number: str
    title_en: str
    title_ko: str
    subtitle_en: str
    subtitle_ko: str
    evidence_en: str
    evidence_ko: str
    evidence_tokens: tuple[str, ...]
    accent: str
    finding_en: str
    finding_ko: str
    boundary_en: str
    boundary_ko: str
    chart_alt: str


TRACKS = (
    Track(
        slug="visual-reward",
        source_dir=PUBLIC_EVIDENCE / "vlm" / "visual-reward-environment",
        chart_name="visual-reward.svg",
        chart_source="reward_audit.svg",
        source_code="src/fmlab/vlm/reward_environment.py",
        number="01",
        title_en="Visual reward contracts",
        title_ko="시각 보상 계약",
        subtitle_en="Hidden-reference grading for document QA, chart QA, and grounding.",
        subtitle_ko="문서 QA·차트 QA·그라운딩을 위한 hidden-reference 채점 환경입니다.",
        evidence_en="Scripted synthetic audit",
        evidence_ko="스크립트 기반 합성 감사",
        evidence_tokens=("synthetic",),
        accent="mint",
        finding_en=(
            "Strict schemas, calibration, grounding, and attack penalties separate true success "
            "from superficially plausible reward exploits."
        ),
        finding_ko=(
            "엄격한 스키마·캘리브레이션·그라운딩·공격 페널티가 실제 성공과 피상적인 "
            "보상 악용을 분리합니다."
        ),
        boundary_en=(
            "Deterministic synthetic tasks and scripted candidates audit the grader. This is not "
            "VLM generation, human-preference agreement, or online RL evidence."
        ),
        boundary_ko=(
            "결정론적 합성 과제와 스크립트 후보로 grader를 검증했습니다. VLM 생성 품질, "
            "사람 선호 일치도, 온라인 RL의 증거는 아닙니다."
        ),
        chart_alt="Robust reward by scripted candidate, including attack candidates.",
    ),
    Track(
        slug="vlm-lora-gpu",
        source_dir=PUBLIC_EVIDENCE / "vlm" / "qwen3-vl-8b-lora-real-2step",
        chart_name="vlm-lora-gpu.svg",
        chart_source=None,
        source_code="src/fmlab/vlm/lora_e2e.py",
        number="02",
        title_en="Qwen3-VL LoRA execution",
        title_ko="Qwen3-VL LoRA 실제 실행",
        subtitle_en="A bounded, local-only BF16 training and held-out generation smoke.",
        subtitle_ko="로컬 BF16 학습과 held-out 생성 경로를 실제 GPU에서 제한적으로 검증했습니다.",
        evidence_en="Actual GPU smoke",
        evidence_ko="실제 GPU smoke",
        evidence_tokens=("actual", "gpu"),
        accent="rose",
        finding_en=(
            "Two optimizer steps completed with a frozen vision tower, an audited LoRA-only "
            "trainable path, and before/after held-out generation."
        ),
        finding_ko=(
            "vision tower 고정, LoRA-only trainable path 감사, 전후 held-out 생성을 "
            "포함해 optimizer 2 step을 완료했습니다."
        ),
        boundary_en=(
            "Three synthetic held-out examples were already perfect: exact match stayed "
            "100% → 100% (delta 0). This proves execution wiring, not quality gain or benchmark value."
        ),
        boundary_ko=(
            "합성 held-out 3개는 baseline부터 완벽해 EM이 100% → 100%(delta 0)였습니다. "
            "실행 경로 증거이며 품질 향상이나 benchmark 성능의 증거가 아닙니다."
        ),
        chart_alt="Measured Qwen3-VL LoRA runtime and GPU memory profile.",
    ),
    Track(
        slug="agent-eval",
        source_dir=PUBLIC_EVIDENCE / "agent" / "reliable-agent-benchmark",
        chart_name="agent-eval.svg",
        chart_source="metrics.svg",
        source_code="src/fmlab/agent/eval_environment.py",
        number="03",
        title_en="Reliable agent evaluation",
        title_ko="신뢰 가능한 에이전트 평가",
        subtitle_en="Outcome + process integrity grading with retries and exact resume.",
        subtitle_ko="결과와 실행 무결성을 함께 채점하고 재시도·정확 재개를 검증합니다.",
        evidence_en="Actual local harness",
        evidence_ko="실제 로컬 하네스",
        evidence_tokens=("actual", "environment"),
        accent="violet",
        finding_en=(
            "Positive and adversarial controls exercise private graders, unsafe-action detection, "
            "fault recovery, immutable tasks, and a content-addressed ledger."
        ),
        finding_ko=(
            "정상·적대 통제 정책으로 private grader, 위험 행동 탐지, 장애 복구, immutable "
            "task와 content-addressed ledger를 검증합니다."
        ),
        boundary_en=(
            "Real files, tools, graders, and processes ran locally, but both policies are "
            "deterministic scripted controls. This is not an LLM benchmark or OS sandbox."
        ),
        boundary_ko=(
            "실제 파일·도구·grader·프로세스를 실행했지만 정책은 스크립트 통제군입니다. "
            "LLM 성능 벤치마크나 OS 보안 샌드박스가 아닙니다."
        ),
        chart_alt="Pass, raw outcome, and reward-hack rates by scripted policy.",
    ),
    Track(
        slug="ddp-correctness",
        source_dir=PUBLIC_EVIDENCE / "distributed" / "ddp-correctness",
        chart_name="ddp-correctness.svg",
        chart_source="resume_equivalence.svg",
        source_code="src/fmlab/distributed/experiment.py",
        number="04",
        title_en="Distributed correctness",
        title_ko="분산 학습 정확성",
        subtitle_en="Two-rank DDP parity, deterministic sharding, and fail-closed resume.",
        subtitle_ko="2-rank DDP 동등성, 결정론적 샤딩, fail-closed 재개를 검증합니다.",
        evidence_en="Actual CPU / Gloo",
        evidence_ko="실제 CPU / Gloo",
        evidence_tokens=("actual",),
        accent="blue",
        finding_en=(
            "A single-process global-batch reference checks gradients and updates while checkpoint "
            "contracts reject drift and byte corruption before state load."
        ),
        finding_ko=(
            "single-process global-batch 기준으로 gradient와 update를 대조하며 checkpoint 계약은 "
            "state load 전에 설정 변화와 byte 손상을 거부합니다."
        ),
        boundary_en=(
            "This is a real two-process CPU/Gloo correctness study on one tiny deterministic "
            "workload. It makes no NCCL, multi-node, convergence, or scaling claim."
        ),
        boundary_ko=(
            "작은 결정론적 workload에서 실제 2-process CPU/Gloo 정확성을 검증했습니다. "
            "NCCL, multi-node, 수렴 또는 확장 성능 주장은 포함하지 않습니다."
        ),
        chart_alt="Overlapping uninterrupted and checkpoint-resumed DDP loss curves.",
    ),
    Track(
        slug="inference-systems",
        source_dir=PUBLIC_EVIDENCE / "systems" / "inference-dynamics",
        chart_name="inference-systems.svg",
        chart_source="capacity.svg",
        source_code="src/fmlab/systems/simulator.py",
        number="05",
        title_en="Inference dynamics",
        title_ko="추론 시스템 동역학",
        subtitle_en="Tail latency, SLO goodput, paged KV, faults, and capacity knees.",
        subtitle_ko="tail latency, SLO goodput, paged KV, 장애, capacity knee를 분석합니다.",
        evidence_en="Simulation + actual CPU probe",
        evidence_ko="시뮬레이션 + 실제 CPU probe",
        evidence_tokens=("simulation", "actual"),
        accent="amber",
        finding_en=(
            "Paired traces expose scheduler trade-offs and overload knees; a separately labeled "
            "tiny CPU probe checks quantization fidelity without claiming integer-kernel speedup."
        ),
        finding_ko=(
            "paired trace로 scheduler trade-off와 과부하 knee를 드러내며 별도 tiny CPU probe는 "
            "정수 kernel 가속을 주장하지 않고 양자화 fidelity만 확인합니다."
        ),
        boundary_en=(
            "Scheduler and capacity results are discrete-event simulation. Fake INT8/INT4 weights "
            "run through FP32 kernels; this is not vLLM, GPU, or fleet telemetry."
        ),
        boundary_ko=(
            "scheduler와 capacity 결과는 discrete-event simulation입니다. Fake INT8/INT4 weight는 "
            "FP32 kernel로 실행되며 vLLM, GPU, fleet telemetry가 아닙니다."
        ),
        chart_alt="Capacity and SLO curves with knees for three scheduler policies.",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _signed_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f}%"


def _dual(en: str, ko: str, *, tag: str = "span", class_name: str = "") -> str:
    attrs = (
        f' data-i18n data-en="{html.escape(en, quote=True)}"'
        f' data-ko="{html.escape(ko, quote=True)}"'
    )
    class_attr = f' class="{class_name}"' if class_name else ""
    return f"<{tag}{class_attr}{attrs}>{html.escape(en)}</{tag}>"


def _metric(value: str, en: str, ko: str, context: str = "") -> dict[str, str]:
    return {"value": value, "en": en, "ko": ko, "context": context}


def _derived_metrics(results: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    visual = results["visual-reward"]["metrics"]
    pref = visual["preference_accuracy"]
    hacks = visual["reward_hacking"]
    gpu = results["vlm-lora-gpu"]["metrics"]
    agent = results["agent-eval"]["metrics"]
    oracle = agent["by_policy"]["oracle_patch_scripted"]
    shortcut = agent["by_policy"]["shortcut_probe_scripted"]
    execution = agent["execution"]
    ddp_result = results["ddp-correctness"]
    ddp = ddp_result["metrics"]
    gates = ddp["gates"]
    simulation = results["inference-systems"]["metrics"]["simulation"]
    static = simulation["static_fcfs"]
    continuous = simulation["continuous_fcfs"]
    throughput_delta = (
        continuous["output_token_throughput_per_second"]
        / static["output_token_throughput_per_second"]
        - 1.0
    )
    goodput_delta = (
        continuous["slo_goodput_tokens_per_second"] / static["slo_goodput_tokens_per_second"] - 1.0
    )
    p99_delta = continuous["latency_ms"]["e2e"]["p99"] / static["latency_ms"]["e2e"]["p99"] - 1.0
    delta = pref["robust_minus_naive"]
    params = gpu["parameters"]
    comparison = gpu["comparison"]
    return {
        "visual-reward": [
            _metric(
                f"{_pct(pref['naive'])} → {_pct(pref['robust'])}",
                "preference accuracy",
                "선호 정확도",
                f"+{delta['delta'] * 100:.1f} pp",
            ),
            _metric(
                f"{_pct(hacks['naive_false_acceptance_rate'], 0)} → "
                f"{_pct(hacks['robust_false_acceptance_rate'], 0)}",
                "attack false acceptance",
                "공격 오수락률",
                f"n={hacks['challenge_count']}",
            ),
            _metric(
                f"[{delta['ci95_low'] * 100:.1f}, {delta['ci95_high'] * 100:.1f}] pp",
                "paired bootstrap 95% CI",
                "paired bootstrap 95% CI",
                f"{delta['samples']} resamples",
            ),
        ],
        "vlm-lora-gpu": [
            _metric(
                f"{gpu['completed_steps']} / {gpu['requested_steps']}",
                "real LoRA steps",
                "실제 LoRA step",
                f"{gpu['runtime_seconds']['total']:.2f} s total",
            ),
            _metric(
                f"{params['trainable_parameters'] / 1_000_000:.2f}M",
                "trainable parameters",
                "학습 parameter",
                f"{params['trainable_percent']:.4f}% of {params['total_parameters'] / 1e9:.2f}B",
            ),
            _metric(
                f"{gpu['gpu_memory_training_peak']['aggregate_peak_allocated_gib']:.2f} GiB",
                "peak GPU allocated",
                "최대 GPU allocated",
                "BF16 · vision frozen",
            ),
            _metric(
                f"{_pct(comparison['before'], 0)} → {_pct(comparison['after'], 0)}",
                "held-out exact match",
                "held-out exact match",
                f"delta 0 · n={comparison['paired_examples']} synthetic",
            ),
        ],
        "agent-eval": [
            _metric(
                f"{_pct(oracle['pass_at_1'], 0)} / {_pct(shortcut['pass_at_1'], 0)}",
                "oracle / shortcut pass@1",
                "oracle / shortcut pass@1",
                f"{agent['episodes']} episodes",
            ),
            _metric(
                f"{execution['episodes_requiring_retry']}/{execution['transient_failures_injected']}",
                "transient faults recovered",
                "일시 장애 복구",
                f"≤ {execution['max_attempts_observed']} attempts",
            ),
            _metric(
                f"{execution['resumed_episodes']}/{agent['episodes']}",
                "episodes exactly resumed",
                "정확히 재개된 episode",
                "0 duplicate executions",
            ),
        ],
        "ddp-correctness": [
            _metric(
                f"{ddp['parity']['gradient']['max_absolute_error']:.2e}",
                "max gradient abs. error",
                "최대 gradient 절대 오차",
                "FP64 reference",
            ),
            _metric(
                f"{ddp['resume']['max_loss_absolute_error']:.1f}",
                "resume loss divergence",
                "재개 loss 차이",
                "exact state match",
            ),
            _metric(
                f"{sum(bool(value) for value in gates.values())}/{len(gates)}",
                "correctness gates passed",
                "통과한 정확성 gate",
                f"{ddp_result['parameters']['world_size']} real processes",
            ),
        ],
        "inference-systems": [
            _metric(
                _signed_pct(throughput_delta),
                "continuous throughput",
                "continuous throughput",
                "vs static FCFS",
            ),
            _metric(
                _signed_pct(goodput_delta),
                "SLO token goodput",
                "SLO token goodput",
                "vs static FCFS",
            ),
            _metric(
                _signed_pct(p99_delta),
                "p99 end-to-end latency",
                "p99 end-to-end latency",
                "paired modeled trace",
            ),
        ],
    }


def _metric_html(metric: dict[str, str]) -> str:
    return (
        '<div class="metric">'
        f'<strong class="metric-value">{html.escape(metric["value"])}</strong>'
        f"{_dual(metric['en'], metric['ko'], class_name='metric-label')}"
        f'<span class="metric-context">{html.escape(metric["context"])}</span></div>'
    )


def _track_card(track: Track, metrics: list[dict[str, str]]) -> str:
    tokens = " ".join(track.evidence_tokens)
    return f"""
<article class="track-card accent-{track.accent}" id="{track.slug}" data-evidence="{tokens}">
  <div class="track-card-header">
    <span class="track-number" aria-hidden="true">{track.number}</span>
    <span class="evidence-badge">{_dual(track.evidence_en, track.evidence_ko)}</span>
  </div>
  {_dual(track.title_en, track.title_ko, tag="h3")}
  {_dual(track.subtitle_en, track.subtitle_ko, tag="p", class_name="track-subtitle")}
  <div class="metrics-grid">{"".join(_metric_html(metric) for metric in metrics)}</div>
  <figure class="evidence-figure">
    <img src="assets/{track.chart_name}" alt="{html.escape(track.chart_alt, quote=True)}"
         width="1120" height="570">
  </figure>
  {_dual(track.finding_en, track.finding_ko, tag="p", class_name="finding")}
  <aside class="claim-boundary">
    <span class="boundary-label">{_dual("Claim boundary", "주장 경계")}</span>
    {_dual(track.boundary_en, track.boundary_ko, tag="p")}
  </aside>
  <div class="card-actions">
    <a class="button button-small" href="evidence/{track.slug}/report.html">
      {_dual("Open report", "보고서 열기")}
    </a>
    <a class="text-link" href="evidence/{track.slug}/result.json">JSON ↗</a>
    <a class="text-link" href="source/{track.slug}.py" data-repo-path="{track.source_code}">
      {_dual("Source", "코드")} ↗
    </a>
  </div>
</article>""".strip()


def _ledger_row(
    track: Track,
    measured: str,
    excludes_en: str,
    excludes_ko: str,
) -> str:
    tokens = " ".join(track.evidence_tokens)
    return f"""
<tr data-evidence="{tokens}">
  <th scope="row"><a href="#{track.slug}">{_dual(track.title_en, track.title_ko)}</a></th>
  <td><span class="table-badge">{_dual(track.evidence_en, track.evidence_ko)}</span></td>
  <td>{html.escape(measured)}</td>
  <td>{_dual(excludes_en, excludes_ko)}</td>
  <td><a href="evidence/{track.slug}/site_manifest.json">Site manifest</a> · <a href="evidence/{track.slug}/source_evidence_manifest.json">Source manifest</a></td>
</tr>""".strip()


def _gpu_profile_svg(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    runtime = metrics["runtime_seconds"]
    total = runtime["total"]
    phases = (
        ("model load", runtime["model_load"], "#fb7185"),
        ("eval before", runtime["evaluation_before"], "#fbbf24"),
        ("train", runtime["training_steps"], "#2dd4bf"),
        ("eval after", runtime["evaluation_after"], "#60a5fa"),
    )
    x = 64.0
    width = 772.0
    pieces: list[str] = []
    for label, value, color in phases:
        segment = width * value / total
        pieces.append(
            f'<rect x="{x:.2f}" y="105" width="{segment:.2f}" height="42" '
            f'rx="4" fill="{color}"><title>{html.escape(label)}: {value:.3f} s</title></rect>'
        )
        x += segment
    legend = "".join(
        f'<circle cx="{76 + index * 185}" cy="178" r="5" fill="{color}"/>'
        f'<text x="{88 + index * 185}" y="183" class="small">{html.escape(label)} '
        f"{value:.2f}s</text>"
        for index, (label, value, color) in enumerate(phases)
    )
    after_load = metrics["gpu_memory_after_load"]["aggregate_peak_allocated_gib"]
    train_peak = metrics["gpu_memory_training_peak"]["aggregate_peak_allocated_gib"]
    memory_scale = 24.0
    after_width = 650 * after_load / memory_scale
    peak_width = 650 * train_peak / memory_scale
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="450"
 viewBox="0 0 900 450" role="img" aria-labelledby="title desc">
<title id="title">Actual Qwen3-VL LoRA GPU smoke profile</title>
<desc id="desc">Runtime phases and measured allocated GPU memory.</desc>
<rect width="100%" height="100%" rx="18" fill="#0b1020"/>
<style>.title{{font:700 25px system-ui;fill:#f4f7ff}}.label{{font:700 15px system-ui;fill:#e7edff}}.small{{font:13px system-ui;fill:#b8c4df}}.value{{font:700 15px ui-monospace;fill:#fff}}</style>
<text x="48" y="48" class="title">Qwen3-VL-8B · actual BF16 LoRA smoke</text>
<text x="48" y="75" class="small">2 optimizer steps · local-only · vision frozen · total {total:.2f}s</text>
<text x="64" y="96" class="label">Measured wall-clock profile</text>
<rect x="64" y="105" width="772" height="42" rx="4" fill="#1d2840"/>
{"".join(pieces)}
{legend}
<text x="64" y="230" class="label">Allocated GPU memory</text>
<text x="64" y="272" class="small">after load</text>
<rect x="164" y="252" width="650" height="26" rx="5" fill="#1d2840"/>
<rect x="164" y="252" width="{after_width:.2f}" height="26" rx="5" fill="#a78bfa"/>
<text x="826" y="271" class="value" text-anchor="end">{after_load:.2f} GiB</text>
<text x="64" y="322" class="small">train peak</text>
<rect x="164" y="302" width="650" height="26" rx="5" fill="#1d2840"/>
<rect x="164" y="302" width="{peak_width:.2f}" height="26" rx="5" fill="#fb7185"/>
<text x="826" y="321" class="value" text-anchor="end">{train_peak:.2f} GiB</text>
<line x1="164" y1="352" x2="814" y2="352" stroke="#43506b"/>
<text x="164" y="374" class="small">0 GiB</text><text x="814" y="374" class="small" text-anchor="end">24 GiB scale</text>
<rect x="48" y="400" width="804" height="30" rx="8" fill="#25182a"/>
<text x="450" y="420" class="small" text-anchor="middle">Held-out synthetic EM: 100% → 100% · delta 0 · not a quality-gain claim</text>
</svg>"""


def _render_index(
    results: dict[str, dict[str, Any]],
    capacity: dict[str, Any],
    scorecard: dict[str, Any],
    digest: str,
) -> str:
    derived = _derived_metrics(results)
    cards = "\n".join(_track_card(track, derived[track.slug]) for track in TRACKS)
    agent_episodes = results["agent-eval"]["metrics"]["episodes"]
    gates = results["ddp-correctness"]["metrics"]["gates"]
    gates_passed = sum(bool(value) for value in gates.values())
    visual_challenges = results["visual-reward"]["metrics"]["reward_hacking"]["challenge_count"]
    gpu = results["vlm-lora-gpu"]["metrics"]
    mode_counts = scorecard["evidence_modes"]["component_counts"]
    rows = "\n".join(
        (
            _ledger_row(
                TRACKS[0],
                f"{visual_challenges} attacks; source-group holdout; paired bootstrap",
                "VLM quality, human preferences, online RL",
                "VLM 품질, 사람 선호, 온라인 RL",
            ),
            _ledger_row(
                TRACKS[1],
                f"{gpu['completed_steps']} real BF16 LoRA steps; "
                f"{gpu['gpu_memory_training_peak']['aggregate_peak_allocated_gib']:.2f} GiB peak",
                "Quality gain, useful convergence, public benchmark value",
                "품질 향상, 유의미한 수렴, 공개 benchmark 성능",
            ),
            _ledger_row(
                TRACKS[2],
                f"{agent_episodes} local episodes; faults, retries, exact resume",
                "LLM capability, SWE-bench scale, security boundary",
                "LLM 성능, SWE-bench 규모, 보안 경계",
            ),
            _ledger_row(
                TRACKS[3],
                f"2 CPU/Gloo ranks; {gates_passed}/{len(gates)} invariant gates",
                "NCCL, multi-node, scaling efficiency",
                "NCCL, multi-node, 확장 효율",
            ),
            _ledger_row(
                TRACKS[4],
                f"{capacity['row_count']} simulations + actual tiny CPU fidelity probe",
                "vLLM, integer kernels, GPU/fleet telemetry",
                "vLLM, 정수 kernel, GPU/fleet telemetry",
            ),
        )
    )
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Evidence-first portfolio for reliable multimodal agents, evaluation, distributed correctness, and inference systems.">
  <meta name="color-scheme" content="dark light">
  <meta name="theme-color" content="#07111e">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'self'; form-action 'none'">
  <title>Reliable Multimodal Agent Systems · Research Engineering Portfolio</title>
  <link rel="stylesheet" href="styles.css">
  <script src="app.js" defer></script>
</head>
<body>
  <a class="skip-link" href="#main">Skip to portfolio evidence</a>
  <header class="site-header">
    <a class="brand" href="#top" aria-label="Portfolio home">
      <span class="brand-mark" aria-hidden="true">RM</span>
      <span class="brand-text">Reliable Systems Lab</span>
    </a>
    <nav class="primary-nav" aria-label="Primary navigation">
      <a href="#flagships">{_dual("Work", "프로젝트")}</a>
      <a href="#evidence-ledger">{_dual("Evidence", "증거")}</a>
      <a href="#method">{_dual("Method", "방법")}</a>
      <a href="#limitations">{_dual("Boundaries", "경계")}</a>
    </nav>
    <div class="header-tools">
      <div class="language-switch" role="group" aria-label="Language">
        <button type="button" data-language="en" aria-pressed="true">EN</button>
        <button type="button" data-language="ko" aria-pressed="false">KO</button>
      </div>
      <button class="theme-toggle" type="button" aria-label="Toggle color theme"
              title="Toggle color theme"><span aria-hidden="true">◐</span></button>
    </div>
  </header>

  <main id="main">
    <section class="hero" id="top" aria-labelledby="hero-title">
      <div class="hero-copy">
        <p class="eyebrow"><span class="status-dot" aria-hidden="true"></span>
          {_dual("Evidence-first research engineering", "증거 중심 리서치 엔지니어링")}
        </p>
        <h1 id="hero-title">
          {
        _dual(
            "Reliable multimodal agent systems—claims you can audit.",
            "감사 가능한 주장으로 만드는 신뢰성 높은 멀티모달 에이전트 시스템.",
        )
    }
        </h1>
        {
        _dual(
            "I connect multimodal post-training and reward design, agent evaluation, distributed "
            "semantics, and inference behavior. Every headline number resolves to code, a result "
            "JSON, and an explicit limitation.",
            "멀티모달 post-training·보상 설계, 에이전트 평가, 분산 학습 의미론, 추론 동작을 "
            "연결합니다. 모든 핵심 수치는 코드·result JSON·명시적 한계로 추적됩니다.",
            tag="p",
            class_name="hero-lede",
        )
    }
        <div class="hero-actions">
          <a class="button button-primary" href="#review-path">
            {_dual("Start 5-minute review", "5분 리뷰 시작")} ↓
          </a>
          <a class="button button-secondary" href="#flagships">
            {_dual("Explore evidence", "증거 살펴보기")}
          </a>
        </div>
        <dl class="hero-stats" aria-label="Portfolio evidence summary">
          <div><dt>5</dt><dd>{_dual("source evidence bundles", "원천 evidence bundle")}</dd></div>
          <div><dt>2</dt><dd>{_dual("real 8B LoRA steps", "실제 8B LoRA step")}</dd></div>
          <div><dt>{capacity["row_count"]}</dt><dd>{
        _dual("systems trials", "systems trial")
    }</dd></div>
          <div><dt>{gates_passed}/{len(gates)}</dt><dd>{
        _dual("DDP gates passed", "통과한 DDP gate")
    }</dd></div>
        </dl>
      </div>
      <aside class="hero-console" aria-label="Research system contract">
        <div class="console-bar"><span></span><span></span><span></span>
          <code>evidence.contract</code>
        </div>
        <div class="console-body">
          <p><span class="console-key">position</span><span class="console-value">reliable_multimodal_agents</span></p>
          <p><span class="console-key">pipeline</span><span class="console-value">data → train/reward → agent → eval → serve</span></p>
          <p><span class="console-key">controls</span><span class="console-value">holdout · attack · fault · resume</span></p>
          <p><span class="console-key">release</span><span class="console-value">manifest + SHA-256 + boundary</span></p>
          <div class="contract-ok"><span aria-hidden="true">✓</span>
            {_dual("Claim boundary attached", "주장 경계 연결됨")}
          </div>
        </div>
        <div class="signal-flow" aria-label="End-to-end research loop">
          <span>DATA</span><i></i><span>TRAIN</span><i></i><span>AGENT</span><i></i>
          <span>EVAL</span><i></i><span>SERVE</span>
        </div>
      </aside>
    </section>

    <section class="review-path section" id="review-path" aria-labelledby="review-title">
      <div class="section-heading">
        <p class="section-kicker">00 — REVIEW PATH</p>
        <h2 id="review-title">
          {
        _dual(
            "A reviewer can verify the work in five minutes.",
            "5분 안에 검증할 수 있는 리뷰 경로",
        )
    }
        </h2>
        {
        _dual(
            "Follow one evidence chain from reward behavior through real GPU wiring, agent "
            "integrity, distributed semantics, and serving limits.",
            "보상 동작에서 실제 GPU 경로, 에이전트 무결성, 분산 의미론, serving 한계까지 "
            "하나의 evidence chain으로 검토합니다.",
            tag="p",
        )
    }
      </div>
      <ol class="review-timeline">
        <li><span class="time">0:00–0:55</span><a href="#visual-reward">{
        _dual("Audit rewards", "보상 감사")
    }</a><p>Ablation · attacks · holdout</p></li>
        <li><span class="time">0:55–1:50</span><a href="#vlm-lora-gpu">{
        _dual("Verify GPU run", "GPU 실행 확인")
    }</a><p>BF16 · LoRA · held-out</p></li>
        <li><span class="time">1:50–2:50</span><a href="#agent-eval">{
        _dual("Probe integrity", "무결성 점검")
    }</a><p>Graders · retry · resume</p></li>
        <li><span class="time">2:50–3:50</span><a href="#ddp-correctness">{
        _dual("Check DDP", "DDP 확인")
    }</a><p>Parity · sharding · contract</p></li>
        <li><span class="time">3:50–5:00</span><a href="#inference-systems">{
        _dual("Find knees", "knee 확인")
    }</a><p>p99 · goodput · capacity</p></li>
      </ol>
    </section>

    <section class="flagships section" id="flagships" aria-labelledby="flagships-title">
      <div class="section-heading split-heading">
        <div><p class="section-kicker">01 — FLAGSHIP WORK</p>
          <h2 id="flagships-title">
            {
        _dual(
            "Five linked systems, not disconnected demos.",
            "분리된 데모가 아닌, 연결된 다섯 개의 시스템",
        )
    }
          </h2>
        </div>
        {
        _dual(
            "Filter by what actually ran. Actual never silently upgrades simulated or scripted evidence.",
            "실제로 실행된 증거 유형으로 필터링하세요. 실측이 simulation이나 scripted evidence를 "
            "암묵적으로 승격하지 않습니다.",
            tag="p",
        )
    }
      </div>
      <div class="filter-bar" role="group" aria-label="Filter projects by evidence class">
        <button type="button" data-filter="all" aria-pressed="true">
          {_dual("All evidence", "전체")}
        </button>
        <button type="button" data-filter="actual" aria-pressed="false">
          <span class="filter-dot dot-actual"></span>{_dual("Actual execution", "실제 실행")}
        </button>
        <button type="button" data-filter="synthetic" aria-pressed="false">
          <span class="filter-dot dot-synthetic"></span>{_dual("Synthetic audit", "합성 감사")}
        </button>
        <button type="button" data-filter="simulation" aria-pressed="false">
          <span class="filter-dot dot-simulation"></span>{_dual("Simulation", "시뮬레이션")}
        </button>
        <span class="filter-status" role="status" aria-live="polite">5 projects shown</span>
      </div>
      <div class="track-grid">{cards}</div>
    </section>

    <section class="ledger section" id="evidence-ledger" aria-labelledby="ledger-title">
      <div class="section-heading split-heading">
        <div><p class="section-kicker">02 — EVIDENCE LEDGER</p>
          <h2 id="ledger-title">
            {
        _dual("Measured, manifested, bounded.", "측정하고, manifest로 남기고, 경계를 명시합니다.")
    }
          </h2>
        </div>
        <div class="search-wrap">
          <label class="visually-hidden" for="evidence-search">Search evidence ledger</label>
          <input id="evidence-search" type="search" placeholder="Search evidence…" autocomplete="off">
          <span aria-hidden="true">⌕</span>
        </div>
      </div>
      <article class="scorecard-panel" aria-labelledby="scorecard-title">
        <div class="scorecard-copy">
          <span class="aggregate-badge">
            {_dual("Aggregate view · not an experiment", "집계 뷰 · 개별 실험 아님")}
          </span>
          <h3 id="scorecard-title">
            {_dual("One map of every evidence mode.", "모든 증거 유형을 한눈에 보는 지도")}
          </h3>
          {
        _dual(
            "The scorecard aggregates five source bundles without creating a sixth performance "
            "claim. Component counts preserve mixed modes inside a track.",
            "scorecard는 다섯 원천 bundle을 집계하며 여섯 번째 성능 주장을 만들지 않습니다. "
            "한 track 안의 혼합 evidence mode도 그대로 유지합니다.",
            tag="p",
        )
    }
          <dl class="mode-counts">
            <div><dt>{mode_counts["actual"]}</dt><dd>{
        _dual("actual components", "actual component")
    }</dd></div>
            <div><dt>{mode_counts["scripted"]}</dt><dd>{
        _dual("scripted controls", "scripted control")
    }</dd></div>
            <div><dt>{mode_counts["simulated"]}</dt><dd>{
        _dual("simulated components", "simulated component")
    }</dd></div>
          </dl>
          <div class="card-actions">
            <a class="button button-small" href="meta/portfolio-scorecard/scorecard.json">Scorecard JSON</a>
            <a class="text-link" href="meta/portfolio-scorecard/evidence_manifest.json">Aggregate manifest ↗</a>
          </div>
        </div>
        <figure>
          <img src="meta/portfolio-scorecard/scorecard.svg"
               alt="Aggregate evidence-mode scorecard across five source bundles."
               width="1200" height="760">
        </figure>
      </article>
      <div class="table-wrap">
        <table>
          <caption>
            {
        _dual(
            "Public evidence classes and unsupported claims for every flagship.",
            "각 flagship의 공개 증거 등급과 지원하지 않는 주장",
        )
    }
          </caption>
          <thead><tr>
            <th scope="col">{_dual("Track", "트랙")}</th>
            <th scope="col">{_dual("Evidence class", "증거 등급")}</th>
            <th scope="col">{_dual("Measured", "측정 내용")}</th>
            <th scope="col">{_dual("Does not establish", "입증하지 않는 것")}</th>
            <th scope="col">{_dual("Integrity", "무결성")}</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <p class="empty-state" hidden>
          {_dual("No evidence rows match that search.", "검색과 일치하는 evidence row가 없습니다.")}
        </p>
      </div>
    </section>

    <section class="method section" id="method" aria-labelledby="method-title">
      <div class="section-heading"><p class="section-kicker">03 — RESEARCH METHOD</p>
        <h2 id="method-title">
          {_dual("Reliability is designed in before the run.", "신뢰성은 실행 전에 설계합니다.")}
        </h2>
      </div>
      <div class="method-grid">
        <article><span class="method-icon">{{ }}</span><h3>{_dual("Contracts", "계약")}</h3><p>{
        _dual(
            "Typed observations, immutable tasks, config hashes, and fail-closed validation.",
            "typed observation, immutable task, config hash와 fail-closed 검증",
        )
    }</p></article>
        <article><span class="method-icon">↯</span><h3>{
        _dual("Negative controls", "음성 통제")
    }</h3><p>{
        _dual(
            "Reward attacks, unsafe actions, injected faults, corruption, and overload.",
            "보상 공격, 위험 행동, 주입 장애, 손상, 과부하",
        )
    }</p></article>
        <article><span class="method-icon">±</span><h3>{
        _dual("Statistical discipline", "통계적 규율")
    }</h3><p>{
        _dual(
            "Grouped holdouts, paired traces, bootstrap intervals, and fixed seeds.",
            "grouped holdout, paired trace, bootstrap interval과 고정 seed",
        )
    }</p></article>
        <article><span class="method-icon">#</span><h3>{
        _dual("Auditable release", "감사 가능한 공개")
    }</h3><p>{
        _dual(
            "Sanitized artifacts, explicit allowlists, SHA-256, and no weights or private traces.",
            "sanitized artifact, allowlist, SHA-256, weight·private trace 제외",
        )
    }</p></article>
      </div>
    </section>

    <section class="limitations section" id="limitations" aria-labelledby="limitations-title">
      <div class="limitations-panel">
        <div class="section-heading"><p class="section-kicker">04 — CLAIM BOUNDARIES</p>
          <h2 id="limitations-title">
            {
        _dual(
            "What remains unproven is part of the result.",
            "아직 입증하지 못한 것도 결과의 일부입니다.",
        )
    }
          </h2>
          {
        _dual(
            "The next promotion must change the evidence class, not only add another toy run.",
            "다음 단계는 toy run을 더하는 것이 아니라 증거 등급을 올려야 합니다.",
            tag="p",
        )
    }
        </div>
        <ol class="promotion-list">
          <li><span>01</span><div><strong>{
        _dual("Real multimodal post-training", "실제 멀티모달 post-training")
    }</strong><p>{
        _dual(
            "Three-seed training on releasable, non-synthetic grouped holdouts.",
            "공개 가능한 비합성 grouped holdout에서 3-seed 학습",
        )
    }</p></div></li>
          <li><span>02</span><div><strong>{
        _dual("Online visual RLVR", "온라인 visual RLVR")
    }</strong><p>{
        _dual(
            "Reward rollouts with overoptimization and grader-transfer evaluation.",
            "reward rollout과 overoptimization·grader transfer 평가",
        )
    }</p></div></li>
          <li><span>03</span><div><strong>{
        _dual("Calibrated GPU serving", "보정된 GPU serving")
    }</strong><p>{
        _dual(
            "Replay against vLLM or SGLang and fit to real GPU traces.",
            "vLLM/SGLang replay 후 실제 GPU trace로 보정",
        )
    }</p></div></li>
          <li><span>04</span><div><strong>{
        _dual("Multi-GPU semantics and scale", "Multi-GPU 의미론과 확장성")
    }</strong><p>{
        _dual(
            "Re-run invariants on NCCL, then measure scaling separately.",
            "NCCL invariant 재검증 후 scaling 별도 측정",
        )
    }</p></div></li>
        </ol>
      </div>
    </section>

    <section class="closing section" aria-labelledby="closing-title">
      <p class="section-kicker">05 — TRACE THE CLAIM</p>
      <h2 id="closing-title">
        {
        _dual(
            "Start with a number. End at the invariant that produced it.",
            "숫자에서 시작해 그 숫자를 만든 invariant까지 추적하세요.",
        )
    }
      </h2>
      <div class="closing-actions">
        <a class="button button-primary" href="#review-path">
          {_dual("Run the review path", "리뷰 경로 실행")}
        </a>
        <a class="button button-secondary" href="evidence/vlm-lora-gpu/result.json">
          {_dual("Inspect machine-readable evidence", "machine-readable evidence 확인")}
        </a>
      </div>
    </section>
  </main>

  <footer>
    <div><strong>Reliable Systems Lab</strong>
      <p>{
        _dual("Multimodal agents, evaluation, and ML systems.", "멀티모달 에이전트·평가·ML 시스템")
    }</p>
    </div>
    <div class="build-stamp"><span>Evidence snapshot</span><code>{digest[:12]}</code>
      <span>· 5 source + 1 aggregate manifests</span>
    </div>
  </footer>
  <noscript><p class="noscript">Filters and language switching require JavaScript. Reports remain accessible.</p></noscript>
</body>
</html>
"""


_PRIVATE_PATTERN = re.compile(
    r"(?:/home/|/data/|BEGIN [A-Z ]+PRIVATE KEY)",
    re.IGNORECASE,
)
_UNSAFE_SVG_PATTERN = re.compile(
    r"(?:<script\\b|\\bon(?:load|error|click)\\s*=|(?:href|src)\\s*=\\s*['\\\"]https?://)",
    re.IGNORECASE,
)


def _validate_public_text(path: Path, text: str) -> None:
    if "\x00" in text:
        raise ValueError(f"NUL byte in public artifact: {path}")
    if _PRIVATE_PATTERN.search(text):
        raise ValueError(f"Private path or key marker in public artifact: {path}")


def _copy_public_text(source: Path, target: Path, *, svg: bool = False) -> None:
    text = source.read_text(encoding="utf-8")
    _validate_public_text(source, text)
    if svg and (not text.lstrip().startswith("<svg") or _UNSAFE_SVG_PATTERN.search(text)):
        raise ValueError(f"Unsafe SVG rejected: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


_HTML_REFERENCE = re.compile(
    r"""(?P<attribute>href|src)=(?P<quote>['"])(?P<target>[^'"]+)(?P=quote)""",
    re.IGNORECASE,
)
_BUNDLE_SUFFIXES = {".html", ".json", ".svg"}


def _verify_evidence_manifest(directory: Path) -> dict[str, Any]:
    """Fail closed unless every manifest entry matches its public file."""

    manifest_path = directory / "evidence_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"Missing regular evidence manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported evidence manifest schema: {manifest_path}")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Evidence manifest has no files: {manifest_path}")

    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid evidence manifest entry: {manifest_path}")
        relative_raw = entry.get("path")
        if not isinstance(relative_raw, str) or not relative_raw:
            raise ValueError(f"Invalid evidence path: {manifest_path}")
        relative = Path(relative_raw)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError(f"Unsafe evidence path: {relative_raw}")
        if relative_raw in seen:
            raise ValueError(f"Duplicate evidence path: {relative_raw}")
        seen.add(relative_raw)

        source = directory / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing regular evidence file: {source}")
        expected_bytes = entry.get("public_bytes")
        expected_sha = entry.get("public_sha256")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f"Invalid public_bytes for {source}")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError(f"Invalid public_sha256 for {source}")
        payload = source.read_bytes()
        if len(payload) != expected_bytes:
            raise ValueError(f"Evidence byte-count mismatch: {source}")
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"Evidence SHA-256 mismatch: {source}")
    return manifest


def _manifest_paths(manifest: dict[str, Any]) -> set[str]:
    return {entry["path"] for entry in manifest["files"]}


def _write_site_manifest(
    target_dir: Path,
    source_manifest_sha256: str,
    transformations: list[str],
) -> None:
    files = []
    for path in sorted(target_dir.iterdir()):
        if not path.is_file() or path.name == "site_manifest.json":
            continue
        payload = path.read_bytes()
        files.append(
            {
                "path": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema": "fmlab.portfolio-site-bundle/v1",
        "source_manifest": "source_evidence_manifest.json",
        "source_manifest_sha256": source_manifest_sha256,
        "transformations": transformations,
        "files": files,
    }
    (target_dir / "site_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _package_evidence_bundle(track: Track, target_dir: Path) -> None:
    """Package only manifest-verified files and neutralize excluded report links."""

    manifest = _verify_evidence_manifest(track.source_dir)
    allowed = _manifest_paths(manifest)
    if "result.json" not in allowed or "report.html" not in allowed:
        raise ValueError(f"Required review artifacts are not manifested: {track.source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = track.source_dir / "evidence_manifest.json"
    _copy_public_text(source_manifest, target_dir / "source_evidence_manifest.json")
    for relative in sorted(allowed):
        source = track.source_dir / relative
        if relative == "report.html":
            continue
        if source.suffix.lower() not in _BUNDLE_SUFFIXES:
            raise ValueError(f"Unsupported manifested site artifact: {source}")
        _copy_public_text(
            source,
            target_dir / relative,
            svg=source.suffix.lower() == ".svg",
        )

    report_source = track.source_dir / "report.html"
    report = report_source.read_text(encoding="utf-8")
    neutralized: list[str] = []
    _validate_public_text(report_source, report)

    def replace_reference(match: re.Match[str]) -> str:
        target = match.group("target")
        if target.startswith(("#", "http://", "https://", "mailto:", "data:")):
            return match.group(0)
        local_part = target.split("#", 1)[0].split("?", 1)[0]
        if local_part in allowed:
            return match.group(0)
        if match.group("attribute").lower() == "src":
            raise ValueError(f"Unmanifested report image: {track.source_dir / local_part}")
        neutralized.append(local_part)
        quote = match.group("quote")
        safe_target = html.escape(target, quote=True)
        return f"data-unpublished-reference={quote}{safe_target}{quote}"

    report = _HTML_REFERENCE.sub(replace_reference, report)
    _validate_public_text(report_source, report)
    (target_dir / "report.html").write_text(report, encoding="utf-8")
    source_manifest_sha = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    _write_site_manifest(
        target_dir,
        source_manifest_sha,
        (
            [f"Neutralized excluded report links: {', '.join(sorted(neutralized))}"]
            if neutralized
            else []
        ),
    )


def _evidence_digest() -> str:
    digest = hashlib.sha256()
    for track in TRACKS:
        digest.update(track.slug.encode())
        digest.update(b"\0")
        digest.update((track.source_dir / "result.json").read_bytes())
    digest.update(b"portfolio-scorecard\0")
    digest.update((SCORECARD_SOURCE / "scorecard.json").read_bytes())
    return digest.hexdigest()


def _copy_scorecard_bundle(output_dir: Path) -> None:
    manifest = _verify_evidence_manifest(SCORECARD_SOURCE)
    target = output_dir / "meta" / "portfolio-scorecard"
    target.mkdir(parents=True, exist_ok=True)
    _copy_public_text(
        SCORECARD_SOURCE / "evidence_manifest.json",
        target / "evidence_manifest.json",
    )
    for relative in sorted(_manifest_paths(manifest)):
        source = SCORECARD_SOURCE / relative
        if source.suffix.lower() not in _BUNDLE_SUFFIXES:
            raise ValueError(f"Unsupported scorecard artifact: {source}")
        _copy_public_text(
            source,
            target / relative,
            svg=source.suffix.lower() == ".svg",
        )


_GENERATED_DIRECTORIES = ("assets", "evidence", "source", "meta")


def _reset_generated_output(output_dir: Path) -> None:
    """Remove only directories and files owned by this deterministic builder."""

    if output_dir in {Path("/"), ROOT, ROOT.parent}:
        raise ValueError(f"Refusing unsafe site output directory: {output_dir}")
    for name in _GENERATED_DIRECTORIES:
        target = output_dir / name
        if target.is_symlink():
            raise ValueError(f"Refusing generated-output symlink: {target}")
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"Generated-output path is not a directory: {target}")
            shutil.rmtree(target)
    index = output_dir / "index.html"
    if index.is_symlink():
        raise ValueError(f"Refusing generated-index symlink: {index}")
    if index.exists():
        if not index.is_file():
            raise ValueError(f"Generated index is not a file: {index}")
        index.unlink()


def build_site(output_dir: Path = SITE_SOURCE) -> Path:
    """Build the site into output_dir and return the generated index path."""

    for track in TRACKS:
        _verify_evidence_manifest(track.source_dir)
    _verify_evidence_manifest(SCORECARD_SOURCE)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_generated_output(output_dir)
    results = {track.slug: _load_json(track.source_dir / "result.json") for track in TRACKS}
    capacity = _load_json(
        PUBLIC_EVIDENCE / "systems" / "inference-dynamics" / "capacity_sweep.json"
    )
    scorecard = _load_json(SCORECARD_SOURCE / "scorecard.json")
    for static_name in ("styles.css", "app.js", "404.html", ".nojekyll"):
        source = SITE_SOURCE / static_name
        if output_dir != SITE_SOURCE:
            _copy_public_text(source, output_dir / static_name)

    for track in TRACKS:
        evidence_dir = output_dir / "evidence" / track.slug
        _copy_public_text(ROOT / track.source_code, output_dir / "source" / f"{track.slug}.py")
        _package_evidence_bundle(track, evidence_dir)
        chart_target = output_dir / "assets" / track.chart_name
        if track.chart_source:
            manifest = _verify_evidence_manifest(track.source_dir)
            if track.chart_source not in _manifest_paths(manifest):
                raise ValueError(f"Card chart is not manifested: {track.chart_source}")
            _copy_public_text(
                track.source_dir / track.chart_source,
                chart_target,
                svg=True,
            )
        else:
            svg = _gpu_profile_svg(results[track.slug])
            _validate_public_text(chart_target, svg)
            chart_target.parent.mkdir(parents=True, exist_ok=True)
            chart_target.write_text(svg, encoding="utf-8")

    _copy_scorecard_bundle(output_dir)
    index = _render_index(results, capacity, scorecard, _evidence_digest())
    _validate_public_text(output_dir / "index.html", index)
    index_path = output_dir / "index.html"
    index_path.write_text(index, encoding="utf-8")
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SITE_SOURCE,
        help="Output directory (default: repository site/)",
    )
    return parser.parse_args()


def main() -> None:
    index = build_site(parse_args().output)
    print(f"Built deterministic portfolio site: {index}")


if __name__ == "__main__":
    main()
