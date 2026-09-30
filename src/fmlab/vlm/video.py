from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    total_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float


@dataclass(frozen=True)
class SampledFrame:
    index: int
    timestamp_seconds: float
    output_path: Path | None = None


def plan_frame_indices(
    total_frames: int,
    fps: float,
    *,
    max_frames: int = 12,
    strategy: Literal["uniform", "head", "every_n_seconds"] = "uniform",
    every_n_seconds: float = 1.0,
) -> list[SampledFrame]:
    """Plan frame sampling without decoding a video (used by tiny/offline mode)."""

    if total_frames < 0 or fps <= 0 or max_frames < 1:
        raise ValueError("total_frames >= 0, fps > 0, and max_frames >= 1 are required")
    if total_frames == 0:
        return []
    if strategy == "head":
        indices = list(range(min(total_frames, max_frames)))
    elif strategy == "every_n_seconds":
        if every_n_seconds <= 0:
            raise ValueError("every_n_seconds must be positive")
        step = max(1, round(fps * every_n_seconds))
        indices = list(range(0, total_frames, step))[:max_frames]
    elif strategy == "uniform":
        sample_count = min(total_frames, max_frames)
        if sample_count == 1:
            indices = [0]
        else:
            indices = [
                round(i * (total_frames - 1) / (sample_count - 1)) for i in range(sample_count)
            ]
            indices = list(dict.fromkeys(indices))
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    return [SampledFrame(index=index, timestamp_seconds=index / fps) for index in indices]


def probe_video(path: Path) -> VideoInfo:
    """Read video metadata with ffprobe; no video frames are retained."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except FileNotFoundError as exc:  # pragma: no cover - system integration
        raise RuntimeError("ffprobe is required for video metadata") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise RuntimeError(f"ffprobe failed: {exc.stderr.strip()}") from exc
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split("/", 1)
    fps = float(numerator) / float(denominator)
    duration = float(stream.get("duration") or 0.0)
    total_frames = int(stream.get("nb_frames") or round(duration * fps))
    return VideoInfo(
        path=path,
        total_frames=total_frames,
        fps=fps,
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_seconds=duration or total_frames / fps,
    )


def sample_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    max_frames: int = 12,
    strategy: Literal["uniform", "head", "every_n_seconds"] = "uniform",
    every_n_seconds: float = 1.0,
    info: VideoInfo | None = None,
) -> list[SampledFrame]:
    """Extract a small, explicit set of JPEGs with ffmpeg."""

    info = info or probe_video(video_path)
    plan = plan_frame_indices(
        info.total_frames,
        info.fps,
        max_frames=max_frames,
        strategy=strategy,
        every_n_seconds=every_n_seconds,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[SampledFrame] = []
    for order, frame in enumerate(plan):
        output_path = output_dir / f"frame-{order:03d}-at-{frame.timestamp_seconds:.3f}s.jpg"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{frame.timestamp_seconds:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        except FileNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("ffmpeg is required to extract video frames") from exc
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            raise RuntimeError(f"ffmpeg extraction failed: {exc.stderr.strip()}") from exc
        extracted.append(SampledFrame(frame.index, frame.timestamp_seconds, output_path))
    return extracted


def write_sampling_plan(
    path: Path,
    frames: list[SampledFrame],
    *,
    title: str = "Video frame sampling plan",
) -> tuple[Path, Path]:
    """Write JSON evidence and an SVG timeline for visual inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    **asdict(frame),
                    "output_path": str(frame.output_path) if frame.output_path else None,
                }
                for frame in frames
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    svg_path = path.with_suffix(".svg")
    maximum = max((frame.timestamp_seconds for frame in frames), default=1.0)
    maximum = max(maximum, 1e-9)
    markers = []
    for frame in frames:
        x = 70 + 650 * frame.timestamp_seconds / maximum
        markers.append(
            f'<line x1="{x:.2f}" y1="90" x2="{x:.2f}" y2="145" stroke="#3575b8" stroke-width="3"/>'
            f'<text x="{x:.2f}" y="170" text-anchor="middle" font-size="11">{frame.timestamp_seconds:.2f}s</text>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="210">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="30" y="38" font-size="24" font-weight="bold">{title}</text>'
        '<line x1="70" y1="120" x2="720" y2="120" stroke="#25364a" stroke-width="4"/>'
        + "".join(markers)
        + "</svg>"
    )
    svg_path.write_text(svg, encoding="utf-8")
    return path, svg_path
