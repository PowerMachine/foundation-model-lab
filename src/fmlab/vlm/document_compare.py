from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fmlab.artifacts import ExperimentResult

from .inference import Qwen3VLRunner
from .metrics import exact_match, numeric_match, token_f1
from .reporting import write_evaluation_html, write_rows, write_score_bars
from .schema import QAItem
from .synthetic import SyntheticSample


class OCRBackend(Protocol):
    name: str

    def extract(self, image_path: Path, sample: SyntheticSample | None = None) -> str: ...


class TextQuestionAnswerer(Protocol):
    name: str

    def answer(self, question: QAItem, ocr_text: str) -> str: ...


@dataclass
class SidecarOCR:
    """Offline OCR stand-in using ground truth, optionally with seeded corruption."""

    character_error_rate: float = 0.0
    seed: int = 0
    name: str = "sidecar-ocr-simulator"

    def extract(self, image_path: Path, sample: SyntheticSample | None = None) -> str:
        if sample is None:
            sidecar = image_path.with_suffix(".txt")
            if not sidecar.is_file():
                raise ValueError("SidecarOCR requires a sample or matching .txt file")
            text = sidecar.read_text(encoding="utf-8")
        else:
            text = sample.ocr_text
        if not 0 <= self.character_error_rate <= 1:
            raise ValueError("character_error_rate must be in [0, 1]")
        if self.character_error_rate == 0:
            return text
        digest = hashlib.sha256(f"{self.seed}:{image_path}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        output = []
        for character in text:
            if character.isalnum() and rng.random() < self.character_error_rate:
                output.append(rng.choice(alphabet))
            else:
                output.append(character)
        return "".join(output)


@dataclass
class TesseractOCR:
    language: str = "eng"
    config: str = "--psm 6"
    name: str = "tesseract"

    def extract(self, image_path: Path, sample: SyntheticSample | None = None) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError("TesseractOCR requires pillow and pytesseract") from exc
        return str(
            pytesseract.image_to_string(
                Image.open(image_path), lang=self.language, config=self.config
            )
        )


@dataclass
class RegexFieldAnswerer:
    """Transparent baseline for OCR text; deliberately not an LLM."""

    name: str = "regex-field-baseline"

    def answer(self, question: QAItem, ocr_text: str) -> str:
        field = str(question.metadata.get("field", ""))
        aliases = {
            "invoice_id": ["invoice id"],
            "receipt_id": ["receipt id"],
            "date": ["date"],
            "vendor": ["vendor"],
            "subtotal": ["subtotal"],
            "tax": ["tax"],
            "total": ["total"],
        }.get(field, [field.replace("_", " ")])
        for line in ocr_text.splitlines():
            for alias in aliases:
                match = re.match(rf"\s*{re.escape(alias)}\s*:\s*(.+?)\s*$", line, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        return "not found"


def run_document_comparison(
    samples: list[SyntheticSample],
    runner: Qwen3VLRunner,
    output_dir: Path,
    *,
    ocr_backend: OCRBackend | None = None,
    text_answerer: TextQuestionAnswerer | None = None,
) -> ExperimentResult:
    """Compare OCR→text extraction with direct image→VLM answering."""

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_backend = ocr_backend or SidecarOCR(character_error_rate=0.04, seed=17)
    text_answerer = text_answerer or RegexFieldAnswerer()
    rows: list[dict[str, Any]] = []
    method_scores: dict[str, list[float]] = {"ocr_pipeline": [], "direct_vlm": []}

    for sample in samples:
        ocr_text = ocr_backend.extract(sample.image_path, sample)
        for qa in sample.qa:
            ocr_prediction = text_answerer.answer(qa, ocr_text)
            vlm_result = runner.infer(
                qa.question,
                [sample.image_path],
                context={
                    "expected_answer": qa.answer,
                    "fields": sample.metadata.get("fields", {}),
                },
            )
            for method, prediction, backend, latency, simulated in (
                (
                    "ocr_pipeline",
                    ocr_prediction,
                    f"{ocr_backend.name}+{text_answerer.name}",
                    0.0,
                    True,
                ),
                (
                    "direct_vlm",
                    vlm_result.text,
                    vlm_result.backend,
                    vlm_result.latency_seconds,
                    vlm_result.simulated,
                ),
            ):
                exact = (
                    numeric_match(prediction, qa.answer)
                    if qa.answer_type in {"number", "integer"}
                    else exact_match(prediction, qa.answer)
                )
                method_scores[method].append(exact)
                rows.append(
                    {
                        "sample": sample.id,
                        "image_path": str(sample.image_path),
                        "question": qa.question,
                        "reference": qa.answer,
                        "method": method,
                        "backend": backend,
                        "prediction": prediction,
                        "exact_match": exact,
                        "token_f1": token_f1(prediction, qa.answer),
                        "latency_seconds": round(latency, 6),
                        "simulated": simulated,
                    }
                )

    accuracy = {
        method: sum(values) / len(values) if values else 0.0
        for method, values in method_scores.items()
    }
    result_rows = write_rows(output_dir / "predictions.jsonl", rows)
    scores_path = write_score_bars(
        output_dir / "method_accuracy.svg", accuracy, title="Document QA exact match"
    )
    report_path = write_evaluation_html(
        output_dir / "report.html",
        title="OCR pipeline vs direct VLM",
        summary={
            "samples": len(samples),
            "questions": sum(len(sample.qa) for sample in samples),
            **{f"{name}_accuracy": value for name, value in accuracy.items()},
        },
        rows=rows,
        score_chart=scores_path,
        notice=(
            "Rows marked simulated are plumbing checks, not model benchmarks. "
            "Run mode=local and a real OCR backend for quality conclusions."
        ),
    )
    experiment = ExperimentResult(
        experiment="vlm_document_comparison",
        status="completed",
        started_at=started,
        finished_at=time.time(),
        metrics={"accuracy": accuracy, "rows": len(rows)},
        parameters={
            "ocr_backend": ocr_backend.name,
            "text_answerer": text_answerer.name,
            "vlm_mode": runner.config.mode,
            "ocr_character_error_rate": getattr(ocr_backend, "character_error_rate", None),
        },
        artifacts=[str(result_rows), str(scores_path), str(report_path)],
        notes=["Offline direct-VLM outputs use deterministic expected-answer routing."]
        if runner.config.mode == "offline"
        else [],
    )
    result_path = experiment.write(output_dir)
    experiment.artifacts.append(str(result_path))
    return experiment
