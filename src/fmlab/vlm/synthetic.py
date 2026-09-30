from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schema import QAItem


@dataclass
class SyntheticSample:
    id: str
    task: str
    image_path: Path
    qa: list[QAItem]
    metadata: dict[str, Any]
    ocr_text: str = ""

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        image_path = self.image_path
        if relative_to is not None:
            try:
                image_path = image_path.relative_to(relative_to)
            except ValueError:
                pass
        return {
            "id": self.id,
            "task": self.task,
            "image_path": str(image_path),
            "qa": [item.to_dict() for item in self.qa],
            "metadata": self.metadata,
            "ocr_text": self.ocr_text,
        }


def generate_document_dataset(
    output_dir: Path,
    *,
    count: int = 8,
    seed: int = 7,
) -> list[SyntheticSample]:
    """Generate deterministic invoices/receipts and exact field-QA labels."""

    if count < 1:
        raise ValueError("count must be positive")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    samples: list[SyntheticSample] = []
    for index in range(count):
        if index % 2:
            sample = _make_receipt(image_dir, index, rng)
        else:
            sample = _make_invoice(image_dir, index, rng)
        samples.append(sample)
    _write_jsonl(
        output_dir / "manifest.jsonl",
        (sample.to_dict(relative_to=output_dir) for sample in samples),
    )
    _write_dataset_card(output_dir, "Synthetic document VQA", samples)
    return samples


def generate_chart_dataset(
    output_dir: Path,
    *,
    count: int = 8,
    seed: int = 11,
) -> list[SyntheticSample]:
    """Generate bar/line charts with labels derived directly from source values."""

    if count < 1:
        raise ValueError("count must be positive")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    samples: list[SyntheticSample] = []
    for index in range(count):
        categories = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        values = {category: rng.randint(12, 92) for category in categories}
        # Avoid ambiguous argmax/argmin labels.
        while len(set(values.values())) != len(values):
            values = {category: rng.randint(12, 92) for category in categories}
        chart_type = "bar" if index % 2 == 0 else "line"
        image_path = image_dir / f"chart-{index:03d}.png"
        _draw_chart(image_path, chart_type, values, title=f"Quarterly metric {index + 1}")
        max_key = max(values, key=values.get)
        min_key = min(values, key=values.get)
        probe = categories[index % len(categories)]
        max_value = values[max_key]
        min_value = values[min_key]
        qa = [
            QAItem(
                id=f"chart-{index:03d}-argmax",
                question="Which category has the highest value?",
                answer=max_key,
                metadata={"operation": "argmax"},
            ),
            QAItem(
                id=f"chart-{index:03d}-argmin",
                question="Which category has the lowest value?",
                answer=min_key,
                metadata={"operation": "argmin"},
            ),
            QAItem(
                id=f"chart-{index:03d}-lookup",
                question=f"What is the value for {probe}? Answer with a number.",
                answer=str(values[probe]),
                answer_type="integer",
                metadata={"operation": "lookup", "category": probe},
            ),
            QAItem(
                id=f"chart-{index:03d}-difference",
                question="What is the difference between the highest and lowest values?",
                answer=str(max_value - min_value),
                answer_type="integer",
                metadata={"operation": "difference"},
            ),
            QAItem(
                id=f"chart-{index:03d}-sum",
                question="What is the sum of all category values?",
                answer=str(sum(values.values())),
                answer_type="integer",
                metadata={"operation": "sum"},
            ),
        ]
        samples.append(
            SyntheticSample(
                id=f"chart-{index:03d}",
                task="chart_qa",
                image_path=image_path,
                qa=qa,
                metadata={
                    "chart_type": chart_type,
                    "values": values,
                    "title": f"Quarterly metric {index + 1}",
                },
            )
        )
    _write_jsonl(
        output_dir / "manifest.jsonl",
        (sample.to_dict(relative_to=output_dir) for sample in samples),
    )
    _write_dataset_card(output_dir, "Synthetic chart QA", samples)
    return samples


def read_manifest(path: Path) -> list[SyntheticSample]:
    samples: list[SyntheticSample] = []
    root = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        image_path = Path(item["image_path"])
        if not image_path.is_absolute():
            image_path = root / image_path
        samples.append(
            SyntheticSample(
                id=item["id"],
                task=item["task"],
                image_path=image_path,
                qa=[QAItem(**qa) for qa in item["qa"]],
                metadata=item.get("metadata", {}),
                ocr_text=item.get("ocr_text", ""),
            )
        )
    return samples


def _make_invoice(image_dir: Path, index: int, rng: random.Random) -> SyntheticSample:
    vendor = rng.choice(["Northwind Labs", "Contoso Tools", "Blue River Data"])
    invoice_id = f"INV-{2026 + index:04d}-{rng.randint(1000, 9999)}"
    date = f"2026-{(index % 12) + 1:02d}-{rng.randint(1, 28):02d}"
    quantities = [rng.randint(1, 5) for _ in range(3)]
    prices = [rng.randint(8, 60) for _ in range(3)]
    subtotal = sum(qty * price for qty, price in zip(quantities, prices, strict=True))
    tax = round(subtotal * 0.1, 2)
    total = round(subtotal + tax, 2)
    fields = {
        "invoice_id": invoice_id,
        "date": date,
        "vendor": vendor,
        "subtotal": f"{subtotal:.2f}",
        "tax": f"{tax:.2f}",
        "total": f"{total:.2f}",
    }
    rows = [(f"Item {chr(65 + i)}", quantities[i], prices[i]) for i in range(3)]
    image_path = image_dir / f"document-{index:03d}.png"
    boxes, ocr_text = _draw_invoice(image_path, fields, rows)
    qa = _field_questions(f"document-{index:03d}", fields, boxes)
    return SyntheticSample(
        id=f"document-{index:03d}",
        task="document_vqa",
        image_path=image_path,
        qa=qa,
        metadata={"document_type": "invoice", "fields": fields, "field_boxes": boxes},
        ocr_text=ocr_text,
    )


def _make_receipt(image_dir: Path, index: int, rng: random.Random) -> SyntheticSample:
    vendor = rng.choice(["Acorn Market", "Tiny Cafe", "Orbit Books"])
    receipt_id = f"R-{rng.randint(100000, 999999)}"
    date = f"2026-{(index % 12) + 1:02d}-{rng.randint(1, 28):02d}"
    item_count = rng.randint(2, 5)
    item_prices = [rng.randint(250, 2900) / 100 for _ in range(item_count)]
    subtotal = round(sum(item_prices), 2)
    tax = round(subtotal * 0.08, 2)
    total = round(subtotal + tax, 2)
    fields = {
        "receipt_id": receipt_id,
        "date": date,
        "vendor": vendor,
        "subtotal": f"{subtotal:.2f}",
        "tax": f"{tax:.2f}",
        "total": f"{total:.2f}",
    }
    rows = [(f"Product {index + 1}", price) for index, price in enumerate(item_prices)]
    image_path = image_dir / f"document-{index:03d}.png"
    boxes, ocr_text = _draw_receipt(image_path, fields, rows)
    qa = _field_questions(f"document-{index:03d}", fields, boxes)
    return SyntheticSample(
        id=f"document-{index:03d}",
        task="document_vqa",
        image_path=image_path,
        qa=qa,
        metadata={"document_type": "receipt", "fields": fields, "field_boxes": boxes},
        ocr_text=ocr_text,
    )


def _field_questions(
    prefix: str, fields: dict[str, str], boxes: dict[str, list[int]]
) -> list[QAItem]:
    labels = {
        "invoice_id": "What is the invoice ID?",
        "receipt_id": "What is the receipt ID?",
        "date": "What is the date?",
        "vendor": "Who is the vendor?",
        "subtotal": "What is the subtotal? Answer with a number.",
        "tax": "What is the tax? Answer with a number.",
        "total": "What is the total? Answer with a number.",
    }
    result = []
    for key, value in fields.items():
        result.append(
            QAItem(
                id=f"{prefix}-{key}",
                question=labels[key],
                answer=value,
                answer_type="number" if key in {"subtotal", "tax", "total"} else "text",
                metadata={"field": key, "bbox_xyxy": boxes[key]},
            )
        )
    return result


def _draw_invoice(
    path: Path, fields: dict[str, str], rows: list[tuple[str, int, int]]
) -> tuple[dict[str, list[int]], str]:
    Image, ImageDraw, _ = _pillow()
    image = Image.new("RGB", (900, 1120), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(44)
    body_font = _font(26)
    small_font = _font(22)
    draw.rectangle((35, 35, 865, 1085), outline="#19324d", width=3)
    draw.text((65, 70), "INVOICE", fill="#19324d", font=title_font)
    boxes: dict[str, list[int]] = {}
    y = 150
    for key in ("invoice_id", "date", "vendor"):
        label = key.replace("_", " ").title()
        line = f"{label}: {fields[key]}"
        draw.text((65, y), line, fill="black", font=body_font)
        boxes[key] = [65, y, 835, y + 36]
        y += 52
    y += 40
    draw.rectangle((65, y, 835, y + 48), fill="#dcebf7")
    draw.text((80, y + 8), "Description", fill="black", font=small_font)
    draw.text((520, y + 8), "Qty", fill="black", font=small_font)
    draw.text((680, y + 8), "Price", fill="black", font=small_font)
    y += 58
    lines = [
        f"Invoice ID: {fields['invoice_id']}",
        f"Date: {fields['date']}",
        f"Vendor: {fields['vendor']}",
    ]
    for description, quantity, price in rows:
        draw.text((80, y), description, fill="black", font=small_font)
        draw.text((530, y), str(quantity), fill="black", font=small_font)
        draw.text((680, y), f"{price:.2f}", fill="black", font=small_font)
        lines.append(f"{description} | {quantity} | {price:.2f}")
        draw.line((65, y + 38, 835, y + 38), fill="#d0d0d0", width=1)
        y += 52
    y += 55
    for key in ("subtotal", "tax", "total"):
        label = key.title()
        draw.text((500, y), f"{label}: {fields[key]}", fill="black", font=body_font)
        boxes[key] = [490, y - 2, 835, y + 38]
        lines.append(f"{label}: {fields[key]}")
        y += 52
    image.save(path)
    return boxes, "\n".join(lines)


def _draw_receipt(
    path: Path, fields: dict[str, str], rows: list[tuple[str, float]]
) -> tuple[dict[str, list[int]], str]:
    Image, ImageDraw, _ = _pillow()
    image = Image.new("RGB", (680, 1000), "#fdfcf6")
    draw = ImageDraw.Draw(image)
    title_font = _font(38)
    body_font = _font(25)
    draw.rectangle((45, 35, 635, 960), outline="#555555", width=2)
    boxes: dict[str, list[int]] = {}
    y = 70
    draw.text((80, y), fields["vendor"], fill="black", font=title_font)
    boxes["vendor"] = [75, y - 4, 610, y + 46]
    y += 75
    for key in ("receipt_id", "date"):
        label = key.replace("_", " ").title()
        draw.text((80, y), f"{label}: {fields[key]}", fill="black", font=body_font)
        boxes[key] = [75, y - 2, 610, y + 34]
        y += 46
    draw.line((75, y, 610, y), fill="#555555", width=2)
    y += 30
    lines = [
        f"Vendor: {fields['vendor']}",
        f"Receipt ID: {fields['receipt_id']}",
        f"Date: {fields['date']}",
    ]
    for item, price in rows:
        draw.text((80, y), item, fill="black", font=body_font)
        draw.text((480, y), f"{price:.2f}", fill="black", font=body_font)
        lines.append(f"{item}: {price:.2f}")
        y += 45
    y += 30
    draw.line((75, y, 610, y), fill="#555555", width=2)
    y += 25
    for key in ("subtotal", "tax", "total"):
        label = key.title()
        draw.text((265, y), f"{label}: {fields[key]}", fill="black", font=body_font)
        boxes[key] = [255, y - 2, 610, y + 34]
        lines.append(f"{label}: {fields[key]}")
        y += 48
    image.save(path)
    return boxes, "\n".join(lines)


def _draw_chart(path: Path, chart_type: str, values: dict[str, int], *, title: str) -> None:
    Image, ImageDraw, _ = _pillow()
    width, height = 920, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(34)
    body_font = _font(21)
    draw.text((55, 28), title, fill="#182b49", font=title_font)
    left, top, right, bottom = 90, 105, 865, 520
    draw.line((left, top, left, bottom), fill="black", width=3)
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    max_scale = 100
    for tick in range(0, max_scale + 1, 20):
        y = bottom - int((bottom - top) * tick / max_scale)
        draw.line((left - 7, y, right, y), fill="#e0e4e8", width=1)
        draw.text((42, y - 12), str(tick), fill="#333333", font=body_font)
    categories = list(values)
    slot = (right - left) / len(categories)
    points: list[tuple[int, int]] = []
    palette = ["#3575b8", "#e47d31", "#4a9d60", "#9a65b5", "#c84d4d"]
    for index, category in enumerate(categories):
        x = int(left + slot * (index + 0.5))
        value = values[category]
        y = bottom - int((bottom - top) * value / max_scale)
        if chart_type == "bar":
            half = int(slot * 0.28)
            draw.rectangle((x - half, y, x + half, bottom), fill=palette[index])
        points.append((x, y))
        draw.text((x - 28, bottom + 18), category, fill="#222222", font=body_font)
        draw.text((x - 12, y - 30), str(value), fill="#222222", font=body_font)
    if chart_type == "line":
        draw.line(points, fill="#3575b8", width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#e47d31", outline="white", width=2)
    image.save(path)


def _font(size: int) -> Any:
    _, _, ImageFont = _pillow()
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - dependency message
        raise RuntimeError(
            "Synthetic visual data requires Pillow: pip install -e '.[vlm]'"
        ) from exc
    return Image, ImageDraw, ImageFont


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_dataset_card(output_dir: Path, title: str, samples: list[SyntheticSample]) -> None:
    questions = sum(len(sample.qa) for sample in samples)
    text = (
        f"# {title}\n\n"
        f"Seeded toy data generated locally. Samples: **{len(samples)}**; "
        f"question/answer pairs: **{questions}**.\n\n"
        "Labels come from the source values used to render each image, so they are deterministic. "
        "This dataset is for pipeline study, not a real-world quality claim.\n"
    )
    (output_dir / "README.md").write_text(text, encoding="utf-8")
