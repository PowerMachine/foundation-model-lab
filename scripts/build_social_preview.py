#!/usr/bin/env python3
"""Build the deterministic GitHub/Open Graph preview card."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "github-social-preview.png"
WIDTH, HEIGHT = 1280, 640


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def build() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#07111e")
    pixels = image.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            distance = ((x - 1120) ** 2 + (y - 70) ** 2) ** 0.5
            glow = max(0.0, 1.0 - distance / 760.0)
            base = (7 + int(15 * glow), 17 + int(24 * glow), 30 + int(31 * glow))
            pixels[x, y] = base

    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((32, 32, 1248, 608), radius=28, outline="#2c4058", width=2)
    draw.rounded_rectangle((74, 67, 126, 119), radius=14, fill="#54dbc3")
    draw.text((88, 82), "RM", font=_font(15, bold=True), fill="#07111e")
    draw.text((146, 78), "RELIABLE SYSTEMS LAB", font=_font(20, bold=True), fill="#dbe8f5")
    draw.text((76, 171), "Reliable multimodal", font=_font(61, bold=True), fill="#f4f7fb")
    draw.text((76, 238), "agent systems.", font=_font(61, bold=True), fill="#f4f7fb")
    draw.text(
        (79, 327),
        "Post-training  ·  Agent evaluation  ·  Distributed correctness  ·  Inference",
        font=_font(21),
        fill="#9fb0c5",
    )

    cards = (
        ("8.77B", "actual VLM load", "#fb8da1"),
        ("24", "agent episodes", "#bda6ff"),
        ("81", "systems trials", "#f7c56a"),
        ("11/11", "DDP gates", "#6eb5ff"),
    )
    card_width = 264
    for index, (value, label, accent) in enumerate(cards):
        left = 76 + index * 285
        draw.rounded_rectangle(
            (left, 402, left + card_width, 535),
            radius=15,
            fill="#101f31",
            outline="#2c4058",
            width=2,
        )
        draw.rectangle((left, 402, left + 7, 535), fill=accent)
        draw.text((left + 28, 425), value, font=_font(36, bold=True), fill=accent)
        draw.text((left + 29, 481), label, font=_font(16), fill="#9fb0c5")

    draw.text(
        (78, 566),
        "Every claim resolves to code, machine-readable evidence, and an explicit boundary.",
        font=_font(16),
        fill="#71849d",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    return OUTPUT


if __name__ == "__main__":
    print(build())
