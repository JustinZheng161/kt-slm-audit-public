#!/usr/bin/env python3
"""Build a non-submission contact sheet for visual QA of Nature-style KT figures."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]
OUT = ROOT / "figures" / "nature-kt-v31"
STEMS = [
    "figure-01-student-disjoint-baselines",
    "figure-02-training-window-sensitivity",
    "figure-03-primary-ablation",
    "figure-04-label-inversion-sensitivity",
    "figure-s01-probability-quality",
    "figure-s02-exploratory-budget",
    "figure-s03-bkt-diagnostics",
]


def main() -> None:
    thumb_w, thumb_h, margin, header = 1000, 500, 35, 80
    rows, cols = 4, 2
    sheet = Image.new("RGB", (cols * (thumb_w + margin) + margin, header + rows * (thumb_h + margin) + margin), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, 25), "Nature-style KT figures — visual QA contact sheet (not for submission)", fill="#20252B", font=font)
    for index, stem in enumerate(STEMS):
        image = Image.open(OUT / f"{stem}.png").convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        col, row = index % cols, index // cols
        x = margin + col * (thumb_w + margin)
        y = header + row * (thumb_h + margin)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline="#D9DEE3", width=1)
        draw.text((x, y + thumb_h + 5), stem, fill="#20252B", font=font)
    sheet.save(OUT / "contact-sheet-nature-kt-v31.png")


if __name__ == "__main__":
    main()
