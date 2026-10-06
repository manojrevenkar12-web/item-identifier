"""A small generated fine-grained dataset for tests, CI and offline demos (no download needed).

Categories are shapes; classes within a category differ only by a subtle hue shift and a small mark
position -- the same "look-alike items in one category" structure as a real resale catalog. It exists
to exercise the pipeline end to end; it says nothing about real-world accuracy.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SHAPES = ("circle", "square", "triangle")
VARIANTS = (("red", (200, 60, 60), 0), ("crimson", (175, 40, 75), 0), ("orange", (210, 120, 40), 1),
            ("teal", (40, 150, 150), 1))


def _draw(shape: str, color, mark: int, rng: random.Random, size: int) -> Image.Image:
    bg = tuple(rng.randint(180, 255) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    s = rng.uniform(0.45, 0.75) * size
    cx, cy = rng.uniform(s / 2, size - s / 2), rng.uniform(s / 2, size - s / 2)
    jitter = tuple(max(0, min(255, c + rng.randint(-12, 12))) for c in color)
    box = (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
    if shape == "circle":
        d.ellipse(box, fill=jitter)
    elif shape == "square":
        d.rectangle(box, fill=jitter)
    else:
        d.polygon([(cx, cy - s / 2), (cx - s / 2, cy + s / 2), (cx + s / 2, cy + s / 2)], fill=jitter)
    r = s * 0.08
    mx = cx - s * 0.18 if mark == 0 else cx + s * 0.18
    d.ellipse((mx - r, cy - r, mx + r, cy + r), fill=(20, 20, 20))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    return img


def generate(root: str | Path, per_class: dict[str, int] | None = None, size: int = 72, seed: int = 0) -> Path:
    per_class = per_class or {"train": 60, "val": 25, "test": 25}
    root = Path(root)
    rng = random.Random(seed)
    cmap = {}
    for split, n in per_class.items():
        for shape in SHAPES:
            for name, color, mark in VARIANTS:
                cls = f"{shape}_{name}"
                cmap[cls] = shape
                d = root / split / cls
                d.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    _draw(shape, color, mark, rng, size).save(d / f"{i:04d}.png")
    (root / "category_map.json").write_text(json.dumps(cmap, indent=2))
    return root
