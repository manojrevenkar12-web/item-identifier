"""Download Stanford Cars (196 classes) from the Hugging Face Hub and write the folder layout itemid expects.

    pip install datasets
    python scripts/prepare_stanford_cars.py --out data/stanford_cars [--robustness]

Produces train/ val/ test/ (val = stratified 15% of the official train split, so test stays untouched)
and category_map.json (car model -> make). With --robustness it also writes the dataset's corrupted
copies of the test split (motion blur, JPEG compression, noise, ...) as test_<corruption>/ so the
model can be evaluated on degraded, phone-camera-like photos.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")
CORRUPTIONS = ("contrast", "gaussian_noise", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
               "spatter")


def make_of(name: str) -> str:
    for m in MULTI_WORD_MAKES:
        if name.startswith(m + " "):
            return m
    return name.split(" ")[0]


def safe(name: str) -> str:
    return re.sub(r"[/\\:*?\"<>|]", "-", name).strip()


def write_split(ds, names, out: Path, indices=None) -> int:
    n = 0
    for i in (indices if indices is not None else range(len(ds))):
        row = ds[i]
        d = out / safe(names[row["label"]])
        d.mkdir(parents=True, exist_ok=True)
        row["image"].convert("RGB").save(d / f"{i:06d}.jpg", quality=95)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/stanford_cars")
    ap.add_argument("--repo", default="tanganke/stanford_cars")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    out = Path(args.out)
    train = load_dataset(args.repo, split="train")
    names = train.features["label"].names
    by_cls: dict[int, list[int]] = {}
    for i, y in enumerate(train["label"]):
        by_cls.setdefault(y, []).append(i)
    rng = random.Random(args.seed)
    tr_idx, va_idx = [], []
    for idxs in by_cls.values():
        rng.shuffle(idxs)
        k = max(1, round(len(idxs) * args.val_fraction))
        va_idx += idxs[:k]
        tr_idx += idxs[k:]
    print("train", write_split(train, names, out / "train", sorted(tr_idx)))
    print("val", write_split(train, names, out / "val", sorted(va_idx)))
    print("test", write_split(load_dataset(args.repo, split="test"), names, out / "test"))
    if args.robustness:
        for c in CORRUPTIONS:
            print(c, write_split(load_dataset(args.repo, split=c), names, out / f"test_{c}"))
    (out / "category_map.json").write_text(json.dumps({safe(n): make_of(n) for n in names}, indent=2))
    print("categories:", len(set(make_of(n) for n in names)))


if __name__ == "__main__":
    main()
