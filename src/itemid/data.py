"""Data loading.

Expected layout (what `scripts/prepare_stanford_cars.py` produces, and what any catalog export can match):

    root/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
      test/<class_name>/*.jpg
      category_map.json          # optional: {"<class_name>": "<category>"}

Class indices are fixed by the sorted train class list and stored in every checkpoint,
so a served model can never silently disagree with the training label space.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def train_transform(size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageFolderSplit(Dataset):
    """Image folder with an externally fixed class list (unknown classes are an error, not a silent skip)."""

    def __init__(self, split_dir: Path, classes: list[str], transform=None):
        self.split_dir = Path(split_dir)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"split directory not found: {self.split_dir}")
        for cls_dir in sorted(p for p in self.split_dir.iterdir() if p.is_dir()):
            if cls_dir.name not in self.class_to_idx:
                raise ValueError(f"class '{cls_dir.name}' in {self.split_dir} is not in the training label space")
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in IMG_EXTS:
                    self.samples.append((f, self.class_to_idx[cls_dir.name]))
        if not self.samples:
            raise ValueError(f"no images found under {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), label

    @property
    def targets(self) -> list[int]:
        return [y for _, y in self.samples]


def discover_classes(root: Path) -> list[str]:
    train = Path(root) / "train"
    classes = sorted(p.name for p in train.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError(f"need at least 2 classes under {train}")
    return classes


def load_category_map(path: Path | None, classes: list[str]) -> dict[str, str]:
    """class_name -> category. Classes missing from the map become their own category."""
    mapping: dict[str, str] = {}
    if path and Path(path).exists():
        mapping = json.loads(Path(path).read_text())
    return {c: mapping.get(c, c) for c in classes}


def stratified_subset(targets: list[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split indices into (labelled, unlabelled pool), keeping at least one labelled example per class."""
    if fraction >= 1.0:
        return list(range(len(targets))), []
    rng = random.Random(seed)
    by_cls: dict[int, list[int]] = {}
    for i, y in enumerate(targets):
        by_cls.setdefault(y, []).append(i)
    labelled, pool = [], []
    for idxs in by_cls.values():
        rng.shuffle(idxs)
        k = max(1, round(len(idxs) * fraction))
        labelled += idxs[:k]
        pool += idxs[k:]
    return sorted(labelled), sorted(pool)


@dataclass
class DataBundle:
    classes: list[str]
    category_of: dict[str, str]
    train: Dataset
    val: Dataset
    test: Dataset | None
    labelled_idx: list[int]
    pool_idx: list[int]

    def loader(self, ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(), drop_last=False,
                          persistent_workers=num_workers > 0)


def build_data(cfg, labelled_idx: list[int] | None = None) -> DataBundle:
    root = Path(cfg.data.root)
    classes = discover_classes(root)
    size = cfg.data.image_size
    train_full = ImageFolderSplit(root / "train", classes, train_transform(size))
    val = ImageFolderSplit(root / "val", classes, eval_transform(size))
    test = ImageFolderSplit(root / "test", classes, eval_transform(size)) if (root / "test").is_dir() else None
    if labelled_idx is None:
        labelled_idx, pool_idx = stratified_subset(train_full.targets, cfg.data.labelled_fraction, cfg.data.seed)
    else:
        s = set(labelled_idx)
        pool_idx = [i for i in range(len(train_full)) if i not in s]
    return DataBundle(
        classes=classes,
        category_of=load_category_map(cfg.data.category_map, classes),
        train=Subset(train_full, labelled_idx) if pool_idx else train_full,
        val=val,
        test=test,
        labelled_idx=labelled_idx,
        pool_idx=pool_idx,
    )
