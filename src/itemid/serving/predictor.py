"""Inference engine used by the API. Runs the PyTorch checkpoint or its ONNX export (same outputs)."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from ..data import eval_transform
from ..model import ModelMeta, load_checkpoint

MAX_PIXELS = 40_000_000  # reject decompression bombs
MIN_SIDE = 32


class InvalidImage(ValueError):
    pass


def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise InvalidImage("empty upload")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise InvalidImage("image too large")
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImage(f"not a decodable image: {e}") from e
    if min(img.size) < MIN_SIDE:
        raise InvalidImage(f"image too small (min side {MIN_SIDE}px)")
    return img


@dataclass
class Prediction:
    label: str
    category: str
    confidence: float
    threshold: float
    decision: str                 # "accept" | "abstain"
    alternatives: list[dict]      # top-k with calibrated probabilities
    margin: float                 # top1 - top2 probability; used to rank the labelling queue
    model_version: str
    latency_ms: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Predictor:
    def __init__(self, checkpoint: str | Path, onnx_path: str | Path | None = None, top_k: int = 5,
                 device: str = "cpu"):
        self.model, self.meta = load_checkpoint(checkpoint, device)
        self.device = device
        self.top_k = min(top_k, len(self.meta.classes))
        self.transform = eval_transform(self.meta.image_size)
        self.session = None
        if onnx_path:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    @property
    def meta_info(self) -> ModelMeta:
        return self.meta

    def _logits(self, x: torch.Tensor) -> np.ndarray:
        if self.session is not None:
            return self.session.run(None, {"image": x.numpy()})[0]
        # inference_mode is thread-local, so it is set per call (the API runs this in a worker thread).
        with torch.inference_mode():
            return self.model(x.to(self.device)).float().cpu().numpy()

    def predict(self, img: Image.Image) -> Prediction:
        t0 = time.perf_counter()
        x = self.transform(img).unsqueeze(0)
        logits = self._logits(x)[0] / self.meta.temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-p)[: self.top_k]
        top = order[0]
        label = self.meta.classes[top]
        conf = float(p[top])
        thr = self.meta.threshold_for(label)
        second = float(p[order[1]]) if len(order) > 1 else 0.0
        return Prediction(
            label=label,
            category=self.meta.category_of.get(label, label),
            confidence=round(conf, 5),
            threshold=round(thr, 5),
            decision="accept" if conf >= thr else "abstain",
            alternatives=[{"label": self.meta.classes[i], "probability": round(float(p[i]), 5)} for i in order],
            margin=round(conf - second, 5),
            model_version=self.meta.version,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
