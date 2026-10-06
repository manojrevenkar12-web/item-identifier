"""Model definition and the checkpoint contract shared by training, evaluation and serving."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import timm
import torch
from torch import nn


class ItemClassifier(nn.Module):
    """A timm backbone (DINOv2 ViT, ConvNeXt, ResNet, ...) with a fresh linear head.

    `embed()` exposes the pooled features; they are reused for nearest-neighbour
    explanations and for diversity-aware active learning.
    """

    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.0,
                 image_size: int | None = None):
        super().__init__()
        kwargs = {"num_classes": 0, "pretrained": pretrained}
        if image_size and ("vit" in backbone or "dinov2" in backbone):
            kwargs["img_size"] = image_size  # DINOv2 defaults to 518; interpolate pos-embeddings to our size
        self.backbone = timm.create_model(backbone, **kwargs)
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.backbone.num_features, num_classes)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(x)))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable


@dataclass
class ModelMeta:
    """Everything needed to serve the model correctly, stored next to the weights."""

    backbone: str
    classes: list[str]
    category_of: dict[str, str]
    image_size: int
    temperature: float = 1.0
    # category -> confidence threshold; "__global__" is the fallback.
    thresholds: dict[str, float] = field(default_factory=lambda: {"__global__": 0.5})
    target_precision: float = 0.95
    version: str = "dev"

    def threshold_for(self, class_name: str) -> float:
        cat = self.category_of.get(class_name, class_name)
        return self.thresholds.get(cat, self.thresholds["__global__"])


def save_checkpoint(path: str | Path, model: ItemClassifier, meta: ModelMeta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": asdict(meta)}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[ItemClassifier, ModelMeta]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ModelMeta(**ckpt["meta"])
    model = ItemClassifier(meta.backbone, len(meta.classes), pretrained=False, image_size=meta.image_size)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
