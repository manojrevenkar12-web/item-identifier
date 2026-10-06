"""Typed configuration. Every run is driven by one YAML file validated here."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    root: Path = Path("data/stanford_cars")
    # Directory layout: root/{train,val,test}/<class_name>/*.jpg
    image_size: int = 224
    num_workers: int = 4
    batch_size: int = 32
    # Optional map class_name -> category (e.g. car model -> make). Enables per-category metrics.
    category_map: Path | None = None
    # Fraction of train images treated as "labelled" at start. <1.0 enables the active-learning pool.
    labelled_fraction: float = Field(1.0, gt=0, le=1)
    seed: int = 42


class ModelConfig(BaseModel):
    backbone: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained: bool = True
    drop_rate: float = 0.1
    # Freeze backbone for the first N epochs (linear probe warm-up), then fine-tune end to end.
    freeze_epochs: int = 1


class TrainConfig(BaseModel):
    epochs: int = 15
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_epochs: int = 1
    amp: bool = True
    early_stop_patience: int = 4
    grad_clip: float = 1.0
    out_dir: Path = Path("runs/latest")


class DecisionConfig(BaseModel):
    """How the product turns a calibrated probability into accept / abstain."""

    # Precision the business needs on *accepted* predictions (e.g. pricing is wrong < 5% of the time).
    target_precision: float = Field(0.95, gt=0, lt=1)
    # Floor so a sparse category never gets an absurdly low threshold.
    min_threshold: float = 0.30
    # Categories with fewer validation samples than this fall back to the global threshold.
    min_samples_per_category: int = 30


class GateConfig(BaseModel):
    """CI quality gates. A model that breaches any of these must not ship."""

    min_top1_accuracy: float = 0.80
    max_ece: float = 0.05
    min_coverage_at_target: float = 0.60
    max_p95_latency_ms: float = 150.0
    # Regression tolerance vs the baseline report (absolute points).
    max_top1_drop: float = 0.01
    max_ece_increase: float = 0.01
    max_category_top1_drop: float = 0.03


class Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    decision: DecisionConfig = DecisionConfig()
    gates: GateConfig = GateConfig()

    @field_validator("*", mode="before")
    @classmethod
    def _none_to_default(cls, v):
        return {} if v is None else v

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f) or {})

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
