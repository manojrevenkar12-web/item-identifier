import pytest

from itemid.config import Config
from itemid.synthetic import generate


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory) -> Config:
    root = tmp_path_factory.mktemp("data")
    generate(root, per_class={"train": 30, "val": 15, "test": 15}, size=48, seed=1)
    out = tmp_path_factory.mktemp("run")
    return Config.model_validate({
        "data": {"root": str(root), "image_size": 48, "batch_size": 32, "num_workers": 0,
                 "category_map": str(root / "category_map.json")},
        "model": {"backbone": "resnet18", "pretrained": False, "drop_rate": 0.0, "freeze_epochs": 0},
        "train": {"epochs": 3, "lr_head": 3e-3, "lr_backbone": 3e-3, "warmup_epochs": 0, "label_smoothing": 0.0,
                  "amp": False, "out_dir": str(out)},
        "decision": {"target_precision": 0.8, "min_samples_per_category": 10},
    })


@pytest.fixture(scope="session")
def trained(tiny_cfg):
    from itemid.train import train
    return train(tiny_cfg, version="test")
