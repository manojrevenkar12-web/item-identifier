"""Fine-tuning, then post-hoc calibration and threshold fitting on the validation split.

Two-phase schedule: a short linear-probe warm-up with the backbone frozen (so the random head does
not push large gradients into pretrained features), then end-to-end fine-tuning with a much lower
backbone learning rate than head learning rate.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .calibration import calibration_summary, fit_temperature
from .config import Config
from .data import DataBundle, build_data
from .infer import run_inference
from .model import ItemClassifier, ModelMeta, pick_device, save_checkpoint
from .selective import fit_thresholds

log = logging.getLogger("itemid.train")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train(cfg: Config, data: DataBundle | None = None, version: str | None = None) -> Path:
    _set_seed(cfg.data.seed)
    device = pick_device()
    data = data or build_data(cfg)
    out = Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.dump(out / "config.yaml")

    model = ItemClassifier(cfg.model.backbone, len(data.classes), cfg.model.pretrained,
                           cfg.model.drop_rate, cfg.data.image_size).to(device)
    params = [
        {"params": model.head.parameters(), "lr": cfg.train.lr_head},
        {"params": model.backbone.parameters(), "lr": cfg.train.lr_backbone},
    ]
    opt = torch.optim.AdamW(params, weight_decay=cfg.train.weight_decay)
    train_loader = data.loader(data.train, cfg.data.batch_size, True, cfg.data.num_workers)
    val_loader = data.loader(data.val, cfg.data.batch_size, False, cfg.data.num_workers)
    steps_per_epoch = len(train_loader)
    total = steps_per_epoch * cfg.train.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, steps_per_epoch * cfg.train.warmup_epochs, total))
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

    best_acc, bad_epochs, history = -1.0, 0, []
    log.info("device=%s classes=%d train=%d val=%d", device, len(data.classes), len(data.train), len(data.val))
    for epoch in range(cfg.train.epochs):
        frozen = epoch < cfg.model.freeze_epochs
        model.set_backbone_trainable(not frozen)
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)

        val = run_inference(model, val_loader, device, with_embeddings=False)
        val_loss = F.cross_entropy(val.logits, val.labels).item()
        val_acc = (val.logits.argmax(1) == val.labels).float().mean().item()
        row = {"epoch": epoch + 1, "train_loss": running / max(1, seen), "val_loss": val_loss,
               "val_top1": val_acc, "backbone_frozen": frozen, "seconds": round(time.time() - t0, 1)}
        history.append(row)
        log.info(json.dumps(row))
        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            torch.save(model.state_dict(), out / "best_weights.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.early_stop_patience:
                log.info("early stop at epoch %d", epoch + 1)
                break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(out / "best_weights.pt", map_location=device))
    meta = calibrate_and_package(model, data, cfg, device, version or time.strftime("%Y%m%d-%H%M%S"))
    path = out / "model.pt"
    save_checkpoint(path, model, meta)
    log.info("saved %s (T=%.3f, global threshold=%.3f)", path, meta.temperature, meta.thresholds["__global__"])
    return path


def calibrate_and_package(model, data: DataBundle, cfg: Config, device, version: str) -> ModelMeta:
    """Fit temperature + abstention thresholds on validation, write calibration.json, return serving meta."""
    val_loader = data.loader(data.val, cfg.data.batch_size, False, cfg.data.num_workers)
    val = run_inference(model, val_loader, device, with_embeddings=False)
    before = calibration_summary(val.logits, val.labels, 1.0)
    t = fit_temperature(val.logits, val.labels)
    after = calibration_summary(val.logits, val.labels, t)

    conf, pred = val.top1(t)
    correct = pred == val.labels.numpy()
    pred_cat = [data.category_of[data.classes[p]] for p in pred]
    thresholds = fit_thresholds(conf, correct, pred_cat, cfg.decision.target_precision,
                                cfg.decision.min_threshold, cfg.decision.min_samples_per_category)
    Path(cfg.train.out_dir, "calibration.json").write_text(json.dumps(
        {"split": "val", "before": {k: v for k, v in before.items() if k != "reliability"},
         "after": {k: v for k, v in after.items() if k != "reliability"}, "thresholds": thresholds},
        indent=2))
    return ModelMeta(backbone=cfg.model.backbone, classes=data.classes, category_of=data.category_of,
                     image_size=cfg.data.image_size, temperature=t, thresholds=thresholds,
                     target_precision=cfg.decision.target_precision, version=version)
