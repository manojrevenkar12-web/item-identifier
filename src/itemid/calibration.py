"""Confidence calibration.

A softmax score is not a probability until it is calibrated. We fit a single temperature on the
validation split (Guo et al., 2017) because it cannot change the argmax -- accuracy is untouched --
and, with one parameter, it cannot overfit a few thousand validation images.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Minimise validation NLL over T > 0 (optimised in log space so T stays positive)."""
    logits = logits.detach().float()
    labels = labels.detach().long()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().item())
    if not np.isfinite(t) or t <= 0:
        return 1.0
    return float(np.clip(t, 0.05, 20.0))


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width-bin ECE: sum over bins of |accuracy - mean confidence| weighted by bin mass."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def reliability_bins(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        out.append({
            "lo": float(lo), "hi": float(hi), "count": int(mask.sum()),
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
            "confidence": float(confidences[mask].mean()) if mask.any() else None,
        })
    return out


def calibration_summary(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict:
    probs = F.softmax(logits.float() / temperature, dim=1)
    conf, pred = probs.max(1)
    correct = (pred == labels).numpy()
    conf_np = conf.numpy()
    onehot = F.one_hot(labels, probs.shape[1]).float()
    return {
        "temperature": temperature,
        "nll": float(F.cross_entropy(logits.float() / temperature, labels).item()),
        "ece": expected_calibration_error(conf_np, correct),
        "brier": float(((probs - onehot) ** 2).sum(1).mean().item()),
        "mean_confidence": float(conf_np.mean()),
        "accuracy": float(correct.mean()),
        "reliability": reliability_bins(conf_np, correct),
    }
