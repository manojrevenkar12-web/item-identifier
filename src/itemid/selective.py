"""Selective prediction: when should the product trust the model, and when should it say "not sure"?

Thresholds are chosen on the validation split and only *measured* on the test split, so the
reported coverage/precision are honest estimates of what production will see.
"""
from __future__ import annotations

import numpy as np

GLOBAL = "__global__"


def threshold_for_precision(conf: np.ndarray, correct: np.ndarray, target: float,
                            floor: float = 0.0) -> float:
    """Lowest threshold whose accepted set reaches `target` precision (max coverage at that precision).

    Returns 1.0 (accept nothing) if no threshold achieves the target -- failing closed.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return 1.0
    order = np.argsort(-conf)
    sorted_conf = conf[order]
    cum_precision = np.cumsum(correct[order]) / np.arange(1, conf.size + 1)
    # Only consider cut points where the next sample has strictly lower confidence (ties move together).
    is_cut = np.r_[sorted_conf[:-1] > sorted_conf[1:], True]
    ok = np.where(is_cut & (cum_precision >= target))[0]
    if ok.size == 0:
        return 1.0
    return float(max(sorted_conf[ok.max()], floor))


def fit_thresholds(conf: np.ndarray, correct: np.ndarray, pred_category: list[str], target: float,
                   floor: float, min_samples: int) -> dict[str, float]:
    """Global threshold plus a per-category one where there is enough validation data.

    Categories differ in difficulty (a rare model of a common make vs a distinctive make), so one
    global cut either over-abstains on easy categories or under-protects hard ones.
    """
    thresholds = {GLOBAL: threshold_for_precision(conf, correct, target, floor)}
    cats = np.asarray(pred_category)
    for cat in sorted(set(pred_category)):
        mask = cats == cat
        if mask.sum() >= min_samples:
            thresholds[cat] = threshold_for_precision(conf[mask], correct[mask], target, floor)
    return thresholds


def apply_thresholds(conf: np.ndarray, pred_category: list[str], thresholds: dict[str, float]) -> np.ndarray:
    t = np.array([thresholds.get(c, thresholds[GLOBAL]) for c in pred_category])
    return np.asarray(conf) >= t


def selective_metrics(conf: np.ndarray, correct: np.ndarray, accepted: np.ndarray) -> dict:
    accepted = np.asarray(accepted, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    n_acc = int(accepted.sum())
    return {
        "coverage": float(accepted.mean()) if accepted.size else 0.0,
        "selective_precision": float(correct[accepted].mean()) if n_acc else None,
        "accepted": n_acc,
        "abstained": int((~accepted).sum()),
        # Wrong answers the product would have shown to a customer -- the number that matters for pricing.
        "accepted_errors": int((accepted & ~correct).sum()),
    }


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray, points: int = 50) -> dict:
    """Risk (error rate on accepted) vs coverage, plus AURC (lower is better)."""
    order = np.argsort(-np.asarray(conf))
    c = np.asarray(correct, dtype=np.float64)[order]
    n = c.size
    risk = 1.0 - np.cumsum(c) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    idx = np.unique(np.linspace(0, n - 1, min(points, n)).astype(int))
    return {
        "coverage": coverage[idx].round(4).tolist(),
        "risk": risk[idx].round(4).tolist(),
        "aurc": float(risk.mean()),
    }
