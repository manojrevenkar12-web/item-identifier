import numpy as np

from itemid.selective import (
    GLOBAL,
    apply_thresholds,
    fit_thresholds,
    risk_coverage_curve,
    selective_metrics,
    threshold_for_precision,
)


def test_threshold_reaches_target_with_max_coverage():
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([1, 1, 1, 1, 0, 1, 0], dtype=bool)
    t = threshold_for_precision(conf, correct, target=0.8)
    acc = conf >= t
    assert correct[acc].mean() >= 0.8
    assert t == 0.6  # accepting down to 0.6 gives 5/6 = 0.83


def test_fails_closed_when_target_unreachable():
    conf = np.array([0.9, 0.8])
    correct = np.array([0, 0], dtype=bool)
    assert threshold_for_precision(conf, correct, 0.95) == 1.0


def test_ties_are_not_split():
    conf = np.array([0.9, 0.9, 0.9, 0.5])
    correct = np.array([1, 0, 1, 1], dtype=bool)
    # Cutting inside the tie would claim 100% precision; it must not.
    t = threshold_for_precision(conf, correct, 0.99)
    assert t == 1.0


def test_per_category_thresholds_and_fallback():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.3, 1, 400)
    cats = ["easy"] * 200 + ["hard"] * 190 + ["rare"] * 10
    correct = np.r_[rng.uniform(0, 1, 200) < 0.97, rng.uniform(0, 1, 190) < conf[200:390], np.ones(10, bool)]
    th = fit_thresholds(conf, correct, cats, target=0.9, floor=0.3, min_samples=30)
    assert "rare" not in th and GLOBAL in th
    assert th["easy"] < th["hard"]
    accepted = apply_thresholds(conf, cats, th)
    m = selective_metrics(conf, correct, accepted)
    assert 0 < m["coverage"] < 1 and m["accepted"] + m["abstained"] == 400


def test_risk_coverage_monotone_for_ideal_ranking():
    conf = np.linspace(1, 0, 100)
    correct = np.arange(100) < 80
    rc = risk_coverage_curve(conf, correct)
    assert rc["risk"][0] == 0.0 and abs(rc["risk"][-1] - 0.2) < 1e-6
