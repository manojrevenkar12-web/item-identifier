import copy

import numpy as np
import pytest

from itemid.active_learning import k_center_greedy, select, uncertainty_scores
from itemid.config import GateConfig
from itemid.data import stratified_subset
from itemid.gates import check
from itemid.prioritize import prioritize


def test_margin_picks_most_ambiguous():
    probs = np.array([[0.9, 0.1, 0.0], [0.5, 0.49, 0.01], [0.6, 0.3, 0.1], [0.34, 0.33, 0.33]])
    assert set(select(probs, None, 2, "margin")) == {1, 3}
    assert uncertainty_scores(probs, "entropy").argmax() == 3


def test_k_center_spreads_out():
    a = np.array([[1, 0], [0.999, 0.045], [0, 1], [0.045, 0.999]])
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    picked = k_center_greedy(a, 2, seed=0)
    assert {p // 2 for p in picked} == {0, 1}  # one from each cluster


def test_margin_diverse_budget_and_unique():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(5), 200)
    emb = rng.normal(size=(200, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    picks = select(probs, emb, 20, "margin_diverse")
    assert len(picks) == len(set(picks)) == 20


def test_stratified_subset_keeps_every_class():
    targets = [0] * 50 + [1] * 3 + [2] * 20
    lab, pool = stratified_subset(targets, 0.1, seed=0)
    assert {targets[i] for i in lab} == {0, 1, 2}
    assert sorted(lab + pool) == list(range(len(targets)))


REPORT = {
    "top1": 0.90, "calibration": {"ece": 0.03},
    "selective": {"coverage": 0.75, "selective_precision": 0.96},
    "per_category": {"A": {"n": 100, "top1": 0.95, "coverage": 0.9, "selective_precision": 0.99},
                     "B": {"n": 100, "top1": 0.85, "coverage": 0.6, "selective_precision": 0.93}},
}


def test_gates_pass_and_catch_category_regression():
    g = GateConfig()
    assert all(r.passed for r in check(REPORT, g, baseline=REPORT, target_precision=0.95))
    worse = copy.deepcopy(REPORT)
    worse["per_category"]["B"]["top1"] = 0.78  # global top1 unchanged, one category regressed
    failed = [r.name for r in check(worse, g, baseline=REPORT) if not r.passed]
    assert failed == ["no_category_regression"]


@pytest.mark.parametrize("field,value,gate", [("top1", 0.7, "min_top1_accuracy")])
def test_gate_absolute(field, value, gate):
    r = copy.deepcopy(REPORT)
    r[field] = value
    assert gate in [x.name for x in check(r, GateConfig()) if not x.passed]


def test_prioritize_recommends_expansion_when_unsupported_dominates():
    out = prioritize(REPORT, traffic={"A": 100, "B": 100}, unsupported=300)
    assert out["recommendation"].startswith("Expand coverage")
    out = prioritize(REPORT, traffic={"A": 100, "B": 100}, unsupported=0)
    assert out["categories"][0]["category"] == "B"
