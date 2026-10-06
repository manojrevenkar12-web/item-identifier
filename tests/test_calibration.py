import numpy as np
import torch

from itemid.calibration import calibration_summary, expected_calibration_error, fit_temperature


def _sample(n=6000, c=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, c, generator=g) * 2.0
    labels = torch.multinomial(torch.softmax(z, 1), 1, generator=g).squeeze(1)
    return z, labels


def test_ece_is_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 200_000)
    correct = rng.uniform(0, 1, conf.size) < conf
    assert expected_calibration_error(conf, correct) < 0.01


def test_ece_detects_overconfidence():
    conf = np.full(1000, 0.95)
    correct = np.arange(1000) < 600  # 60% accurate while claiming 95%
    assert abs(expected_calibration_error(conf, correct) - 0.35) < 1e-9


def test_temperature_recovers_known_overconfidence():
    z, y = _sample()
    t = fit_temperature(z * 3.0, y)  # logits were made 3x too sharp
    assert 2.6 < t < 3.4


def test_temperature_scaling_preserves_accuracy_and_reduces_ece():
    z, y = _sample(seed=1)
    sharp = z * 3.0
    t = fit_temperature(sharp, y)
    before, after = calibration_summary(sharp, y, 1.0), calibration_summary(sharp, y, t)
    assert before["accuracy"] == after["accuracy"]
    assert after["ece"] < before["ece"] / 3
    assert after["nll"] < before["nll"]
