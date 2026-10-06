"""Release gates: absolute quality bars plus no-regression checks against the production baseline.

`itemid gate` exits non-zero on any breach, so CI blocks the merge / promotion.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import GateConfig


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def check(report: dict, gates: GateConfig, baseline: dict | None = None, latency: dict | None = None,
          target_precision: float | None = None) -> list[GateResult]:
    r: list[GateResult] = []
    top1, ece = report["top1"], report["calibration"]["ece"]
    sel = report["selective"]
    r.append(GateResult("min_top1_accuracy", top1 >= gates.min_top1_accuracy,
                        f"top1={top1:.4f} >= {gates.min_top1_accuracy}"))
    r.append(GateResult("max_ece", ece <= gates.max_ece, f"ece={ece:.4f} <= {gates.max_ece}"))
    r.append(GateResult("min_coverage_at_target", sel["coverage"] >= gates.min_coverage_at_target,
                        f"coverage={sel['coverage']:.4f} >= {gates.min_coverage_at_target}"))
    if target_precision is not None and sel["selective_precision"] is not None:
        # Thresholds were fitted on val; allow 2 points of slack for val->test sampling noise.
        floor = target_precision - 0.02
        ok = sel["selective_precision"] >= floor
        r.append(GateResult("precision_holds_on_test", ok,
                            f"precision_when_accepted={sel['selective_precision']:.4f} >= {floor:.2f}"))
    if latency:
        p95 = latency["p95_ms"]
        r.append(GateResult("max_p95_latency_ms", p95 <= gates.max_p95_latency_ms,
                            f"p95={p95:.1f}ms <= {gates.max_p95_latency_ms}ms"))
    if baseline:
        d = baseline["top1"] - top1
        r.append(GateResult("no_top1_regression", d <= gates.max_top1_drop,
                            f"top1 drop={d:+.4f} <= {gates.max_top1_drop}"))
        de = ece - baseline["calibration"]["ece"]
        r.append(GateResult("no_ece_regression", de <= gates.max_ece_increase,
                            f"ece increase={de:+.4f} <= {gates.max_ece_increase}"))
        worst, worst_cat = 0.0, None
        for cat, m in report["per_category"].items():
            b = baseline["per_category"].get(cat)
            if b and b["n"] >= 10:
                drop = b["top1"] - m["top1"]
                if drop > worst:
                    worst, worst_cat = drop, cat
        r.append(GateResult("no_category_regression", worst <= gates.max_category_top1_drop,
                            f"worst category drop={worst:+.4f} ({worst_cat}) <= {gates.max_category_top1_drop}"))
    return r


def run_gates(report_path, gates: GateConfig, baseline_path=None, latency_path=None,
              target_precision: float | None = None) -> tuple[bool, list[GateResult]]:
    report = json.loads(Path(report_path).read_text())
    baseline = json.loads(Path(baseline_path).read_text()) if baseline_path else None
    latency = json.loads(Path(latency_path).read_text()) if latency_path else None
    tp = target_precision if target_precision is not None else report.get("model", {}).get("target_precision")
    results = check(report, gates, baseline, latency, tp)
    return all(g.passed for g in results), results
