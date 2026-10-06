"""Coverage vs accuracy: where should the next month of ML work go?

Combines the evaluation report (how good we are per category) with traffic (how often each category
is actually submitted, from the serving feedback store) into three numbers per category:

  * automation_gap   = traffic share x (1 - auto-accept coverage)  -> volume still going to humans
  * error_exposure   = traffic share x coverage x (1 - precision)  -> wrong answers customers see
  * unsupported_share (global) = traffic for items outside the catalog -> case for adding categories

The output is a ranked, explainable table, not an oracle: it is the starting point for the
prioritisation conversation with product.
"""
from __future__ import annotations

import json
from pathlib import Path


def prioritize(report: dict, traffic: dict[str, int] | None = None, unsupported: int = 0) -> dict:
    cats = report["per_category"]
    if not traffic:
        traffic = {c: v["n"] for c, v in cats.items()}  # fall back to test-set distribution
    total = sum(traffic.values()) + unsupported
    rows = []
    for cat, m in cats.items():
        share = traffic.get(cat, 0) / total if total else 0.0
        prec = m["selective_precision"] if m["selective_precision"] is not None else 1.0
        rows.append({
            "category": cat,
            "traffic_share": round(share, 4),
            "top1": round(m["top1"], 4),
            "coverage": round(m["coverage"], 4),
            "automation_gap": round(share * (1 - m["coverage"]), 4),
            "error_exposure": round(share * m["coverage"] * (1 - prec), 4),
        })
    unsupported_share = unsupported / total if total else 0.0
    gap_total = sum(r["automation_gap"] for r in rows)
    rows.sort(key=lambda r: (r["error_exposure"], r["automation_gap"]), reverse=True)
    if unsupported_share > gap_total:
        recommendation = (f"Expand coverage: {unsupported_share:.1%} of traffic is outside the catalog, more than "
                          f"the {gap_total:.1%} of supported traffic that still needs human review.")
    else:
        top = rows[0]["category"] if rows else "-"
        recommendation = (f"Improve accuracy in existing categories first: supported traffic sent to review "
                          f"({gap_total:.1%}) exceeds unsupported traffic ({unsupported_share:.1%}). "
                          f"Start with '{top}'.")
    return {"recommendation": recommendation, "unsupported_share": round(unsupported_share, 4),
            "automation_gap_total": round(gap_total, 4), "categories": rows}


def prioritize_files(report_path: str | Path, traffic_path: str | Path | None = None) -> dict:
    report = json.loads(Path(report_path).read_text())
    traffic, unsupported = None, 0
    if traffic_path:
        t = json.loads(Path(traffic_path).read_text())
        traffic, unsupported = t.get("by_category"), t.get("unsupported", 0)
    return prioritize(report, traffic, unsupported)
