"""Held-out evaluation: the numbers that decide whether a model ships.

Reports what a product owner actually needs, not just top-1:
  * calibrated confidence quality (ECE, Brier, reliability diagram)
  * coverage and precision of the accept/abstain policy at the stored thresholds
  * per-category accuracy, so a regression in one category cannot hide inside a global average
  * fine-grained error structure: how many errors are confusions *within* a category (hard, look-alike
    items) vs across categories (usually a data or pipeline bug)
"""
from __future__ import annotations

import base64
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .calibration import calibration_summary
from .config import Config
from .data import ImageFolderSplit, eval_transform
from .infer import run_inference
from .model import load_checkpoint, pick_device
from .selective import apply_thresholds, risk_coverage_curve, selective_metrics


def compute_report(logits: torch.Tensor, labels: torch.Tensor, classes: list[str], category_of: dict[str, str],
                   temperature: float, thresholds: dict[str, float]) -> dict:
    probs = torch.softmax(logits.float() / temperature, 1)
    conf_t, pred_t = probs.max(1)
    conf, pred, y = conf_t.numpy(), pred_t.numpy(), labels.numpy()
    correct = pred == y
    k = min(5, probs.shape[1])
    top5 = (probs.topk(k, 1).indices == labels[:, None]).any(1).float().mean().item()

    pred_cat = [category_of[classes[p]] for p in pred]
    true_cat = [category_of[classes[t]] for t in y]
    accepted = apply_thresholds(conf, pred_cat, thresholds)

    per_class = {}
    for c_idx, c_name in enumerate(classes):
        m = y == c_idx
        if m.any():
            per_class[c_name] = {"n": int(m.sum()), "top1": float(correct[m].mean())}

    per_category = {}
    tc = np.asarray(true_cat)
    for cat in sorted(set(true_cat)):
        m = tc == cat
        sel = selective_metrics(conf[m], correct[m], accepted[m])
        per_category[cat] = {
            "n": int(m.sum()),
            "n_classes": sum(1 for c in classes if category_of[c] == cat),
            "top1": float(correct[m].mean()),
            "coverage": sel["coverage"],
            "selective_precision": sel["selective_precision"],
            "threshold": thresholds.get(cat, thresholds["__global__"]),
        }

    errors = ~correct
    within = sum(1 for i in np.where(errors)[0] if pred_cat[i] == true_cat[i])
    confused = Counter((classes[y[i]], classes[pred[i]]) for i in np.where(errors)[0])

    cal = calibration_summary(logits, labels, temperature)
    raw = calibration_summary(logits, labels, 1.0)
    return {
        "n": int(y.size),
        "num_classes": len(classes),
        "top1": float(correct.mean()),
        "top5": float(top5),
        "macro_top1": float(np.mean([v["top1"] for v in per_class.values()])),
        "calibration": {
            "temperature": temperature,
            "ece": cal["ece"], "ece_uncalibrated": raw["ece"],
            "nll": cal["nll"], "brier": cal["brier"],
            "reliability": cal["reliability"],
        },
        "selective": {**selective_metrics(conf, correct, accepted),
                      "target_precision_note": "thresholds fitted on val, measured here"},
        "risk_coverage": risk_coverage_curve(conf, correct),
        "errors": {
            "total": int(errors.sum()),
            "within_category": int(within),
            "cross_category": int(errors.sum() - within),
            "top_confusions": [{"true": a, "pred": b, "count": n} for (a, b), n in confused.most_common(15)],
        },
        "per_category": per_category,
        "worst_classes": sorted(({"class": k, **v} for k, v in per_class.items()), key=lambda r: r["top1"])[:15],
    }


def evaluate(checkpoint: str | Path, cfg: Config, split: str = "test", out: str | Path | None = None) -> dict:
    device = pick_device()
    model, meta = load_checkpoint(checkpoint, device)
    ds = ImageFolderSplit(Path(cfg.data.root) / split, meta.classes, eval_transform(meta.image_size))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    outputs = run_inference(model, loader, device, with_embeddings=False)
    report = compute_report(outputs.logits, outputs.labels, meta.classes, meta.category_of,
                            meta.temperature, meta.thresholds)
    report["model"] = {"backbone": meta.backbone, "version": meta.version, "split": split,
                       "target_precision": meta.target_precision}
    out = Path(out or Path(checkpoint).parent / f"report_{split}.json")
    out.write_text(json.dumps(report, indent=2))
    write_html_report(report, out.with_suffix(".html"))
    return report


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def write_html_report(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = [b for b in report["calibration"]["reliability"] if b["count"]]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.bar([(b["lo"] + b["hi"]) / 2 for b in rel], [b["accuracy"] for b in rel], width=1 / 15,
           edgecolor="#1f4e79", color="#9cc3e6", label="accuracy")
    ax.plot([b["confidence"] for b in rel], [b["accuracy"] for b in rel], "o", color="#1f4e79", ms=3)
    ax.set(xlabel="calibrated confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1),
           title=f"Reliability (ECE {report['calibration']['ece']:.3f})")
    rel_png = _png(fig)
    plt.close(fig)

    rc = report["risk_coverage"]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(rc["coverage"], rc["risk"], color="#c55a11")
    ax.axvline(report["selective"]["coverage"], ls=":", color="#555")
    ax.set(xlabel="coverage (fraction auto-accepted)", ylabel="error rate on accepted",
           title=f"Risk-coverage (AURC {rc['aurc']:.3f})", xlim=(0, 1), ylim=(0, None))
    rc_png = _png(fig)
    plt.close(fig)

    def row(cells):
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    def fmt(v):
        return "–" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)

    cats = "".join(row([k, v["n"], fmt(v["top1"]), fmt(v["coverage"]), fmt(v["selective_precision"]),
                        fmt(v["threshold"])])
                   for k, v in sorted(report["per_category"].items(), key=lambda kv: kv[1]["top1"]))
    conf = "".join(row([c["true"], c["pred"], c["count"]]) for c in report["errors"]["top_confusions"])
    s, c, e = report["selective"], report["calibration"], report["errors"]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Evaluation report</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:8px 0 24px}}
td,th{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left}}
.kpi{{display:inline-block;margin:0 24px 12px 0}}.kpi b{{display:block;font-size:22px}}</style></head><body>
<h1>Evaluation report — {report.get('model', {}).get('backbone', '')}</h1>
<p>Split: {report.get('model', {}).get('split', '')} · {report['n']} images · {report['num_classes']} classes ·
version {report.get('model', {}).get('version', '')}</p>
<div class="kpi">Top-1<b>{report['top1']:.3f}</b></div><div class="kpi">Top-5<b>{report['top5']:.3f}</b></div>
<div class="kpi">ECE (raw → calibrated)<b>{c['ece_uncalibrated']:.3f} → {c['ece']:.3f}</b></div>
<div class="kpi">Auto-accept coverage<b>{s['coverage']:.3f}</b></div>
<div class="kpi">Precision when accepted<b>{fmt(s['selective_precision'])}</b></div>
<div class="kpi">Errors within / across category<b>{e['within_category']} / {e['cross_category']}</b></div>
<p><img src="data:image/png;base64,{rel_png}"> <img src="data:image/png;base64,{rc_png}"></p>
<h2>Per category (worst first)</h2><table><tr><th>category</th><th>n</th><th>top-1</th><th>coverage</th>
<th>precision when accepted</th><th>threshold</th></tr>{cats}</table>
<h2>Most frequent confusions</h2><table><tr><th>true</th><th>predicted</th><th>count</th></tr>{conf}</table>
</body></html>"""
    path.write_text(html)
