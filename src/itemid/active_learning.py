"""Active learning: spend the labelling budget where the model is wrong or unsure.

Two uses:
  1. `simulate()` -- an offline experiment. Start with a small labelled fraction, treat the rest of
     the train split as an unlabelled pool, and compare query strategies round by round (labels are
     revealed by the "oracle" only when queried). This answers "is uncertainty sampling worth it on
     *our* catalog?" with a learning curve instead of an opinion.
  2. `select()` on production traffic -- ranks abstained / corrected requests for the labelling queue
     (see serving.feedback).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import ImageFolderSplit, build_data, eval_transform
from .infer import run_inference
from .model import load_checkpoint, pick_device

log = logging.getLogger("itemid.al")
STRATEGIES = ("random", "least_confidence", "margin", "entropy", "margin_diverse")


def uncertainty_scores(probs: np.ndarray, strategy: str) -> np.ndarray:
    """Higher = more informative to label."""
    if strategy == "least_confidence":
        return 1.0 - probs.max(1)
    if strategy in ("margin", "margin_diverse"):
        top2 = np.sort(probs, axis=1)[:, -2:]
        return 1.0 - (top2[:, 1] - top2[:, 0])
    if strategy == "entropy":
        return -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1)
    raise ValueError(f"unknown strategy {strategy}")


def k_center_greedy(emb: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """Pick k points that cover the embedding space (Sener & Savarese, 2018). emb rows L2-normalised."""
    n = emb.shape[0]
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(n))]
    dist = 1.0 - emb @ emb[chosen[0]]
    for _ in range(k - 1):
        nxt = int(dist.argmax())
        chosen.append(nxt)
        dist = np.minimum(dist, 1.0 - emb @ emb[nxt])
    return chosen


def select(probs: np.ndarray, emb: np.ndarray | None, budget: int, strategy: str, seed: int = 0) -> list[int]:
    """Return positions (into probs) to send for labelling."""
    n = probs.shape[0]
    budget = min(budget, n)
    if strategy == "random":
        return np.random.default_rng(seed).choice(n, budget, replace=False).tolist()
    scores = uncertainty_scores(probs, strategy)
    if strategy == "margin_diverse" and emb is not None and emb.size:
        # Uncertainty alone picks near-duplicates of the same confusing pair; take a 5x uncertain
        # shortlist and spread the budget across it in embedding space.
        shortlist = np.argsort(-scores)[: budget * 5]
        picked = k_center_greedy(emb[shortlist], budget, seed)
        return shortlist[picked].tolist()
    return np.argsort(-scores)[:budget].tolist()


def simulate(cfg: Config, strategies: list[str], rounds: int, budget: int, out_dir: str | Path) -> dict:
    """Run the pool-based experiment. Each round retrains from the pretrained backbone (no warm start),
    which is slower but avoids the well-known bias of warm-started AL comparisons."""
    from .train import train  # local import: train imports this package

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    base = build_data(cfg)
    log.info("AL simulation: %d labelled, %d in pool", len(base.labelled_idx), len(base.pool_idx))
    pool_view = ImageFolderSplit(Path(cfg.data.root) / "train", base.classes, eval_transform(cfg.data.image_size))
    results: dict[str, list[dict]] = {}

    for strategy in strategies:
        labelled, pool = list(base.labelled_idx), list(base.pool_idx)
        curve = []
        for r in range(rounds + 1):
            rcfg = cfg.model_copy(deep=True)
            rcfg.train.out_dir = out_dir / strategy / f"round{r}"
            data = build_data(rcfg, labelled_idx=labelled)
            ckpt = train(rcfg, data=data, version=f"al-{strategy}-r{r}")
            meta_report = json.loads((Path(rcfg.train.out_dir) / "calibration.json").read_text())
            model, _ = load_checkpoint(ckpt, device)
            val_out = run_inference(model, data.loader(data.val, cfg.data.batch_size, False, cfg.data.num_workers),
                                    device, with_embeddings=False)
            acc = float((val_out.logits.argmax(1) == val_out.labels).float().mean())
            curve.append({"round": r, "labelled": len(labelled), "val_top1": acc,
                          "val_ece": meta_report["after"]["ece"]})
            log.info("%s round %d: labelled=%d val_top1=%.4f", strategy, r, len(labelled), acc)
            if r == rounds or not pool:
                break
            pool_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(pool_view, pool),
                                                      batch_size=cfg.data.batch_size,
                                                      num_workers=cfg.data.num_workers)
            po = run_inference(model, pool_loader, device, with_embeddings=True)
            t = json.loads((Path(rcfg.train.out_dir) / "calibration.json").read_text())["after"]["temperature"]
            picks = select(po.probs(t).numpy(), po.embeddings.numpy(), budget, strategy, seed=cfg.data.seed + r)
            chosen = {pool[i] for i in picks}
            labelled = sorted(set(labelled) | chosen)
            pool = [i for i in pool if i not in chosen]
        results[strategy] = curve

    (out_dir / "al_results.json").write_text(json.dumps(results, indent=2))
    _plot(results, out_dir / "al_curve.png")
    return results


def _plot(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, curve in results.items():
        ax.plot([c["labelled"] for c in curve], [c["val_top1"] for c in curve], marker="o", label=name)
    ax.set(xlabel="labelled training images", ylabel="val top-1", title="Active learning: accuracy per label")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
