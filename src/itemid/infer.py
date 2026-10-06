"""Batched inference helpers shared by calibration, evaluation and active learning."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class Outputs:
    logits: torch.Tensor      # [N, C] raw (uncalibrated) logits
    labels: torch.Tensor      # [N]
    embeddings: torch.Tensor  # [N, D] pooled backbone features

    def probs(self, temperature: float = 1.0) -> torch.Tensor:
        return F.softmax(self.logits / temperature, dim=1)

    def top1(self, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        conf, pred = self.probs(temperature).max(1)
        return conf.numpy(), pred.numpy()


@torch.no_grad()
def run_inference(model, loader: DataLoader, device, with_embeddings: bool = True) -> Outputs:
    model.eval()
    all_logits, all_labels, all_emb = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        emb = model.embed(x)
        logits = model.head(emb)
        all_logits.append(logits.float().cpu())
        all_labels.append(torch.as_tensor(y))
        if with_embeddings:
            all_emb.append(F.normalize(emb.float(), dim=1).cpu())
    return Outputs(
        logits=torch.cat(all_logits),
        labels=torch.cat(all_labels).long(),
        embeddings=torch.cat(all_emb) if with_embeddings else torch.empty(0),
    )
