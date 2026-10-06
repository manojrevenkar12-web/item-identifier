"""ONNX export with a numerical parity check, and a latency benchmark for the release gate."""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .model import load_checkpoint


def export_onnx(checkpoint: str | Path, out: str | Path, opset: int = 17, atol: float = 1e-3) -> dict:
    model, meta = load_checkpoint(checkpoint, "cpu")
    dummy = torch.randn(2, 3, meta.image_size, meta.image_size)
    out = Path(out)
    torch.onnx.export(model, dummy, str(out), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset,
                      dynamo=False)  # TorchScript exporter: stable for timm CNN/ViT graphs
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, meta.image_size, meta.image_size)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    same_argmax = bool((ref.argmax(1) == got.argmax(1)).all())
    if max_diff > atol or not same_argmax:
        raise RuntimeError(f"ONNX parity failed: max |diff|={max_diff:.2e}, argmax match={same_argmax}")
    return {"path": str(out), "max_abs_diff": max_diff, "argmax_match": same_argmax}


def benchmark(checkpoint: str | Path, onnx_path: str | Path | None = None, runs: int = 100, warmup: int = 10,
              threads: int | None = None, out: str | Path | None = None) -> dict:
    """Batch-1 end-to-end model latency (preprocessed tensor in, logits out) on this machine's CPU."""
    if threads:
        torch.set_num_threads(threads)
    model, meta = load_checkpoint(checkpoint, "cpu")
    x = torch.randn(1, 3, meta.image_size, meta.image_size)
    if onnx_path:
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        xn = x.numpy()

        def fn():
            sess.run(None, {"image": xn})
        runtime = "onnxruntime"
    else:
        def fn():
            with torch.inference_mode():
                model(x)
        runtime = "torch"
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    q = lambda p: times[min(len(times) - 1, int(p * len(times)))]  # noqa: E731
    result = {"runtime": runtime, "backbone": meta.backbone, "image_size": meta.image_size, "runs": runs,
              "p50_ms": round(statistics.median(times), 2), "p95_ms": round(q(0.95), 2),
              "p99_ms": round(q(0.99), 2), "threads": threads or torch.get_num_threads(),
              "cpu": platform.processor() or platform.machine()}
    if out:
        Path(out).write_text(json.dumps(result, indent=2))
    return result
