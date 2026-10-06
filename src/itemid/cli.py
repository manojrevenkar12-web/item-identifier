"""Command line entry point: `itemid <command> ...`."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Config


def _cfg(args) -> Config:
    return Config.load(args.config) if args.config else Config()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="itemid")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate the small synthetic dataset")
    s.add_argument("--out", default="data/synthetic")

    s = sub.add_parser("train", help="fine-tune, calibrate and fit abstention thresholds")
    s.add_argument("--config", required=True)

    s = sub.add_parser("evaluate", help="held-out evaluation report (JSON + HTML)")
    s.add_argument("--config", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--out")

    s = sub.add_parser("gate", help="release gates; exits 1 on any breach")
    s.add_argument("--config", required=True)
    s.add_argument("--report", required=True)
    s.add_argument("--baseline")
    s.add_argument("--latency")

    s = sub.add_parser("export", help="export ONNX with parity check")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out", required=True)

    s = sub.add_parser("bench", help="CPU latency benchmark")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--onnx")
    s.add_argument("--runs", type=int, default=100)
    s.add_argument("--threads", type=int)
    s.add_argument("--out")

    s = sub.add_parser("al-simulate", help="compare active-learning strategies")
    s.add_argument("--config", required=True)
    s.add_argument("--strategies", default="random,margin,margin_diverse")
    s.add_argument("--rounds", type=int, default=4)
    s.add_argument("--budget", type=int, default=500)
    s.add_argument("--out", default="runs/al")

    s = sub.add_parser("prioritize", help="coverage-vs-accuracy priorities per category")
    s.add_argument("--report", required=True)
    s.add_argument("--traffic", help="JSON from GET /v1/traffic")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--onnx")
    s.add_argument("--db", default="data/feedback.sqlite")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)

    if args.cmd == "synth":
        from .synthetic import generate
        print(generate(args.out))
    elif args.cmd == "train":
        from .train import train
        print(train(_cfg(args)))
    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        r = evaluate(args.checkpoint, _cfg(args), args.split, args.out)
        print(json.dumps({k: r[k] for k in ("n", "top1", "top5", "macro_top1")} |
                         {"ece": r["calibration"]["ece"], **{k: r["selective"][k] for k in
                                                             ("coverage", "selective_precision")}}, indent=2))
    elif args.cmd == "gate":
        from .gates import run_gates
        ok, results = run_gates(args.report, _cfg(args).gates, args.baseline, args.latency)
        for g in results:
            print(f"[{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.detail}")
        print("GATES PASSED" if ok else "GATES FAILED")
        return 0 if ok else 1
    elif args.cmd == "export":
        from .export import export_onnx
        print(json.dumps(export_onnx(args.checkpoint, args.out), indent=2))
    elif args.cmd == "bench":
        from .export import benchmark
        print(json.dumps(benchmark(args.checkpoint, args.onnx, args.runs, threads=args.threads, out=args.out),
                         indent=2))
    elif args.cmd == "al-simulate":
        from .active_learning import STRATEGIES, simulate
        strategies = [x.strip() for x in args.strategies.split(",")]
        bad = [x for x in strategies if x not in STRATEGIES]
        if bad:
            p.error(f"unknown strategies {bad}; choose from {STRATEGIES}")
        print(json.dumps(simulate(_cfg(args), strategies, args.rounds, args.budget, args.out), indent=2))
    elif args.cmd == "prioritize":
        from .prioritize import prioritize_files
        print(json.dumps(prioritize_files(args.report, args.traffic), indent=2))
    elif args.cmd == "serve":
        import uvicorn

        from .serving.app import create_app
        uvicorn.run(create_app(args.checkpoint, args.db, args.onnx), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
