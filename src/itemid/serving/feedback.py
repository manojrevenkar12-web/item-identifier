"""Prediction log + human feedback, the raw material for the error-driven labelling loop.

Every prediction is logged with a hash of the image. Reviewers (or customers) confirm or correct
the label via /v1/feedback. The labelling queue then prioritises:
  1. corrected predictions where the model was *confident* (accepted errors -- most expensive),
  2. abstentions with the smallest top-1/top-2 margin (most informative per label).
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  request_id TEXT PRIMARY KEY,
  ts REAL NOT NULL,
  model_version TEXT NOT NULL,
  image_sha256 TEXT NOT NULL,
  image_path TEXT,
  label TEXT NOT NULL,
  category TEXT NOT NULL,
  confidence REAL NOT NULL,
  margin REAL NOT NULL,
  decision TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
  request_id TEXT PRIMARY KEY REFERENCES predictions(request_id),
  ts REAL NOT NULL,
  true_label TEXT NOT NULL,
  in_catalog INTEGER NOT NULL,
  source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pred_decision ON predictions(decision);
"""


class FeedbackStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def log_prediction(self, request_id: str, image_sha256: str, pred, image_path: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (request_id, time.time(), pred.model_version, image_sha256, image_path, pred.label,
                 pred.category, pred.confidence, pred.margin, pred.decision))

    def add_feedback(self, request_id: str, true_label: str, in_catalog: bool, source: str) -> bool:
        with self._lock, self._conn:
            if not self._conn.execute("SELECT 1 FROM predictions WHERE request_id=?", (request_id,)).fetchone():
                return False
            self._conn.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?)",
                               (request_id, time.time(), true_label, int(in_catalog), source))
            return True

    def labeling_queue(self, limit: int = 100) -> list[dict]:
        sql = """
        SELECT p.request_id, p.image_sha256, p.image_path, p.label, p.category, p.confidence, p.margin,
               p.decision, f.true_label,
               CASE WHEN f.true_label IS NOT NULL AND f.true_label != p.label AND p.decision='accept' THEN 0
                    WHEN f.true_label IS NULL AND p.decision='abstain' THEN 1
                    WHEN f.true_label IS NOT NULL AND f.true_label != p.label THEN 2
                    ELSE 3 END AS priority
        FROM predictions p LEFT JOIN feedback f USING(request_id)
        WHERE (p.decision='abstain' AND f.true_label IS NULL)
           OR (f.true_label IS NOT NULL AND f.true_label != p.label)
        ORDER BY priority ASC, p.margin ASC
        LIMIT ?"""
        cols = ["request_id", "image_sha256", "image_path", "predicted", "category", "confidence", "margin",
                "decision", "true_label", "priority"]
        with self._lock:
            return [dict(zip(cols, r, strict=True)) for r in self._conn.execute(sql, (limit,)).fetchall()]

    def traffic(self) -> dict:
        """Traffic by predicted category; items reviewers flagged as out-of-catalog are counted separately."""
        with self._lock:
            rows = self._conn.execute("""
                SELECT p.category, f.in_catalog FROM predictions p LEFT JOIN feedback f USING(request_id)
            """).fetchall()
            acc = self._conn.execute("""
                SELECT p.decision, SUM(CASE WHEN f.true_label = p.label THEN 1 ELSE 0 END),
                       SUM(CASE WHEN f.true_label IS NOT NULL THEN 1 ELSE 0 END), COUNT(*)
                FROM predictions p LEFT JOIN feedback f USING(request_id) GROUP BY p.decision
            """).fetchall()
        by_cat: dict[str, int] = {}
        unsupported = 0
        for cat, in_cat in rows:
            if in_cat == 0:
                unsupported += 1
            else:
                by_cat[cat] = by_cat.get(cat, 0) + 1
        live = {d: {"total": n, "with_feedback": fb, "confirmed_correct": ok,
                    "observed_precision": (ok / fb) if fb else None} for d, ok, fb, n in acc}
        return {"by_category": by_cat, "unsupported": unsupported, "live_quality": live}
