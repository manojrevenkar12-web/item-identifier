"""HTTP API.

    POST /v1/identify        multipart image -> label, calibrated confidence, accept/abstain, top-k
    POST /v1/feedback        reviewer/customer confirms or corrects a prediction
    GET  /v1/labeling-queue  error-driven queue for the labelling team
    GET  /v1/traffic         per-category traffic + live precision (input to `itemid prioritize`)
    GET  /v1/model           served model version, classes, thresholds
    GET  /healthz /readyz /metrics

Config via env: ITEMID_MODEL (checkpoint), ITEMID_ONNX (optional), ITEMID_DB, ITEMID_STORE_IMAGES_DIR,
ITEMID_MAX_UPLOAD_MB.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .feedback import FeedbackStore
from .predictor import InvalidImage, Predictor, decode_image

log = logging.getLogger("itemid.api")

REQUESTS = Counter("itemid_requests_total", "Identify requests", ["decision"])
ERRORS = Counter("itemid_errors_total", "Rejected requests", ["reason"])
LATENCY = Histogram("itemid_inference_seconds", "Model inference latency",
                    buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.2))
CONFIDENCE = Histogram("itemid_confidence", "Calibrated top-1 confidence",
                       buckets=tuple(i / 10 for i in range(1, 11)))


class FeedbackIn(BaseModel):
    request_id: str
    true_label: str = Field(min_length=1, max_length=200)
    in_catalog: bool = True
    source: str = Field("reviewer", max_length=50)


class State:
    predictor: Predictor | None = None
    store: FeedbackStore | None = None
    image_dir: Path | None = None
    max_bytes: int = 10 * 1024 * 1024


state = State()


def create_app(checkpoint: str | None = None, db: str | None = None, onnx: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ckpt = checkpoint or os.environ.get("ITEMID_MODEL", "models/model.pt")
        state.predictor = Predictor(ckpt, onnx or os.environ.get("ITEMID_ONNX") or None)
        state.store = FeedbackStore(db or os.environ.get("ITEMID_DB", "data/feedback.sqlite"))
        img_dir = os.environ.get("ITEMID_STORE_IMAGES_DIR")
        state.image_dir = Path(img_dir) if img_dir else None
        state.max_bytes = int(float(os.environ.get("ITEMID_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
        log.info("loaded model version=%s classes=%d", state.predictor.meta.version,
                 len(state.predictor.meta.classes))
        yield

    app = FastAPI(title="Item Identifier", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        if state.predictor is None:
            raise HTTPException(503, "model not loaded")
        return {"status": "ready", "model_version": state.predictor.meta.version}

    @app.get("/metrics")
    def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model")
    def model_info():
        m = state.predictor.meta
        return {"version": m.version, "backbone": m.backbone, "num_classes": len(m.classes),
                "temperature": m.temperature, "target_precision": m.target_precision, "thresholds": m.thresholds}

    @app.post("/v1/identify")
    async def identify(file: UploadFile = File(...)):
        data = await file.read(state.max_bytes + 1)
        if len(data) > state.max_bytes:
            ERRORS.labels("too_large").inc()
            raise HTTPException(413, "upload too large")
        try:
            img = decode_image(data)
        except InvalidImage as e:
            ERRORS.labels("invalid_image").inc()
            raise HTTPException(422, str(e)) from e
        pred = await run_in_threadpool(state.predictor.predict, img)
        request_id = uuid.uuid4().hex
        sha = hashlib.sha256(data).hexdigest()
        image_path = None
        if state.image_dir is not None:
            state.image_dir.mkdir(parents=True, exist_ok=True)
            image_path = str(state.image_dir / f"{sha}.jpg")
            if not Path(image_path).exists():
                img.save(image_path, quality=92)
        state.store.log_prediction(request_id, sha, pred, image_path)
        REQUESTS.labels(pred.decision).inc()
        LATENCY.observe(pred.latency_ms / 1000)
        CONFIDENCE.observe(pred.confidence)
        return {"request_id": request_id, **pred.as_dict()}

    @app.post("/v1/feedback")
    def feedback(body: FeedbackIn):
        classes = set(state.predictor.meta.classes)
        if body.in_catalog and body.true_label not in classes:
            raise HTTPException(422, "true_label is not in the catalog; set in_catalog=false for new items")
        if not state.store.add_feedback(body.request_id, body.true_label, body.in_catalog, body.source):
            raise HTTPException(404, "unknown request_id")
        return {"status": "recorded"}

    @app.get("/v1/labeling-queue")
    def labeling_queue(limit: int = 100):
        return {"items": state.store.labeling_queue(max(1, min(limit, 1000)))}

    @app.get("/v1/traffic")
    def traffic():
        return state.store.traffic()

    return app


app = create_app()
