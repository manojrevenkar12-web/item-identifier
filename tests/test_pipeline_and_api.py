"""End-to-end: train -> evaluate -> gate -> ONNX export -> serve -> feedback -> labelling queue."""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from itemid.evaluate import evaluate
from itemid.export import export_onnx
from itemid.gates import run_gates
from itemid.model import load_checkpoint


def test_checkpoint_contract(trained, tiny_cfg):
    _, meta = load_checkpoint(trained)
    assert len(meta.classes) == 12 and meta.temperature > 0
    assert "__global__" in meta.thresholds
    assert set(meta.category_of.values()) == {"circle", "square", "triangle"}


def test_evaluate_and_gate(trained, tiny_cfg):
    report = evaluate(trained, tiny_cfg, "test")
    assert report["n"] == 12 * 15
    assert report["errors"]["total"] == report["errors"]["within_category"] + report["errors"]["cross_category"]
    assert Path(trained).parent.joinpath("report_test.html").exists()
    ok, results = run_gates(Path(trained).parent / "report_test.json", tiny_cfg.gates.model_copy(
        update={"min_top1_accuracy": 0.0, "max_ece": 1.0, "min_coverage_at_target": 0.0}), target_precision=0.0)
    assert ok, [r for r in results if not r.passed]


def test_onnx_parity(trained, tmp_path):
    res = export_onnx(trained, tmp_path / "m.onnx")
    assert res["argmax_match"] and res["max_abs_diff"] < 1e-3


@pytest.fixture()
def client(trained, tmp_path):
    from itemid.serving.app import create_app
    with TestClient(create_app(str(trained), str(tmp_path / "fb.sqlite"))) as c:
        yield c


def _png(path: Path) -> bytes:
    buf = io.BytesIO()
    Image.open(path).save(buf, format="PNG")
    return buf.getvalue()


def test_api_identify_feedback_queue(client, tiny_cfg):
    img = next((Path(tiny_cfg.data.root) / "test" / "circle_red").glob("*.png"))
    r = client.post("/v1/identify", files={"file": ("x.png", _png(img), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] in ("accept", "abstain")
    assert abs(sum(a["probability"] for a in body["alternatives"]) - 1) < 0.2
    assert body["alternatives"][0]["label"] == body["label"]

    wrong = "square_teal" if body["label"] != "square_teal" else "circle_red"
    assert client.post("/v1/feedback", json={"request_id": body["request_id"], "true_label": wrong}).status_code == 200
    queue = client.get("/v1/labeling-queue").json()["items"]
    assert queue and queue[0]["request_id"] == body["request_id"] and queue[0]["true_label"] == wrong

    traffic = client.get("/v1/traffic").json()
    assert sum(traffic["by_category"].values()) == 1
    assert client.get("/metrics").text.count("itemid_requests_total") > 0


def test_api_rejects_bad_input(client):
    assert client.post("/v1/identify", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 422
    tiny = io.BytesIO()
    Image.new("RGB", (8, 8)).save(tiny, format="PNG")
    assert client.post("/v1/identify", files={"file": ("t.png", tiny.getvalue(), "image/png")}).status_code == 422
    r = client.post("/v1/feedback", json={"request_id": "nope", "true_label": "circle_red"})
    assert r.status_code == 404
    r = client.post("/v1/feedback", json={"request_id": "x", "true_label": "unknown_item"})
    assert r.status_code == 422


def test_out_of_catalog_feedback_counts_as_unsupported(client, tiny_cfg):
    img = next((Path(tiny_cfg.data.root) / "test" / "square_teal").glob("*.png"))
    rid = client.post("/v1/identify", files={"file": ("x.png", _png(img), "image/png")}).json()["request_id"]
    assert client.post("/v1/feedback", json={"request_id": rid, "true_label": "hexagon_blue",
                                             "in_catalog": False}).status_code == 200
    assert client.get("/v1/traffic").json()["unsupported"] == 1
