"""v2 장면 분석 테스트 · 모델은 가짜(스텁) · 라우터(OpenCV)는 실제로 실행."""
import os
import tempfile
from io import BytesIO
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///" + str(Path(tempfile.mkdtemp()) / "scene_test.db"))

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import ai.services as services
import ai.scene_service as scene_service
from ai.leaf_router import analyze_scene, check_quality
from main import app

ROOT = Path(__file__).resolve().parent.parent
FAR_PHOTO = ROOT / "robust" / "variants" / "00_original.jpg"
SOIL = (110, 80, 60)


def leaf(canvas, center, axes, angle, color=(70, 160, 60)):
    cv2.ellipse(canvas, center, axes, angle, 0, 360, color, -1)
    cv2.ellipse(canvas, center, axes, angle, 0, 360, (30, 70, 30), 3)   # 잎 테두리 그림자


def close_image():
    rng = np.random.default_rng(0)
    img = np.full((500, 500, 3), SOIL, np.uint8)
    img = np.clip(img + rng.integers(-25, 25, img.shape), 0, 255).astype(np.uint8)
    leaf(img, (250, 260), (190, 150), 20)                                  # 화면을 크게 차지하는 잎 1장
    return img


def far_image():
    rng = np.random.default_rng(1)
    img = np.full((600, 600, 3), SOIL, np.uint8)
    img = np.clip(img + rng.integers(-25, 25, img.shape), 0, 255).astype(np.uint8)
    for cx, cy, ang in [(160, 110, 5), (440, 110, -5), (160, 300, 0), (440, 300, 8), (160, 490, -8), (440, 490, 0)]:
        leaf(img, (cx, cy), (128, 88), ang)                               # 비슷한 크기 잎 6장 (각 화면의 약 9%)
    return img


def to_png(rgb):
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def fake_classifier(images, top_k=3):
    one = [{"label": "healthy", "score": 0.9}, {"label": "bean_rust", "score": 0.1}]
    return [one for _ in images] if isinstance(images, list) else one


def test_router_close_and_far_synthetic():
    assert analyze_scene(close_image()).mode == "close"
    far = analyze_scene(far_image())
    assert far.mode == "far"
    assert len(far.boxes) >= 3


def test_router_far_on_field_photo():
    rgb = np.asarray(Image.open(FAR_PHOTO).convert("RGB"))
    scene = analyze_scene(rgb)
    assert scene.mode == "far"
    assert len(scene.boxes) >= 3


def test_quality_gate():
    dark = (close_image() * 0.25).astype(np.uint8)
    assert "dark" in check_quality(dark).reasons
    blurred = cv2.GaussianBlur(close_image(), (0, 0), 12)
    assert not check_quality(blurred).ok
    assert check_quality(close_image()).ok


def test_decision_rules():
    sick = {"label": "bean_rust", "check": False}
    unsure = {"label": "healthy", "check": True}
    fine = {"label": "healthy", "check": False}
    assert scene_service._decide([fine, sick]) == "DISEASE"
    assert scene_service._decide([fine, unsure]) == "CHECK"
    assert scene_service._decide([fine, fine]) == "HEALTHY"


def test_v2_api(monkeypatch):
    monkeypatch.setattr(services, "get_classifier", lambda: fake_classifier)
    with TestClient(app) as client:
        assert client.get("/vision/scene/").status_code == 200
        far = client.post("/api/v2/classify/?draw=true", files={"image": ("far.png", to_png(far_image()), "image/png")})
        assert far.status_code == 200
        body = far.json()
        assert body["mode"] == "far" and len(body["leaves"]) >= 3
        assert body["decision"] == "HEALTHY" and body["image_jpg_base64"]
        close = client.post("/api/v2/classify/", files={"image": ("c.png", to_png(close_image()), "image/png")})
        assert close.json()["mode"] == "close" and len(close.json()["leaves"]) == 1
        assert client.post("/api/v2/classify/", files={"image": ("x.txt", b"bad", "text/plain")}).status_code == 400
        assert client.get("/api/sync/classify/").status_code == 405                      # v1 경로는 그대로
