"""v2 장면 분석 서비스 · 사진 → 품질 게이트 → 근접/원거리 판별 → 잎별 분류 → 종합 판정

    close (근접 · 잎 1장)  : 사진 전체를 그대로 공통 모델에 (지금 v1 과 같은 입력)
    far   (원거리 · 밭)     : 잎마다 잘라서 공통 모델에 → 잎별 결과
    판정 순서 (불량을 놓치지 않는 쪽 우선):
        확실한 병 잎이 1장이라도 있음 → DISEASE
        애매한 잎이 있음 (점수 < SCENE_CHECK_SCORE) → CHECK (사람 확인)
        전부 확실한 healthy → HEALTHY
"""
import base64
import time

import cv2
import numpy as np

from . import services
from .config import MODEL_PATH, SCENE_CHECK_SCORE
from .leaf_router import analyze_scene, check_quality, draw_scene

HEALTHY = "healthy"


def _predict(images):
    """공통 모델 (v1 과 같은 get_classifier) 로 여러 장을 한 번에 분류."""
    raw = services.get_classifier()(images, top_k=3)
    if images and isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = [raw]                                     # 1장일 때 리스트가 한 겹 덜 감싸진 경우
    results = []
    for item in raw:
        items = sorted(({"label": str(x["label"]), "score": float(x["score"])} for x in item),
                       key=lambda x: x["score"], reverse=True)
        results.append(items)
    return results


def _decide(leaves):
    sure_sick = [x for x in leaves if not x["check"] and x["label"] != HEALTHY]
    if sure_sick:
        return "DISEASE"
    if any(x["check"] for x in leaves):
        return "CHECK"
    return "HEALTHY"


def classify_scene(image_bytes: bytes, draw: bool = False) -> dict:
    image = services.load_rgb_image(image_bytes)        # v1 과 같은 검증 (형식 · 크기)
    rgb = np.asarray(image)
    started = time.perf_counter()
    quality = check_quality(rgb)
    scene = analyze_scene(rgb)
    result = {"model_id": MODEL_PATH, "quality": quality.__dict__, "scene": scene.to_dict(),
              "mode": scene.mode, "leaves": [], "decision": None, "inference_ms": None}

    if not quality.ok:
        result["decision"] = "RETAKE"
    elif scene.mode == "no_leaf":
        result["decision"] = "NO_LEAF"
    else:
        if scene.mode == "close":
            crops, boxes = [image], [scene.boxes[0] if scene.boxes else (0, 0, image.width, image.height)]
        else:
            boxes = scene.boxes
            crops = [image.crop(b) for b in boxes]
        t0 = time.perf_counter()
        preds = _predict(crops)
        result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        for i, (box, pred) in enumerate(zip(boxes, preds), start=1):
            top = pred[0]
            result["leaves"].append({"no": i, "box": list(box), "label": top["label"], "score": round(top["score"], 4),
                                     "check": top["score"] < SCENE_CHECK_SCORE, "all": pred})
        result["decision"] = _decide(result["leaves"])

    counts = {}
    for leaf in result["leaves"]:
        key = "CHECK" if leaf["check"] else leaf["label"]
        counts[key] = counts.get(key, 0) + 1
    result["counts"] = counts
    result["total_ms"] = round((time.perf_counter() - started) * 1000, 2)

    if draw:
        labels = [f'{x["label"]} {x["score"]:.2f}' + (" ?" if x["check"] else "") for x in result["leaves"]]
        canvas = draw_scene(rgb, scene, labels, quality)
        if max(canvas.shape[:2]) > 900:
            s = 900 / max(canvas.shape[:2])
            canvas = cv2.resize(canvas, (round(canvas.shape[1] * s), round(canvas.shape[0] * s)))
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result["image_jpg_base64"] = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    return result
