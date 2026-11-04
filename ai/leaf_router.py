"""사진 상태를 보고 길을 나누는 라우터 (OpenCV 규칙 기반 · 학습 모델 없음)

    입력 사진
      ↓
    ① 품질 게이트  흐림 · 어두움 · 밝음 · 대비 부족 → RETAKE (재촬영)
      ↓
    ② 잎 색 마스크  HSV 초록 ~ 노랑 (흙 · 하늘 제외) · 반점 구멍 메움
      ↓
    ③ 잎 분리      MeanShift 로 잎맥 지우기 → 긴 경계선만 남겨 잎끼리 떼기 → 연결 영역
      ↓
    ④ 길 나누기    가장 큰 잎이 화면을 크게 차지하면 close (근접 · 잎 1장)
                   비슷한 크기의 잎이 여러 장 + 속이 꽉 찬 잎 모양 조각이 있으면 far (원거리 · 밭 사진)
                   또는 잎이 빽빽한 밭 사진: 속이 꽉 찬 작은 잎(화면 1.5% 이상)이 SCENE_FAR_SMALL_LEAVES 장 이상
                   (잎맥 때문에 쪼개진 조각은 쐐기 모양 = solidity 낮음 → 원거리로 보지 않음)
"""
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .config import (SCENE_CLOSE_HULL, SCENE_FAR_SIMILAR, SCENE_MAX_LEAVES, SCENE_LEAF_SOLIDITY, SCENE_FAR_SMALL_LEAVES,
                     SCENE_MIN_BRIGHT, SCENE_MAX_BRIGHT, SCENE_MIN_CONTRAST, SCENE_MIN_SHARPNESS)

WORK_SIZE = 512            # 계산용 긴 변 (원본 크기와 상관없이 같은 기준으로 잼)
HUE_RANGE = (18, 95)       # OpenCV Hue 0 ~ 180 · 노랑(병든 잎) ~ 초록
MIN_SAT, MIN_VAL = 45, 45  # 회색 · 그림자 제외
MIN_LEAF = 0.012           # 화면의 1.2% 보다 작은 조각은 잎으로 안 봄
MIN_CROP = 0.02            # 원거리에서 분류할 잎의 최소 크기 (볼록 껍질 기준 2% · 가장 큰 잎의 30% 이상)
SMALL_LEAF = 0.015         # 빽빽한 밭 사진에서 잎 1장으로 셀 최소 크기 (속이 꽉 찬 모양일 때)
PAD = 0.08                 # 잎을 잘라낼 때 여백 8%


@dataclass
class Quality:
    sharpness: float       # Laplacian 분산 · 낮으면 흐림
    brightness: float      # 회색 평균 0 ~ 255
    contrast: float        # 회색 표준편차
    ok: bool
    reasons: list = field(default_factory=list)


@dataclass
class Scene:
    mode: str                                   # close | far | no_leaf
    leaf_ratio: float                           # 잎 색 영역 비율
    largest_hull: float                         # 가장 큰 잎(구멍 메운 모양) 비율
    similar: int                                # 가장 큰 잎의 절반 이상 크기인 잎 수
    whole_leaves: int = 0                       # 속이 꽉 찬 잎 모양 조각 수 (화면 8% 이상 · solidity 기준 이상)
    boxes: list = field(default_factory=list)   # 원본 좌표 (x1, y1, x2, y2) · 큰 잎부터
    hulls: list = field(default_factory=list)   # 잎마다 화면 비율

    def to_dict(self):
        return asdict(self)


def _resize(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    scale = WORK_SIZE / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2BGR), scale


def check_quality(rgb: np.ndarray) -> Quality:
    bgr, _ = _resize(rgb)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright, contrast = float(gray.mean()), float(gray.std())
    reasons = []
    if sharp < SCENE_MIN_SHARPNESS:
        reasons.append("blur")
    if bright < SCENE_MIN_BRIGHT:
        reasons.append("dark")
    if bright > SCENE_MAX_BRIGHT:
        reasons.append("bright")
    if contrast < SCENE_MIN_CONTRAST:
        reasons.append("low_contrast")
    return Quality(round(sharp, 1), round(bright, 1), round(contrast, 1), not reasons, reasons)


def _leaf_mask(bgr):
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    mask = ((h >= HUE_RANGE[0]) & (h <= HUE_RANGE[1]) & (s >= MIN_SAT) & (v >= MIN_VAL)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)                 # 작은 흰 점 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)  # 작은 틈 메움
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:                                               # 병 반점 구멍 메움
        if cv2.contourArea(c) < 0.01 * mask.size:
            cv2.drawContours(mask, [c], -1, 255, -1)
    return mask


def _long_edges(bgr):
    """MeanShift 로 잎맥 · 반점을 뭉갠 뒤 긴 경계선(잎과 잎 사이)만 남긴다."""
    smooth = cv2.pyrMeanShiftFiltering(bgr, 12, 28)
    edges = cv2.Canny(cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY), 25, 70)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(edges)
    min_len = int(0.12 * max(bgr.shape[:2]))
    keep = np.zeros_like(edges)
    for i in range(1, n):
        if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= min_len:
            keep[lab == i] = 255
    return keep


def analyze_scene(rgb: np.ndarray) -> Scene:
    """rgb = H × W × 3 uint8 (np.asarray(PIL 이미지))."""
    bgr, scale = _resize(rgb)
    area = bgr.shape[0] * bgr.shape[1]
    mask = _leaf_mask(bgr)
    leaf_ratio = float((mask > 0).sum()) / area
    if leaf_ratio < MIN_LEAF * 2:
        return Scene("no_leaf", round(leaf_ratio, 3), 0.0, 0, 0)

    core = cv2.bitwise_and(mask, cv2.bitwise_not(_long_edges(bgr)))
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels = cv2.connectedComponents(core)
    segs = []
    for i in range(1, n):
        part = (labels == i).astype(np.uint8)
        if part.sum() / area < MIN_LEAF:
            continue
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        hull_area = cv2.contourArea(hull)
        solidity = part.sum() / hull_area if hull_area else 0.0     # 조각 넓이 / 볼록 껍질 넓이
        segs.append((hull_area / area, cv2.boundingRect(hull), solidity))
    if not segs:
        segs = [(leaf_ratio, cv2.boundingRect(mask), 1.0)]
    segs.sort(key=lambda s: s[0], reverse=True)

    largest = segs[0][0]
    similar = sum(1 for r, _, _ in segs if r >= 0.5 * largest)
    whole = sum(1 for r, _, sol in segs if r >= 0.08 and sol >= SCENE_LEAF_SOLIDITY)
    small_whole = sum(1 for r, _, sol in segs if r >= SMALL_LEAF and sol >= SCENE_LEAF_SOLIDITY)
    spread = largest < SCENE_CLOSE_HULL and similar >= SCENE_FAR_SIMILAR and whole >= 1        # 잎 여러 장
    dense = largest < SCENE_CLOSE_HULL and small_whole >= SCENE_FAR_SMALL_LEAVES and leaf_ratio >= 0.6  # 빽빽한 밭
    mode = "far" if (spread or dense) else "close"

    def keep(r, sol):  # 원거리에서 잘라 분류할 조각: 큰 조각 또는 속이 꽉 찬 작은 잎
        return r >= max(MIN_CROP, 0.3 * largest) or (r >= SMALL_LEAF and sol >= SCENE_LEAF_SOLIDITY)

    boxes, hulls = [], []
    for r, (x, y, w, h), sol in segs:
        if len(boxes) >= SCENE_MAX_LEAVES:
            break
        if boxes and not keep(r, sol):
            continue
        px, py = int(w * PAD), int(h * PAD)
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(bgr.shape[1], x + w + px), min(bgr.shape[0], y + h + py)
        boxes.append((int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)))
        hulls.append(round(r, 3))
    return Scene(mode, round(leaf_ratio, 3), round(largest, 3), similar, whole, boxes, hulls)


def _label_color(label: str | None):
    """잎 결과 색 (BGR) · healthy 초록 · 병 빨강 · 애매(?) 주황 · 결과 없음 파랑."""
    if not label:
        return (230, 130, 30)
    if label.rstrip().endswith("?"):
        return (0, 150, 255)
    return (60, 170, 60) if label.startswith("healthy") else (40, 40, 220)


def draw_scene(rgb: np.ndarray, scene: Scene, labels=None, quality: Quality | None = None) -> np.ndarray:
    """판별 결과를 그린 BGR 영상 (화면 · 증빙용).

    박스가 겹쳐도 읽히도록 사진에는 **번호 동그라미만** 그리고, 점수는 화면의 표 (leaves) 에서 본다.
    색 = 결과 (healthy 초록 · 병 빨강 · 애매 주황). labels = 잎마다 "bean_rust 0.91" 같은 글자.
    """
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = out.shape[:2]
    t = max(2, round(max(h, w) / 400))
    boxes = scene.boxes if scene.mode == "far" else scene.boxes[:1]
    colors = [_label_color(labels[i] if labels and i < len(labels) else None) for i in range(len(boxes))]
    for (x1, y1, x2, y2), c in zip(boxes, colors):
        cv2.rectangle(out, (x1, y1), (x2, y2), c, t)

    r = max(14, round(max(h, w) / 45))                  # 번호 동그라미 반지름
    placed = []
    for i, ((x1, y1, x2, y2), c) in enumerate(zip(boxes, colors)):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2          # 박스 가운데에서 시작
        for step in range(40):                           # 다른 번호와 겹치면 조금씩 옮김
            if all((cx - px) ** 2 + (cy - py) ** 2 >= (2.3 * r) ** 2 for px, py in placed):
                break
            cx += round(1.2 * r) if step % 2 == 0 else 0
            cy += round(1.2 * r) if step % 2 == 1 else 0
            if cx > x2 - r or cx > w - r:
                cx = max(x1 + r, r)
            if cy > y2 - r or cy > h - r:
                cy = max(y1 + r, r)
        placed.append((cx, cy))
        cv2.circle(out, (cx, cy), r, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), r, c, max(2, t))
        text = str(i + 1)
        fs = r / 22
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(2, t))
        cv2.putText(out, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, c, max(2, t))

    head = f"{scene.mode.upper()}  leaves {len(boxes)}"
    if quality is not None and not quality.ok:
        head = "RETAKE " + ",".join(quality.reasons) + "  |  " + head
    bar = max(24, round(h / 22))
    cv2.rectangle(out, (0, 0), (w, bar), (0, 0, 0), -1)
    cv2.putText(out, head, (8, round(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, bar / 40, (255, 255, 255), max(1, t // 2))
    return out
