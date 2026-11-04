"""새 콩잎 사진 1장 → 병해 합성 · 사진 효과 · 기하 변형 18장 (교과 9 전처리 · 강건성 테스트용)"""
import sys, cv2, numpy as np
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
img = cv2.imread(str(src)); h, w = img.shape[:2]
rng = np.random.default_rng(42)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
leaf = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))           # 잎(초록) 마스크
leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
ys, xs = np.where(leaf > 0)
def pts(n): i = rng.choice(len(xs), n, replace=False); return list(zip(xs[i], ys[i]))
V = {}
V["00_original"] = img
# 1 각진잎반점병 느낌: 잎맥에 막힌 각진 갈색 반점
a = img.copy()
for x, y in pts(140):
    r = int(rng.integers(6, 13)); poly = np.array([[x + rng.integers(-r, r), y + rng.integers(-r, r)] for _ in range(6)], np.int32)
    cv2.fillPoly(a, [poly], (40, 150, 175)); cv2.fillPoly(a, [((poly - [x, y]) * 0.7 + [x, y]).astype(np.int32)], (25, 45, 85))
V["01_disease_angular_spots"] = np.where(leaf[..., None] > 0, a, img)
# 2 녹병 느낌: 작은 주황 · 갈색 돌기 + 노란 테두리
b = img.copy()
for x, y in pts(260):
    r = int(rng.integers(2, 4))
    cv2.circle(b, (int(x), int(y)), r + 2, (60, 200, 210), -1)
    cv2.circle(b, (int(x), int(y)), r, (20, 70, 160), -1)
V["02_disease_rust_pustules"] = np.where(leaf[..., None] > 0, cv2.GaussianBlur(b, (3, 3), 0), img)
# 3 갈변: 잎 가장자리부터 마름
big = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
dist = cv2.distanceTransform(big, cv2.DIST_L2, 5); edge = cv2.GaussianBlur(np.clip(1 - dist / 30, 0, 1), (9, 9), 0)[..., None] * (big[..., None] > 0)
brown = (img * 0.35 + np.array([35, 70, 115]) * 0.65)
V["03_browning_edge"] = np.clip(img * (1 - edge) + brown * edge, 0, 255).astype(np.uint8)
# 4 전체 황변 · 갈변
y4 = hsv.copy(); y4[..., 0] = np.where(leaf > 0, np.clip(y4[..., 0] - 22, 0, 179), y4[..., 0]); y4[..., 1] = (y4[..., 1] * 0.8).astype(np.uint8)
V["04_browning_yellow"] = cv2.cvtColor(y4, cv2.COLOR_HSV2BGR)
# 사진 효과
V["05_blur_gaussian"] = cv2.GaussianBlur(img, (21, 21), 0)
k = np.zeros((25, 25)); k[12, :] = 1 / 25; V["06_blur_motion"] = cv2.filter2D(img, -1, k)
V["07_noise"] = np.clip(img + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)
V["08_dark"] = cv2.convertScaleAbs(img, alpha=0.45, beta=0)
V["09_bright"] = cv2.convertScaleAbs(img, alpha=1.0, beta=90)
V["10_low_contrast"] = cv2.convertScaleAbs(img, alpha=0.4, beta=80)
V["11_jpeg_q5"] = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 5])[1], 1)
V["12_lowres_32px"] = cv2.resize(cv2.resize(img, (32, 48)), (w, h), interpolation=cv2.INTER_NEAREST)
# 기하 변형
M = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1); V["13_rotate_25"] = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
V["14_rotate_90"] = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
V["15_flip"] = cv2.flip(img, 1)
P = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32([[w * .2, h * .05], [w * .8, 0], [w, h], [0, h * .95]]))
V["16_perspective_tilt"] = cv2.warpPerspective(img, P, (w, h), borderMode=cv2.BORDER_REFLECT)
V["17_crop_zoom"] = cv2.resize(img[int(h * .3):int(h * .7), int(w * .2):int(w * .8)], (w, h))
V["18_occlusion"] = img.copy(); cv2.rectangle(V["18_occlusion"], (int(w * .25), int(h * .3)), (int(w * .75), int(h * .6)), (200, 190, 230), -1)
for k_, v in V.items():
    cv2.imwrite(str(out / f"{k_}.jpg"), v, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"{len(V)}장 저장 → {out}")
