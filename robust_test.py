"""① Test 128장 정확도 (학습 · 선택에 쓰지 않은 데이터 · 1회 측정)
② 새 콩잎 사진 변형(병해 합성 · 사진 효과 · 기하 변형) 강건성 테스트
모델: Head-only(서비스) · Full Fine-tuning(선택 실험) 두 개를 같은 조건으로 비교
사용: python robust_test.py  (프로젝트 폴더 · .venv 에서)"""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import pipeline

MODELS = {"headonly": "artifacts/beans_resnet18", "full": "artifacts/beans_resnet18_full"}
MODELS = {k: v for k, v in MODELS.items() if Path(v, "config.json").exists()}
TEST = Path("dataset/test"); VAR = Path("robust/variants")
DATA = Path("reports/data"); FIG = Path("reports/figures"); DATA.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
dev = 0 if torch.cuda.is_available() else -1
clf = {k: pipeline("image-classification", model=v, device=dev) for k, v in MODELS.items()}
print("모델:", MODELS, "· device:", "GPU" if dev == 0 else "CPU")

def predict(name, path):
    with Image.open(path) as im:
        out = clf[name](im.convert("RGB"), top_k=3)
    return out[0]["label"], float(out[0]["score"]), {o["label"]: round(float(o["score"]), 4) for o in out}

# ① Test 정확도
files = [(p, c) for c in CLASSES for p in sorted((TEST / c).iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
metrics = {}
for name in clf:
    cm = np.zeros((3, 3), int)
    for p, c in files:
        cm[CLASSES.index(c), CLASSES.index(predict(name, p)[0])] += 1
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; prec = tp / max(cm[:, i].sum(), 1); rec = tp / max(cm[i].sum(), 1)
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "n": int(cm[i].sum())}
    metrics[name] = {"model": MODELS[name], "test_count": len(files), "accuracy": round(float(np.trace(cm) / cm.sum()), 4),
                     "per_class": per, "confusion_matrix": {"rows=actual, cols=pred": CLASSES, "values": cm.tolist()}}
    print(f"[Test] {name:8} accuracy {metrics[name]['accuracy']:.4f} ({np.trace(cm)}/{cm.sum()})")
(DATA / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

# ② 변형 사진
rows = []
for p in sorted(VAR.glob("*.jpg")):
    r = {"image": p.stem}
    for name in clf:
        lab, sc, allp = predict(name, p)
        r[f"{name}_top1"] = lab; r[f"{name}_score"] = round(sc, 4)
        for c in CLASSES: r[f"{name}_{c}"] = allp.get(c, 0)
    rows.append(r)
    print(f"[변형] {p.stem:28} " + " | ".join(f"{n}: {r[f'{n}_top1']} {r[f'{n}_score']:.3f}" for n in clf))
with (DATA / "robust_results.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# 결과 그림 (영문 라벨)
tiles = []
for r in rows:
    im = cv2.resize(cv2.imread(str(VAR / f"{r['image']}.jpg")), (220, 330))
    band = np.full((86, 220, 3), 255, np.uint8)
    cv2.putText(band, r["image"][:26], (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    for i, n in enumerate(clf):
        txt = f"{'H' if n == 'headonly' else 'F'}: {r[f'{n}_top1'][:14]} {r[f'{n}_score']:.2f}"
        col = {"healthy": (60, 140, 40), "bean_rust": (30, 90, 200), "angular_leaf_spot": (150, 60, 140)}[r[f"{n}_top1"]]
        cv2.putText(band, txt, (5, 42 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
    tiles.append(np.vstack([im, band]))
while len(tiles) % 5: tiles.append(np.full_like(tiles[0], 255))
sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
cv2.imwrite(str(FIG / "robust_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"\n저장: {DATA/'test_metrics.json'} · {DATA/'robust_results.csv'} · {FIG/'robust_sheet.jpg'}")
