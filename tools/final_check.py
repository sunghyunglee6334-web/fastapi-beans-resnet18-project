"""최종 점검 · 모델 4개를 같은 Validation 으로 비교 → Val 1등을 최종 모델로 선택 → 그 모델만 Test 1회
실행: ~/fastapi_resnet18 에서  python tools/final_check.py [외부사진폴더]
출력:
  reports/data/final_models.json   모델별 Val (학습 기록 + 다시 계산) · 선택 모델 · 선택 모델 Test 1회
  reports/data/v2_external.json    외부 사진: v1 (사진 전체 1번 분류 · 모델별) vs v2 (장면 분석 · 선택 모델)
  reports/scene/final/*.jpg        외부 사진 v2 박스 그림 (Git 제외 · 발표 화면용)
규칙: 모델 선택은 Validation 으로만 · Test 는 선택이 끝난 뒤 선택 모델 1개만 1회 측정 (평가 개정본)
"""
import json, os, sys, types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import dotenv  # noqa: F401
except ImportError:                       # 학습용 .venv 에 python-dotenv 가 없어도 돌도록
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)

DS = Path("dataset"); OUT = Path("reports/data"); OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
MODELS = {
    "headonly":        ("artifacts/beans_resnet18",                "Head-only (분류층만)"),
    "full":            ("artifacts/beans_resnet18_full",           "Full (전체 학습)"),
    "full_aug_strong": ("artifacts/beans_resnet18_full_aug",       "Full + 강한 증강"),
    "full_aug_basic":  ("artifacts/beans_resnet18_full_aug_basic", "Full + 기본 증강 (회전15·반전·밝기)"),
}
EXT = {".jpg", ".jpeg", ".png", ".webp"}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def files(split):
    return [(c, p) for c in CLASSES for p in sorted((DS / split / c).glob("*")) if p.suffix.lower() in EXT]


def predict(path, images):
    proc = AutoImageProcessor.from_pretrained(path)
    model = AutoModelForImageClassification.from_pretrained(path).to(dev).eval()
    id2 = model.config.id2label; out = []
    for k in range(0, len(images), 32):
        x = proc(images=[im.convert("RGB") for im in images[k:k + 32]], return_tensors="pt").to(dev)
        with torch.no_grad():
            prob = model(**x).logits.softmax(-1).cpu().numpy()
        out += [{id2[i]: float(r[i]) for i in range(len(r))} for r in prob]
    del model; torch.cuda.empty_cache()
    return out


def metrics(items, probs):
    pred = [max(p, key=p.get) for p in probs]; true = [c for c, _ in items]
    ok = np.array([a == b for a, b in zip(pred, true)])
    rng = np.random.default_rng(0)
    boot = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(2000)]
    cm = [[sum(1 for t, q in zip(true, pred) if t == a and q == b) for b in CLASSES] for a in CLASSES]
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i][i]; fp = sum(cm[r][i] for r in range(3)) - tp; fn = sum(cm[i]) - tp
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        per[c] = {"precision": round(pr, 4), "recall": round(rc, 4),
                  "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0, "n": sum(cm[i])}
    top = np.array([max(p.values()) for p in probs])
    return {"count": len(ok), "correct": int(ok.sum()), "accuracy": round(float(ok.mean()), 4),
            "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
            "mean_top_score": round(float(top.mean()), 4), "per_class": per,
            "confusion_matrix": {"rows=actual, cols=pred": CLASSES, "values": cm}}


# ---------- 1. Validation 으로 4개 비교 (같은 133장 · 증강 없음) ----------
val = files("val"); val_img = [Image.open(p) for _, p in val]
result = {"rule": "모델 선택 = Validation 1등 · Test 는 선택 뒤 선택 모델만 1회", "val_count": len(val), "models": {}}
for key, (path, name) in MODELS.items():
    if not Path(path).exists():
        print(f"[없음] {path} · 건너뜀"); continue
    rep = Path(path) / "training_report.json"
    r = json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else {}
    m = metrics(val, predict(path, val_img))
    result["models"][key] = {"path": path, "name": name, "trained_params": 1539 if key == "headonly" else 11178051,
                             "augment": r.get("aug_mode") or ("strong" if key == "full_aug_strong" else None),
                             "epochs": r.get("epochs", 3), "history": r.get("history"),
                             "val_report": round(r["best_val_accuracy"], 4) if "best_val_accuracy" in r else None,
                             "val": {k: m[k] for k in ("correct", "accuracy", "ci95", "mean_top_score")}}
    print(f"{key:16s} Val {m['correct']}/{m['count']} = {m['accuracy']:.4f}  평균 점수 {m['mean_top_score']:.3f}")

order = list(result["models"])
best = max(order, key=lambda k: (result["models"][k]["val"]["accuracy"], result["models"][k]["val"]["mean_top_score"]))
result["chosen"] = best
print(f"\n▶ Val 1등 = 최종 모델: {best} ({MODELS[best][0]})")

# ---------- 2. 선택 모델만 Test 1회 ----------
test = files("test")
tm = metrics(test, predict(MODELS[best][0], [Image.open(p) for _, p in test]))
result["chosen_test"] = tm
print(f"  Test {tm['correct']}/{tm['count']} = {tm['accuracy']:.4f}  (95% 신뢰구간 {tm['ci95'][0]}~{tm['ci95'][1]})")
(OUT / "final_models.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 3. 외부 사진 · v1 (사진 전체) vs v2 (장면 분석) ----------
ext = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/c/Users/user1/Downloads/교과9_평가/외부사진")
photos = sorted(p for p in ext.glob("*") if p.suffix.lower() in EXT) if ext.exists() else []
if photos:
    os.environ["MODEL_PATH"] = MODELS[best][0]              # v2 는 서비스와 같은 코드 (scene_service) 로 실행
    from ai.scene_service import classify_scene
    import base64
    pics = Path("reports/scene/final"); pics.mkdir(parents=True, exist_ok=True)
    v1 = {k: predict(MODELS[k][0], [Image.open(p) for p in photos]) for k in result["models"]}
    rows = []
    for i, p in enumerate(photos):
        s = classify_scene(p.read_bytes(), draw=True)
        (pics / (p.stem + ".jpg")).write_bytes(base64.b64decode(s.pop("image_jpg_base64") or b""))
        leaves = s["leaves"]
        avg = {c: round(float(np.mean([next(x["score"] for x in l["all"] if x["label"] == c) for l in leaves])), 4)
               for c in CLASSES} if leaves else {}
        rows.append({"file": p.name,
                     "v1_whole_image": {k: {"label": max(v1[k][i], key=v1[k][i].get),
                                            "score": round(max(v1[k][i].values()), 4)} for k in v1},
                     "v2": {"model": best, "mode": s["mode"], "decision": s["decision"], "counts": s["counts"],
                            "leaves": [{"no": l["no"], "label": l["label"], "score": l["score"], "check": l["check"]} for l in leaves],
                            "average": avg}})
        r = rows[-1]
        print(f"\n[{p.name}]  v1 " + " · ".join(f"{k} {v['label']} {v['score']:.2f}" for k, v in r["v1_whole_image"].items()))
        print(f"  v2 {s['mode']} · 잎 {len(leaves)}장 · 판정 {s['decision']} · {s['counts']} · 평균 {avg}")
    (OUT / "v2_external.json").write_text(json.dumps({"folder": "외부사진 (인터넷 사진 · 원본은 Git 제외)", "photos": rows},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
print("\n저장: reports/data/final_models.json · reports/data/v2_external.json · reports/scene/final/")
