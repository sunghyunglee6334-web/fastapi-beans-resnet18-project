"""데이터 · 누수 · 점수 정밀 분석 (Head-only vs Full)
실행: ~/fastapi_resnet18 에서  python leak_check.py [결과복사폴더]
출력: reports/analysis/
  data_audit.json        split·클래스별 개수 · 크기 · 밝기/색 통계 · 학습코드 test 사용 여부
  leak_exact.csv         split 사이 완전 동일 파일 (md5)
  leak_phash.csv         split 사이 거의 같은 사진 (pHash 해밍거리 <= 6)
  nearest_train.csv      val/test 각 사진과 가장 닮은 train 사진 (ImageNet 특징 코사인 유사도)
  predictions.csv        train/val/test 전 사진 · 두 모델 예측 · 점수 · 정답여부
  summary.json           재계산 정확도 · 95% 신뢰구간 · 맥니마 · 유사도 구간별 정확도 · 학습 파라미터 수
"""
import csv, hashlib, json, math, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

DS = Path("dataset"); OUT = Path("reports/analysis"); OUT.mkdir(parents=True, exist_ok=True)
SPLITS = ["train", "val", "test"]; CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
MODELS = {"headonly": "artifacts/beans_resnet18", "full": "artifacts/beans_resnet18_full"}
EXT = {".jpg", ".jpeg", ".png"}
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)

items = [(s, c, p) for s in SPLITS for c in CLASSES for p in sorted((DS / s / c).glob("*")) if p.suffix.lower() in EXT]
print(f"사진 {len(items)}장 · device {dev}")

# ---------- A. 데이터 감사 ----------
audit = {"counts": defaultdict(dict), "sizes": defaultdict(Counter), "stats": defaultdict(dict)}
md5, phash, img_stats = {}, {}, defaultdict(list)
for s, c, p in items:
    b = p.read_bytes(); md5[p] = hashlib.md5(b).hexdigest()
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    h, w = im.shape[:2]; audit["sizes"][s][f"{w}x{h}"] += 1
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    d = cv2.dct(np.float32(cv2.resize(g, (32, 32))))[:8, :8].flatten()
    phash[p] = d > np.median(d)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    img_stats[(s, c)].append([g.mean(), hsv[..., 0].mean(), hsv[..., 1].mean(),
                              cv2.Laplacian(g, cv2.CV_64F).var()])
for s in SPLITS:
    for c in CLASSES:
        a = np.array(img_stats[(s, c)])
        audit["counts"][s][c] = len(a)
        audit["stats"][s][c] = {k: round(float(v), 1) for k, v in zip(["brightness", "hue", "saturation", "sharpness"], a.mean(0))}
src = Path("training/finetune.py")
audit["finetune_test_lines"] = [f"{i}: {l.strip()}" for i, l in enumerate(src.read_text(encoding="utf-8").splitlines(), 1)
                                if "test" in l.lower()] if src.exists() else "training/finetune.py 없음"

# ---------- B. 완전 중복 (md5) ----------
by_md5 = defaultdict(list)
for s, c, p in items: by_md5[md5[p]].append((s, c, p))
exact = [g for g in by_md5.values() if len({x[0] for x in g}) > 1]
with open(OUT / "leak_exact.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["md5", "files"])
    for g in exact: w.writerow([md5[g[0][2]], " | ".join(f"{s}/{c}/{p.name}" for s, c, p in g)])

# ---------- C. 거의 같은 사진 (pHash) ----------
tr = [x for x in items if x[0] == "train"]; ev = [x for x in items if x[0] != "train"]
A = np.array([phash[p] for *_, p in ev]); B = np.array([phash[p] for *_, p in tr])
ham = (A[:, None, :] != B[None, :, :]).sum(-1)
near = []
with open(OUT / "leak_phash.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["split", "class", "file", "train_class", "train_file", "hamming"])
    for i, j in zip(*np.where(ham <= 6)):
        r = [ev[i][0], ev[i][1], ev[i][2].name, tr[j][1], tr[j][2].name, int(ham[i, j])]; near.append(r); w.writerow(r)

# ---------- D. 모델 예측 + ImageNet 특징 ----------
def load(path):
    return AutoImageProcessor.from_pretrained(path), AutoModelForImageClassification.from_pretrained(path).to(dev).eval()

preds, feats = {}, None
for key, path in MODELS.items():
    proc, model = load(path); id2 = model.config.id2label
    P, F = [], []
    for k in range(0, len(items), 32):
        ims = [Image.open(p).convert("RGB") for *_, p in items[k:k + 32]]
        x = proc(images=ims, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**x, output_hidden_states=True)
            prob = out.logits.softmax(-1).cpu().numpy()
            if key == "headonly":   # 백본 고정 = ImageNet 원래 특징 → 유사도 계산용
                F.append(torch.nn.functional.normalize(model.resnet(**x).pooler_output.flatten(1), dim=1).cpu().numpy())
        P.extend([(id2[int(r.argmax())], float(r.max()), r) for r in prob])
    preds[key] = P
    if key == "headonly": feats = np.concatenate(F)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for n, p in model.named_parameters() if "classifier" in n) if key == "headonly" else total
    audit.setdefault("params", {})[key] = {"total": total, "trained": trainable}
    del model; torch.cuda.empty_cache()

ti = [i for i, x in enumerate(items) if x[0] == "train"]
sim = feats @ feats[ti].T
nn_sim = {i: (float(sim[i].max()), items[ti[int(sim[i].argmax())]]) for i, x in enumerate(items) if x[0] != "train"}
with open(OUT / "nearest_train.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["split", "class", "file", "nearest_train_class", "nearest_train_file", "cosine"])
    for i, (v, t) in sorted(nn_sim.items(), key=lambda z: -z[1][0]):
        w.writerow([items[i][0], items[i][1], items[i][2].name, t[1], t[2].name, round(v, 4)])

with open(OUT / "predictions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "true", "file", "head_pred", "head_score", "head_ok", "full_pred", "full_score", "full_ok",
                "case", "nearest_train_cos"] + [f"head_{c}" for c in CLASSES] + [f"full_{c}" for c in CLASSES])
    for i, (s, c, p) in enumerate(items):
        h, fu = preds["headonly"][i], preds["full"][i]; ho, fo = h[0] == c, fu[0] == c
        case = {(1, 1): "both_ok", (0, 1): "full_only", (1, 0): "head_only", (0, 0): "both_wrong"}[(ho, fo)]
        w.writerow([s, c, p.name, h[0], round(h[1], 4), int(ho), fu[0], round(fu[1], 4), int(fo), case,
                    round(nn_sim[i][0], 4) if i in nn_sim else ""] + [round(float(v), 4) for v in h[2]] + [round(float(v), 4) for v in fu[2]])

# ---------- E. 요약 ----------
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)

summary = {"leak_exact_pairs": len(exact), "leak_phash_pairs(<=6)": len(near), "splits": {}}
for s in SPLITS:
    idx = [i for i, x in enumerate(items) if x[0] == s]
    ok = {k: np.array([preds[k][i][0] == items[i][1] for i in idx]) for k in MODELS}
    d = {"n": len(idx)}
    for k in MODELS:
        boots = [ok[k][rng.integers(0, len(idx), len(idx))].mean() for _ in range(2000)]
        d[k] = {"accuracy": round(float(ok[k].mean()), 4), "correct": int(ok[k].sum()),
                "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                "mean_score": round(float(np.mean([preds[k][i][1] for i in idx])), 4),
                "mean_score_when_wrong": round(float(np.mean([preds[k][i][1] for i, o in zip(idx, ok[k]) if not o] or [0])), 4)}
    b = int((ok["headonly"] & ~ok["full"]).sum()); c = int((~ok["headonly"] & ok["full"]).sum())
    d["cases"] = {"both_ok": int((ok["headonly"] & ok["full"]).sum()), "full_only_ok": c, "head_only_ok": b,
                  "both_wrong": int((~ok["headonly"] & ~ok["full"]).sum()), "mcnemar_p": mcnemar(b, c)}
    if s != "train":
        sims = np.array([nn_sim[i][0] for i in idx]); qs = np.quantile(sims, [0, .25, .5, .75, 1])
        d["nearest_train_cos"] = {"min": round(float(sims.min()), 4), "median": round(float(np.median(sims)), 4),
                                  "max": round(float(sims.max()), 4), ">=0.95": int((sims >= 0.95).sum())}
        d["acc_by_similarity_quartile"] = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (sims >= lo) & (sims <= hi)
            d["acc_by_similarity_quartile"].append({"cos": f"{lo:.3f}~{hi:.3f}", "n": int(m.sum()),
                "headonly": round(float(ok["headonly"][m].mean()), 3), "full": round(float(ok["full"][m].mean()), 3)})
    summary["splits"][s] = d

audit["sizes"] = {k: dict(v.most_common(5)) for k, v in audit["sizes"].items()}
json.dump(audit, open(OUT / "data_audit.json", "w"), ensure_ascii=False, indent=2, default=str)
json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=1))
if len(sys.argv) > 1:
    dst = Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True)
    for f in OUT.iterdir(): shutil.copy(f, dst / f.name)
    print("복사:", dst)
