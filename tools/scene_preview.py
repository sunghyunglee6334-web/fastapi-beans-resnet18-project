"""폴더 안 사진을 근접 / 원거리로 판별해 그림 · CSV 로 저장 (모델 없이 라우터만 · 기준값 맞추기용)

실행 (프로젝트 폴더에서):
    python tools/scene_preview.py robust/variants                  # 인터넷 사진 19장
    python tools/scene_preview.py dataset/val --limit 60           # 원래 데이터 → 대부분 close 여야 정상
출력:
    reports/scene/<폴더이름>/*.jpg     판별 결과 그림
    reports/scene/<폴더이름>.csv       파일별 mode · 품질 · 잎 수
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image

from ai.leaf_router import analyze_scene, check_quality, draw_scene

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--limit", type=int, default=0, help="폴더마다 앞에서부터 N장만 (0 = 전부)")
    args = ap.parse_args()
    src = Path(args.folder)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXT)
    if args.limit:
        by_dir = {}
        for p in files:
            by_dir.setdefault(p.parent, []).append(p)
        files = [p for ps in by_dir.values() for p in ps[:args.limit]]
    name = "_".join(src.parts[-2:]) if len(src.parts) > 1 else src.name
    out_dir = Path("reports/scene") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, modes = [], Counter()
    for p in files:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        q, s = check_quality(rgb), analyze_scene(rgb)
        route = "retake" if not q.ok else s.mode
        modes[route] += 1
        rows.append({"file": str(p.relative_to(src)), "route": route, "mode": s.mode, "leaves": len(s.boxes),
                     "largest_hull": s.largest_hull, "similar": s.similar, "whole_leaves": s.whole_leaves, "leaf_ratio": s.leaf_ratio,
                     "sharpness": q.sharpness, "brightness": q.brightness, "contrast": q.contrast,
                     "retake_reason": ",".join(q.reasons)})
        cv2.imwrite(str(out_dir / (p.stem + ".jpg")), draw_scene(rgb, s, quality=q))
    with open(out_dir.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{src} · {len(rows)}장 · " + " · ".join(f"{k} {v}" for k, v in modes.most_common()))
    for r in rows[:40]:
        print(f"  {r['file']:42} {r['route']:7} leaves {r['leaves']:2}  hull {r['largest_hull']:.2f}  similar {r['similar']}  whole {r['whole_leaves']}  {r['retake_reason']}")
    print(f"저장: {out_dir}/ · {out_dir.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
