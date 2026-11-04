"""성능 측정: 동기/비동기 × MISS/HIT 각 5회 → reports/data/timings.csv"""
import csv, statistics, time
from pathlib import Path
import httpx

BASE = "http://127.0.0.1:8000"
imgs = sorted(Path("dataset/test/bean_rust").glob("*.jpg"))
c = httpx.Client(timeout=120)
rows = []

def post(url, p):
    t = time.perf_counter()
    r = c.post(BASE + url, files={"image": (p.name, p.read_bytes(), "image/jpeg")})
    return r, (time.perf_counter() - t) * 1000, t

def sync(p, run):
    r, ms, _ = post("/api/sync/classify/", p)
    j = r.json()
    rows.append([run, "sync", "HIT" if j["cache_hit"] else "MISS", round(ms, 1), round(ms, 1), j["inference_ms"], "", p.name])

def asyn(p, run):
    r, acc, t0 = post("/api/async/classify/", p)
    j = r.json()
    while True:
        s = c.get(BASE + j["check_url"]).json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    done = (time.perf_counter() - t0) * 1000
    rows.append([run, "async", "HIT" if s["cache_hit"] else "MISS", round(acc, 1), round(done, 1), s["inference_ms"], j["job_id"], p.name])

for i, p in enumerate(imgs[10:15], 1): sync(p, i)        # 동기 MISS: 다른 사진 5장
for i in range(1, 6): sync(imgs[10], i)                  # 동기 HIT: 같은 사진 5번
for i, p in enumerate(imgs[20:25], 1): asyn(p, i)        # 비동기 MISS: 다른 사진 5장
for i in range(1, 6): asyn(imgs[20], i)                  # 비동기 HIT: 같은 사진 5번

out = Path("reports/data/timings.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "path", "cache_status", "http_ms", "final_ms", "inference_ms", "job_id", "image"])
    w.writerows(rows)

print("run path  cache  http_ms  final_ms  inference_ms  job_id")
for r in rows:
    print(*r[:7])
print("\n[중앙값]")
for path in ("sync", "async"):
    for cs in ("MISS", "HIT"):
        g = [r for r in rows if r[1] == path and r[2] == cs]
        if g:
            inf = [r[5] for r in g if r[5] is not None]
            print(f"{path:5} {cs:4} n={len(g)}  HTTP {statistics.median(r[3] for r in g):.1f}ms  "
                  f"최종 {statistics.median(r[4] for r in g):.1f}ms  추론 {statistics.median(inf):.1f}ms" if inf else
                  f"{path:5} {cs:4} n={len(g)}  HTTP {statistics.median(r[3] for r in g):.1f}ms  "
                  f"최종 {statistics.median(r[4] for r in g):.1f}ms  추론 null")
print(f"\n저장: {out}")
