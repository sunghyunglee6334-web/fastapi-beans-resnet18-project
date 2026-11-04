"""reports/data/timings.csv → reports/figures/perf_compare.png (중앙값 비교 그래프)"""
import csv, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("reports/data/timings.csv", encoding="utf-8")))
groups = [("sync", "MISS", "Sync MISS"), ("sync", "HIT", "Sync HIT"),
          ("async", "MISS", "Async MISS"), ("async", "HIT", "Async HIT")]
http, final = [], []
for p, c, _ in groups:
    g = [r for r in rows if r["path"] == p and r["cache_status"] == c]
    http.append(statistics.median(float(r["http_ms"]) for r in g))
    final.append(statistics.median(float(r["final_ms"]) for r in g))

BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
y = list(range(len(groups)))
h = 0.36
b1 = ax.barh([i - h / 2 - 0.01 for i in y], http, height=h, color=BLUE, label="HTTP response (sync = result / async = 202 accepted)")
b2 = ax.barh([i + h / 2 + 0.01 for i in y], final, height=h, color=ORANGE, label="Final result confirmed (async includes 0.2 s polling)")
for bars in (b1, b2):
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 3, bar.get_y() + bar.get_height() / 2, f"{w:.1f} ms", va="center", fontsize=9, color=INK)
ax.set_yticks(y, [g[2] for g in groups], fontsize=10, color=INK)
ax.invert_yaxis()
ax.set_xlabel("Median time, ms (5 runs each)", color=MUTED, fontsize=9)
ax.set_xlim(0, max(final) * 1.18)
ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", colors=MUTED, labelsize=8); ax.tick_params(axis="y", length=0)
ax.set_title("ResNet18 beans API - sync vs async, cache MISS vs HIT", loc="left", fontsize=12, color=INK, pad=44)
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.15), ncol=1, frameon=False, fontsize=8.5, labelcolor=INK)
fig.tight_layout()
out = Path("reports/figures/perf_compare.png"); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=SURF)
print("저장:", out)
