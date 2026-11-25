"""A2.9: bar chart of the baseline time per stage, from results/profile_stages.json.

    python scripts/plot_profile_stages.py   ->  results/figures/a2_9_baseline_stages.png
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/profile_stages.json"))
stages = ["loading (decode+resize)", "backward", "forward", "preprocessing (total)",
          "optimizer_step", "other"]
secs = [d["stages_s"][s] for s in stages]
pct = [100 * d["fractions"][s] for s in stages]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(stages[::-1], secs[::-1], color="#1F4E79")
for b, p in zip(bars, pct[::-1]):
    ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{p:.1f}%", va="center", fontsize=9)
ax.set_xlabel("time in one profiled CPU epoch (s)")
ax.set_title(f"Baseline time per stage (cProfile, num_workers=0, total {d['total_s']:.0f} s)")
ax.set_xlim(0, max(secs) * 1.15)
fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/a2_9_baseline_stages.png", dpi=150)
print("-> results/figures/a2_9_baseline_stages.png")
