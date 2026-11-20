"""Build the mandatory figures and tables from results/ (assignment A8).

Read only measured artifacts. Never type a number by hand.

Inputs you will have produced:
  results/profile_stages.json      baseline time per stage          (benchmarks/profile_baseline.py)
  results/timing.csv               per-stage CPU vs GPU             (benchmarks/benchmark_stages.py)
  results/<run>/metrics.json       classification metrics           (src/evaluate.py)
  results/<run>/history.json       per-epoch times and accuracy     (src/train.py)
  results/<run>/resources.csv      CPU/GPU utilization over time    (--monitor)
Outputs: results/figures/*.png and results/tables.md
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (used by the functions you write)

OUT = os.path.join("results", "figures")


def fig1_baseline_breakdown():
    """TODO(A8.1): Figure 1. Bar chart (or stacked bar) of the baseline runtime per stage,
    as a share of the total, from results/profile_stages.json. Label the axes with units."""
    st = prof["stages_s"]
    top = {k: v for k, v in st.items() if not k.startswith(" ")}
    items = sorted(top.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    total = sum(vals) or 1.0
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bars = ax.bar(labels, [100 * v / total for v in vals], color="#2E5496")
    ax.set_ylabel("Share of baseline epoch time (%)")
    ax.set_title("Figure 1 - Baseline runtime per stage")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig1_baseline_breakdown.png"), dpi=150)
    plt.close(fig)


def fig2_speedup():
    """TODO(A8.2): Figure 2. Stage-level speedups next to the end-to-end speedup.
    Use a log scale if values differ by orders of magnitude. Mark speedup = 1 with a line."""
    raise NotImplementedError("TODO(A8.2)")


def fig3_accuracy_vs_time():
    """TODO(A8.3): Figure 3. One point per tested configuration: x = steady-state epoch time,
    y = validation macro-F1 (`val_f1` in history.json; never select with the test set).
    Add the test macro-F1 of the baseline and of the final model as two marked points.
    Annotate each point with the configuration name."""
    raise NotImplementedError("TODO(A8.3)")


def fig4_ablation():
    """TODO(A8.4): Figure 4. Ablation: start from your full optimized configuration and
    show what each single optimization contributes (one config with one change per bar)."""
    raise NotImplementedError("TODO(A8.4)")


def fig5_resources():
    """TODO(A8.5): Figure 5. CPU utilization, GPU utilization and memory over time for the
    baseline and the optimized run, from the resources.csv files."""
    raise NotImplementedError("TODO(A8.5)")


def tables():
    """TODO(A8.6): write Tables 2 and 3 of the brief as Markdown to results/tables.md.
    Report stages that do not apply or that you combined explicitly, not as invented numbers.
    Also compute the Amdahl prediction S = 1 / ((1 - p) + p / s) for your main optimization
    and put it next to the measured end-to-end speedup."""
    raise NotImplementedError("TODO(A8.6)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig1_baseline_breakdown, fig2_speedup, fig3_accuracy_vs_time,
               fig4_ablation, fig5_resources, tables):
        fn()
