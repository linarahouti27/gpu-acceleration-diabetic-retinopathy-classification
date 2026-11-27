import os, re, sys, glob, json, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ.get("RESULTS_DIR", "results")
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)
BASE = os.environ.get("BASELINE_RUN", "baseline")
OPT = os.environ.get("OPT_RUN", "optimized")
FINAL = os.environ.get("FINAL_RUN", OPT)
FINAL = os.environ.get("FINAL_RUN", OPT)
missing, produced = [], []


def jload(path):
    if not os.path.exists(path):
        missing.append(path)
        return None
    return json.load(open(path))


def hist(name):
    return jload(f"{R}/{name}/history.json")


def epoch_s(h):
    return h.get("steady_epoch_s") if h else None


def best_val_f1(h):
    rows = (h or {}).get("history") or []
    if not rows:
        return None
    keys = [k for k in rows[0] if "f1" in k.lower() and "val" in k.lower()]
    return max(r[keys[0]] for r in rows) if keys else None


def peak_mem(h):
    vals = [e.get("gpu_peak_mem_mb") for e in (h or {}).get("history", []) if e.get("gpu_peak_mem_mb")]
    return max(vals) if vals else None


def save(fig, name):
    fig.tight_layout()
    p = f"{FIG}/{name}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    produced.append(p)
    print("wrote", p)


def ablation_names():
    return sorted(os.path.basename(p) for p in glob.glob(f"{R}/exp_*_off") if os.path.isdir(p))


def run_names():
    others = sorted(os.path.basename(p) for p in glob.glob(f"{R}/exp_*") + glob.glob(f"{R}/final_*") if os.path.isdir(p))
    return [BASE, OPT] + [n for n in others if n not in (BASE, OPT)]


def end_to_end_speedup():
    hb, ho = hist(BASE), hist(OPT)
    if epoch_s(hb) and epoch_s(ho):
        return epoch_s(hb) / epoch_s(ho)
    return None


def fig1_baseline_breakdown():
    d = jload(f"{R}/profile_stages.json")
    if not d:
        return
    items = sorted(((k.strip(), v, k.startswith(" ")) for k, v in d["fractions"].items()),
                   key=lambda t: -t[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [i[0] for i in items][::-1]
    vals = [i[1] * 100 for i in items][::-1]
    cols = ["#9bb7d4" if i[2] else "#1f5fa6" for i in items][::-1]
    ax.barh(labels, vals, color=cols)
    for y, v in enumerate(vals):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Share of total time (%)")
    ax.set_title(f"Baseline runtime per stage (total {d.get('total_s', float('nan')):.1f} s)")
    ax.text(0.98, 0.05, "light bars: parts of preprocessing", transform=ax.transAxes,
            ha="right", fontsize=7)
    save(fig, "fig1_baseline_breakdown")


def fig2_stage_speedup():
    p = f"{R}/timing.csv"
    if not os.path.exists(p):
        missing.append(p)
        return
    t = pd.read_csv(p)
    names, vals, cols = [], [], []
    sp = next((c for c in t.columns if c.startswith("speedup")), None)
    if sp:
        g = t[(t["device"] != "cpu") & t[sp].notna()]
        for _, r in g.iterrows():
            names.append(r["stage"])
            vals.append(float(r[sp]))
            cols.append("#1f5fa6")
    else:
        missing.append(f"{p}: no speedup column (A5.6-5.7)")
    e2e = end_to_end_speedup()
    if e2e:
        names.append("END-TO-END (steady epoch)")
        vals.append(e2e)
        cols.append("#c9672b")
    if not names:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    for y, v in enumerate(vals[::-1]):
        ax.text(v, y, f" {v:.1f}x", va="center", fontsize=8)
    ax.set_xlabel("Speedup vs CPU (x, log scale)")
    ax.set_xscale("log")
    ax.set_title("Stage speedups vs end-to-end speedup")
    save(fig, "fig2_stage_speedup")


def fig3_accuracy_vs_time():
    pts = []
    for n in run_names():
        if not os.path.isdir(f"{R}/{n}"):
            continue
        h = hist(n)
        if epoch_s(h) is not None and best_val_f1(h) is not None:
            pts.append((n, epoch_s(h), best_val_f1(h)))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, x, y in pts:
        c = "#c9672b" if n == BASE else "#2a9d55" if n == OPT else "#1f5fa6"
        ax.scatter(x, y, color=c, s=40)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Steady-state epoch time (s, log scale)")
    ax.set_ylabel("Best validation macro-F1")
    ax.set_title("Accuracy vs epoch time per configuration")
    save(fig, "fig3_accuracy_vs_epoch_time")


def fig4_ablation():
    ho = hist(OPT)
    if epoch_s(ho) is None:
        return
    names, ratio, f1 = [], [], []
    for n in ablation_names():
        h = hist(n)
        if epoch_s(h) is None:
            continue
        names.append(n.replace("exp_", "").replace("_off", ""))
        ratio.append(epoch_s(h) / epoch_s(ho))
        f1.append(best_val_f1(h))
    if not names:
        missing.append(f"{R}/exp_*_off/history.json (no ablation runs)")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, ratio, color="#c9672b")
    ax.axhline(1.0, color="k", lw=0.8)
    for i, r in enumerate(ratio):
        lab = f"{r:.2f}x" + (f"\nF1 {f1[i]:.3f}" if f1[i] is not None else "")
        ax.text(i, r, lab, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Epoch time with the optimisation OFF / optimized")
    ax.set_title("Ablation: slowdown when one optimisation is removed")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save(fig, "fig4_ablation")


def _pick(cols, *pats):
    for c in cols:
        if all(re.search(p, c, re.I) for p in pats):
            return c
    return None


def fig5_resources():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    drawn = False
    for n in (BASE, OPT):
        p = f"{R}/{n}/resources.csv"
        if not os.path.exists(p):
            missing.append(p)
            continue
        df = pd.read_csv(p)
        tc = _pick(df.columns, r"time|elapsed|^t$|^t_s$|ts")   # resources.csv names it t_s
        x = df[tc] - df[tc].iloc[0] if tc else np.arange(len(df))
        cpu = _pick(df.columns, "cpu")
        gpu = _pick(df.columns, "gpu", "util") or _pick(df.columns, "gpu")
        if cpu:
            axes[0].plot(x, df[cpu], label=n)
            axes[0].set_title(cpu)
            drawn = True
        if gpu:
            axes[1].plot(x, df[gpu], label=n)
            axes[1].set_title(gpu)
            drawn = True
        for c in [c for c in df.columns if re.search("mem", c, re.I)][:2]:
            axes[2].plot(x, df[c], label=f"{n}: {c}")
            axes[2].set_title("memory")
            drawn = True
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Utilisation (%)")
    if drawn:
        save(fig, "fig5_resources")


def md(df):
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def tables():
    parts = ["# Tables (auto-generated from results/)"]
    p = f"{R}/timing.csv"
    if os.path.exists(p):
        cols = ["stage", "device", "mean_s", "std_s", "imgs_per_s", "speedup", "speedup_vs_cpu"]
        t = pd.read_csv(p)
        parts.append("## Table 2 - Per-stage timings, CPU vs GPU\n\n" + md(t[[c for c in cols if c in t]]))
    else:
        parts.append("## Table 2 - (results/timing.csv missing)")
        missing.append(p)
    hb = hist(BASE)
    rows = []
    for n in run_names():
        if not os.path.isdir(f"{R}/{n}"):
            continue
        h = hist(n)
        if epoch_s(h) is None:
            continue
        amp = ((h.get("cfg") or {}).get("train") or {}).get("amp")
        mem = peak_mem(h)
        rows.append({"run": n, "device": h.get("device", ""),
                     "precision": "fp16 (amp)" if amp else "fp32",
                     "preprocessing": "gpu" if h.get("gpu_preprocess") else "cpu",
                     "steady epoch (s)": epoch_s(h),
                     "speedup vs baseline": (epoch_s(hb) / epoch_s(h)) if epoch_s(hb) else None,
                     "gpu mem (MB)": round(mem, 1) if mem else None,
                     "val macro-F1": best_val_f1(h)})
    fin = [r for r in rows if r["run"].startswith("final_r")]
    if len(fin) > 1:
        v = np.array([r["steady epoch (s)"] for r in fin])
        rows.append({"run": f"final (mean of {len(fin)} repeats)", "device": fin[0]["device"],
                     "precision": fin[0]["precision"], "preprocessing": fin[0]["preprocessing"],
                     "steady epoch (s)": f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}",
                     "speedup vs baseline": float(epoch_s(hb) / v.mean()) if epoch_s(hb) else None,
                     "gpu mem (MB)": None,
                     "val macro-F1": float(np.mean([r["val macro-F1"] for r in fin]))})
    if rows:
        parts.append("## Table 3 - End-to-end per configuration\n\n" + md(pd.DataFrame(rows)))
    q, cms = [], {}
    for n in (BASE, FINAL):
        m = jload(f"{R}/{n}/metrics.json")
        if not m:
            continue
        row = {"run": n}
        for k, v in m.items():
            if k in ("n_test", "accuracy", "macro_precision", "macro_recall", "macro_f1", "inference_imgs_per_s"):
                row[k] = float(v)
            elif "confusion" in k.lower():
                cms[n] = v
        q.append(row)
    if q:
        parts.append("## Table 4 - Quality on the test set (baseline and final model only)\n\n" + md(pd.DataFrame(q)))
    for n, cm in cms.items():
        parts.append(f"### Confusion matrix: {n}\n\n" + md(pd.DataFrame(cm)))
    path = f"{R}/tables.md"
    open(path, "w").write("\n\n".join(parts) + "\n")
    produced.append(path)
    print("wrote", path)


def notebook_entry():
    nb = "docs/lab_notebook.md"
    if not os.path.exists(nb) or "### A8.1-A8.6" in open(nb).read():
        return
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dev = (hist(BASE) or {}).get("device", "unknown")
    miss = "\n".join(f"  - `{m}`" for m in dict.fromkeys(missing)) or "  - none"
    outs = "\n".join(f"  - `{p}`" for p in produced)
    e = f"""
### A8.1-A8.6 Figures and tables

- **Config / commit:** `scripts/make_figures.py` @ {commit}, inputs from `{R}/` (baseline device field: {dev})
- **Question:** what do the profiling, timing and run results look like as figures and tables for the report?
- **Command:** `python scripts/make_figures.py`
- **Result:** files produced:
{outs}
- **Inputs still missing when this was generated:**
{miss}
- **Decision:** figures regenerated from the official results of the RTX 3050.
"""
    open(nb, "a").write(e)
    print("notebook entry added")


if __name__ == "__main__":
    for fn in (fig1_baseline_breakdown, fig2_stage_speedup, fig3_accuracy_vs_time,
               fig4_ablation, fig5_resources, tables):
        try:
            fn()
        except Exception as ex:
            print(f"SKIPPED {fn.__name__}: {type(ex).__name__}: {ex}")
    print("\nMISSING INPUTS:")
    for m in dict.fromkeys(missing):
        print(" -", m)
    if "--notebook" in sys.argv:
        notebook_entry()
