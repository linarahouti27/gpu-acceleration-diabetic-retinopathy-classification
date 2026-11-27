"""End-to-end benchmark (assignment A7): complete training runs of the CPU baseline
and of your optimized GPU configuration, including loader start-up, worker spawn,
host/device transfers and validation.

Protocol: one discarded warm-up run per configuration (file cache, cuDNN autotune,
weight download), then N timed repeats. Each repeat records
  wall_s       whole train.main() call (cold: setup + all epochs)
  epochs_s     wall time minus setup time (epochs plus checkpoint saves)
  steady_epoch mean epoch time excluding epoch 0
Writes results/end_to_end_runs.csv (every repeat) and results/end_to_end.csv (mean/std).
Outputs of these runs go to results/e2e_* so they never overwrite real runs.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch

from src.train import main as train_main
from src.utils import load_json

CONFIGS = [("baseline_cpu", "configs/baseline.yaml", "cpu"),
           ("optimized_gpu", "configs/optimized.yaml", "cuda")]


def one_run(name, cfg_path, device, epochs, rep):
    run = f"e2e_{name}_r{rep}"
    train_main(cfg_path, device, run=run, epochs=epochs)
    h = load_json(f"results/{run}/history.json")
    return {"config": name, "repeat": rep, "epochs": epochs, "wall_s": h["wall_time_s"],
            "setup_s": h["setup_time_s"], "epochs_s": h["wall_time_s"] - h["setup_time_s"],
            "cold_epoch_s": h["cold_epoch_s"], "steady_epoch_s": h["steady_epoch_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    runs = []
    for name, cfg_path, device in CONFIGS:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        print(f"[e2e] warmup {name}", flush=True)
        one_run(name, cfg_path, device, 1, "warmup")
        for rep in range(args.repeats):
            print(f"[e2e] {name} repeat {rep}", flush=True)
            runs.append(one_run(name, cfg_path, device, args.epochs, rep))
    df = pd.DataFrame(runs)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/end_to_end_runs.csv", index=False)

    # A7.1: aggregate per configuration (mean and std over the timed repeats; the warm-up
    # run is not in df) and add the speedups relative to the CPU baseline.
    agg = df.groupby("config", sort=False).agg(
        repeats=("repeat", "count"),
        wall_s_mean=("wall_s", "mean"), wall_s_std=("wall_s", "std"),
        steady_epoch_s_mean=("steady_epoch_s", "mean"),
        steady_epoch_s_std=("steady_epoch_s", "std"),
        cold_epoch_s_mean=("cold_epoch_s", "mean"),
    ).reset_index()
    if "baseline_cpu" in set(agg["config"]):
        base = agg.set_index("config").loc["baseline_cpu"]
        agg["speedup_wall"] = base["wall_s_mean"] / agg["wall_s_mean"]
        agg["speedup_steady_epoch"] = base["steady_epoch_s_mean"] / agg["steady_epoch_s_mean"]
    else:
        agg["speedup_wall"] = np.nan
        agg["speedup_steady_epoch"] = np.nan
    agg.to_csv("results/end_to_end.csv", index=False)
    # For the report: speedup_wall includes the one-time costs (setup, worker spawn, the cold
    # first epoch, validation), speedup_steady_epoch only the repeated cost per epoch.

    print(df.to_string(index=False))
    print()
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
