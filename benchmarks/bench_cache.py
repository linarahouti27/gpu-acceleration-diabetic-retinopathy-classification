# benchmarks/bench_cache.py
# A3.5: loading stage with and without the decoded-image cache (train split only)
import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data_loader import find_images, split_by_patient, DRDataset, build_cache


def folder_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e6


def time_loading(ds, n, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(n):
            ds.load(i).copy()  # copy() forces the pixels to be really read
        times.append(time.perf_counter() - t0)
    return times


def make_row(stage, times, n):
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return [stage, "cpu", mean, std, n, n / mean]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cache-dir", default=os.path.join(os.path.expanduser("~"), "dr_cache_A35"))
    ap.add_argument("--mode", choices=["full", "cold"], default="full")
    ap.add_argument("--out", default="results/A35_cache.csv")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    size = cfg["data"]["image_size"]
    splits = split_by_patient(find_images(cfg["data"]["root"]), cfg["data"]["train_frac"],
                              cfg["data"]["val_frac"], cfg["seed"])
    train = splits["train"]
    cache_exists = os.path.isdir(a.cache_dir) and len(os.listdir(a.cache_dir)) > 0
    rows = []

    if a.mode == "full":
        if cache_exists:
            sys.exit("Cache folder is not empty. Delete it first: Remove-Item -Recurse -Force " + a.cache_dir)
        ds = DRDataset(train, size)
        rows.append(make_row("loading_nocache", time_loading(ds, a.n, a.repeats), a.n))

        t0 = time.perf_counter()
        cache = build_cache(train, size, a.cache_dir)  # first run: the cache is built here
        build_s = time.perf_counter() - t0
        rows.append(["cache_build_first_run", "cpu", build_s, 0.0, len(train), len(train) / build_s])

        t0 = time.perf_counter()
        cache = build_cache(train, size, a.cache_dir)  # cache already exists: reopen only
        reopen_s = time.perf_counter() - t0
        rows.append(["cache_reopen", "cpu", reopen_s, 0.0, len(train), len(train) / reopen_s])

        ds = DRDataset(train, size, cache=cache)
        rows.append(make_row("loading_cache_warm", time_loading(ds, a.n, a.repeats), a.n))
    else:
        if not cache_exists:
            sys.exit("No cache found. Run --mode full first.")
        cache = build_cache(train, size, a.cache_dir)
        ds = DRDataset(train, size, cache=cache)
        rows.append(make_row("loading_cache_cold", time_loading(ds, a.n, 1), a.n))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "device", "mean_s", "std_s", "n_images", "imgs_per_s"])
        w.writerows(rows)

    for r in rows:
        print(f"{r[0]:24s} mean {r[2]:.4f} s  std {r[3]:.4f}  n {r[4]}  {r[5]:.1f} img/s")
    src_mb = sum(os.path.getsize(p) for p, _ in train) / 1e6
    print(f"train images on disk: {src_mb:.1f} MB ({len(train)} images)")
    print(f"cache folder: {folder_mb(a.cache_dir):.1f} MB")


if __name__ == "__main__":
    main()