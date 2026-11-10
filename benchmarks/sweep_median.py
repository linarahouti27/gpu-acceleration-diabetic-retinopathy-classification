"""A4.13: median filter time per image, CPU (OpenCV) vs GPU (torch), across batch sizes.

The batch is created on the GPU before timing, so the host-to-device copy is NOT included
(it is measured separately in A4.14). Writes results/sweep_median.csv.

    python benchmarks/sweep_median.py
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
from src import filters_cpu, filters_gpu


def time_it(fn, cuda, repeats=5, warmup=2):
    """Run fn a few times without timing (warm-up), then time it `repeats` times."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        if cuda: torch.cuda.synchronize()   # GPU work is asynchronous: wait before starting the clock
        t0 = time.perf_counter()
        fn()
        if cuda: torch.cuda.synchronize()   # ...and wait until the GPU has finished before stopping it
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/sweep_median.csv")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available: run this on a GPU machine.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    rng = np.random.default_rng(0)
    rows = []
    for n in args.batch_sizes:
        imgs = rng.uniform(0, 1, (n, 224, 224, 3)).astype(np.float32)          # CPU format (N,H,W,3)
        t = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().cuda()    # GPU format (N,3,H,W)

        cpu_mean, cpu_std = time_it(lambda: [filters_cpu.median_filter(i) for i in imgs], cuda=False)
        gpu_mean, gpu_std = time_it(lambda: filters_gpu.median_filter_gpu(t), cuda=True)

        rows.append({"batch_size": n,
                     "cpu_ms_per_image": (cpu_mean / n) * 10 ** 3,
                     "gpu_ms_per_image": (gpu_mean / n) * 10 ** 3,
                     "cpu_std_ms_per_image": (cpu_std / n) * 10 ** 3,
                     "gpu_std_ms_per_image": (gpu_std / n) * 10 ** 3})
        print(rows[-1], flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
