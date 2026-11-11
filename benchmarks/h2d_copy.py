"""A4.14: host-to-device copy time of one batch, uint8 vs float32, pageable vs pinned memory.

Writes results/h2d_copy.csv.

    python benchmarks/h2d_copy.py
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
from benchmarks.sweep_median import time_it      # the stopwatch from A4.13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default="results/h2d_copy.csv")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available: run this on a GPU machine.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    rows = []
    for n in args.batch_sizes:
        u8 = torch.randint(0, 256, (n, 3, 224, 224), dtype=torch.uint8)   # a batch as the loader gives it
        f32 = u8.float() / 255.0                                          # the same batch as float32
        for name, x in [("uint8", u8), ("float32", f32)]:
            for pinned in [False, True]:
                src = x.pin_memory() if pinned else x
                mean, std = time_it(lambda: src.to("cuda"), cuda=True, repeats=args.repeats)
                mb = src.numel() * src.element_size() / 10 ** 6       # size of the batch in megabytes
                ms = mean * 1e3
                rows.append({"batch_size": n, "dtype": name, "pinned": pinned,
                             "size_mb": mb, "copy_ms": ms, "std_ms": std * 1e3,
                             "bandwidth_gb_s": mb / ms})              # MB / ms = GB / s
                print(rows[-1], flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
