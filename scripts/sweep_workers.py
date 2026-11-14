"""Sweep num_workers and batch_size; write results/sweep_workers.csv (A5.8).

P2 writes this; P1 runs it on the RTX 3050 and plots images/second.
Method: one warm-up epoch (discarded), then --epochs timed epochs, mean and std.
Same splits and config in every cell; only num_workers and batch_size change.
"""
import os, sys, argparse, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import yaml
import torch
from src.utils import GpuTimer, set_seed
from src.data_loader import make_loaders
from src.model import build_model


def run_pass(loader, model, opt, crit, device, max_batches):
    """One training pass; returns the number of images processed."""
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()
        n += x.size(0)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_cache.yaml")
    ap.add_argument("--workers", default="0,1,2,4,8")
    ap.add_argument("--batch", default="32,64,128")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = full epoch")
    ap.add_argument("--out", default="results/sweep_workers.csv")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} config={args.config}", flush=True)
    rows = []
    for nw in [int(v) for v in args.workers.split(",")]:
        for bs in [int(v) for v in args.batch.split(",")]:
            row = {"num_workers": nw, "batch_size": bs, "device": str(device),
                   "imgs_per_s": None, "std": None, "n_epochs": args.epochs, "error": ""}
            try:
                cfg = json.loads(json.dumps(base))
                cfg["data"]["num_workers"] = nw
                cfg["data"]["batch_size"] = bs
                set_seed(cfg["seed"])
                loaders, _ = make_loaders(cfg, device.type, which=("train",))
                loader = loaders["train"]
                model = build_model(cfg["train"]["model"], cfg["data"]["num_classes"]).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
                crit = torch.nn.CrossEntropyLoss()
                model.train()
                for _ in range(args.warmup):
                    run_pass(loader, model, opt, crit, device, args.max_batches)
                rates = []
                for _ in range(args.epochs):
                    with GpuTimer(device) as t:
                        n = run_pass(loader, model, opt, crit, device, args.max_batches)
                    rates.append(n / t.elapsed if t.elapsed > 0 else 0.0)
                row["imgs_per_s"] = float(np.mean(rates))
                row["std"] = float(np.std(rates))
                print(f"nw={nw} bs={bs}: {row['imgs_per_s']:.0f} +/- {row['std']:.0f} img/s", flush=True)
            except Exception as e:
                row["error"] = str(e)[:100]
                print(f"nw={nw} bs={bs}: ERROR {row['error']}", flush=True)
            finally:
                for name in ("loader", "loaders", "model", "opt"):
                    globals().pop(name, None)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
