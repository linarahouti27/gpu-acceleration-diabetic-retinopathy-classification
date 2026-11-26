"""Stage-level benchmark: CPU vs GPU per pipeline stage on one real batch of
training images. Writes results/timing.csv (assignments A2 and A5).

Every stage processes the same N images, so per-stage times are comparable.
The CPU rows are provided. The GPU rows are yours.

Methodology (brief, section 7): warm-up runs, CUDA synchronized around every
timed region (GpuTimer), several repeats, mean +/- std.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import yaml
import torch

from src.utils import GpuTimer, set_seed
from src.data_loader import find_images, split_by_patient, DRDataset
from src import preprocess_cpu, filters_cpu
from src.model import build_model


def bench(fn, device, repeats, warmup=2):
    """Run fn `warmup` times (discarded), then `repeats` timed runs. Returns (mean, std) in s."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        with GpuTimer(device) as t:
            fn()
        ts.append(t.elapsed)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n", type=int, default=64, help="images per stage")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/timing.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    has_cuda = torch.cuda.is_available()
    cpu = torch.device("cpu")
    N, R = args.n, args.repeats
    rows = []

    def add(stage, dev, fn, device, repeats=R, warmup=2, n=N):
        m, s = bench(fn, device, repeats, warmup)
        rows.append({"stage": stage, "device": dev, "mean_s": m, "std_s": s,
                     "n_images": n, "imgs_per_s": n / m})
        print(f"{stage:20s} {dev:4s} {m * 1e3:10.2f} ms +/- {s * 1e3:.2f}", flush=True)

    items = find_images(cfg["data"]["root"])
    train = split_by_patient(items, cfg["data"]["train_frac"], cfg["data"]["val_frac"],
                             cfg["data"].get("split_seed", cfg["seed"]))["train"][:N]
    ds = DRDataset(train, cfg["data"]["image_size"])

    # ---- CPU rows (given) ----
    add("loading", "cpu", lambda: [ds.load(i) for i in range(N)], cpu, repeats=3, warmup=1)
    u8 = np.stack([ds.load(i) for i in range(N)])            # N,H,W,3 uint8
    f32 = u8.astype(np.float32) / 255.0
    add("preprocessing", "cpu", lambda: preprocess_cpu.batch_preprocess_cpu(f32, cfg), cpu)
    add("filter_gaussian", "cpu", lambda: [filters_cpu.gaussian_blur(i) for i in f32], cpu)
    add("filter_median", "cpu", lambda: [filters_cpu.median_filter(i) for i in f32], cpu)
    add("filter_green", "cpu",
        lambda: [np.ascontiguousarray(filters_cpu.green_channel(i)) for i in f32], cpu)
    add("filter_contrast", "cpu", lambda: [filters_cpu.contrast_stretch(i) for i in f32], cpu)
    rng = np.random.default_rng(0)
    add("augmentation", "cpu", lambda: [preprocess_cpu.augment_image(i, rng) for i in f32], cpu)

    x = torch.from_numpy(preprocess_cpu.batch_preprocess_cpu(f32, cfg))
    y = torch.tensor([l for _, l in train])
    model = build_model(cfg["train"]["model"], cfg["data"]["num_classes"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = torch.nn.CrossEntropyLoss()

    def train_step():
        model.train()
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()

    def infer():
        model.eval()
        with torch.no_grad():
            model(x)

    add("train_step", "cpu", train_step, cpu, repeats=3, warmup=1)
    add("inference", "cpu", infer, cpu, repeats=3, warmup=1)

    # ---- GPU rows (A5.6) ----
    if has_cuda:
        from src import preprocess_gpu, filters_gpu
        from src.data_loader import build_cache
        gpu = torch.device("cuda")

        # loading: from the uint8 cache of A3 (our images are PNGs, so no GPU JPEG decode).
        # Reading the cache is CPU work, so the timer uses the CPU device. np.array forces a real
        # read: load() alone returns a view of the memory-mapped file and reads nothing.
        cache = build_cache(train, cfg["data"]["image_size"], "data/cache")   # built once, not timed
        ds_cached = DRDataset(train, cfg["data"]["image_size"], cache=cache)
        add("loading", "cuda", lambda: [np.array(ds_cached.load(i)) for i in range(N)], cpu,
            repeats=3, warmup=1)

        # h2d_transfer: a pinned uint8 batch copied to the device
        u8_pinned = torch.from_numpy(u8).pin_memory()     # N,H,W,3 uint8, pinned, still on the CPU
        add("h2d_transfer", "cuda", lambda: u8_pinned.to(gpu, non_blocking=True), gpu)

        # preprocessing, host-to-device copy included
        add("preprocessing", "cuda",
            lambda: preprocess_gpu.preprocess_batch_gpu(u8_pinned, cfg, device=gpu), gpu)

        # filters and augmentation: the input is already on the GPU (copy not timed)
        t_gpu = preprocess_gpu.to_float_bchw(u8_pinned, gpu)      # float N,C,H,W on the GPU
        add("filter_gaussian", "cuda", lambda: filters_gpu.gaussian_blur_gpu(t_gpu), gpu)
        add("filter_median", "cuda", lambda: filters_gpu.median_filter_gpu(t_gpu), gpu)
        try:   # needs a working numba-cuda install; skip the row instead of crashing
            add("filter_median_numba", "cuda", lambda: filters_gpu.median_filter_numba(t_gpu), gpu)
        except Exception as e:
            print(f"filter_median_numba  skipped: {type(e).__name__}: {e}", flush=True)
        # .contiguous() makes a real copy, like np.ascontiguousarray in the CPU row
        # (the slice alone is a view and does no work)
        add("filter_green", "cuda", lambda: filters_gpu.green_channel_gpu(t_gpu).contiguous(), gpu)
        add("filter_contrast", "cuda", lambda: filters_gpu.contrast_stretch_gpu(t_gpu), gpu)
        add("augmentation", "cuda", lambda: preprocess_gpu.augment_gpu(t_gpu), gpu)

        # train_step / inference: same model type as the CPU rows, on the GPU, then with AMP
        x_gpu, y_gpu = x.to(gpu), y.to(gpu)
        model_gpu = build_model(cfg["train"]["model"], cfg["data"]["num_classes"]).to(gpu)
        opt_gpu = torch.optim.AdamW(model_gpu.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        def train_step_gpu():
            model_gpu.train()
            opt_gpu.zero_grad(set_to_none=True)
            crit(model_gpu(x_gpu), y_gpu).backward()
            opt_gpu.step()

        def infer_gpu():
            model_gpu.eval()
            with torch.no_grad():
                model_gpu(x_gpu)

        def train_step_gpu_amp():
            model_gpu.train()
            opt_gpu.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = crit(model_gpu(x_gpu), y_gpu)
            scaler.scale(loss).backward()
            scaler.step(opt_gpu)
            scaler.update()

        def infer_gpu_amp():
            model_gpu.eval()
            with torch.no_grad(), torch.autocast("cuda"):
                model_gpu(x_gpu)

        add("train_step", "cuda", train_step_gpu, gpu, repeats=3, warmup=1)
        add("inference", "cuda", infer_gpu, gpu, repeats=3, warmup=1)
        add("train_step_amp", "cuda", train_step_gpu_amp, gpu, repeats=3, warmup=1)
        add("inference_amp", "cuda", infer_gpu_amp, gpu, repeats=3, warmup=1)

    df = pd.DataFrame(rows)

    # ---- A5.7: speedup column (CPU time / GPU time of the same stage) ----
    cpu_mean = {r["stage"]: r["mean_s"] for r in rows if r["device"] == "cpu"}

    def speedup(row):
        if row["device"] != "cuda":
            return float("nan")
        base = row["stage"].replace("_amp", "").replace("_numba", "")   # train_step_amp -> train_step
        if base not in cpu_mean:                                       # h2d_transfer has no CPU row
            return float("nan")
        return cpu_mean[base] / row["mean_s"]

    df["speedup"] = df.apply(speedup, axis=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
