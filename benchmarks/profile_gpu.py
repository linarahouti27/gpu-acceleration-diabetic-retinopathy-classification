"""PyTorch Profiler on the GPU training loop (assignment A7).

Goal: find out where wall time goes during steady-state training: GPU kernels,
host-to-device copies, or GPU idle time caused by the input pipeline.
Writes results/torch_profiler/{summary.json, top_ops.txt, trace.json}.
trace.json opens in https://ui.perfetto.dev or chrome://tracing.

    python benchmarks/profile_gpu.py --config configs/optimized.yaml

The training step mirrors src/train.py: GPU preprocessing, AMP, channels_last and the
CUDA prefetcher are used only when the config turns them on, so the same script profiles
the baseline pipeline and the optimized one.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import torch.nn as nn
from torch.autograd import DeviceType
from torch.profiler import profile, schedule, ProfilerActivity

from src.utils import set_seed
from src.data_loader import make_loaders
from src.model import build_model
from src.train import resolve_gpu_preprocess, make_gpu_pre


def union_length(intervals):
    """intervals: a list of (start, end) pairs. Returns the total length covered,
    counting overlapping parts only once."""
    if not intervals:
        return 0

    # sorting intervals by start time
    intervals = sorted(intervals)

    # count length
    length = 0
    current = intervals[0]
    for t in intervals[1:]:
        if t[0] < current[1]:
            current = (current[0], max(t[1], current[1]))
        else:
            length += current[1] - current[0]
            current = t
    length += current[1] - current[0]

    return length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimized.yaml")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--out", default="results/torch_profiler")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available: run this on a GPU machine.")

    # ---- A7.2: build everything exactly like src/train.py ----
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    device = torch.device("cuda")
    gpu_pre_on = resolve_gpu_preprocess(cfg, device)
    augment = bool(cfg["preprocess"].get("augment", False))
    loaders, _ = make_loaders(cfg, device.type, gpu_preprocess=gpu_pre_on, which=("train",))
    loader = loaders["train"]
    pre = make_gpu_pre(cfg, device, augment=augment) if gpu_pre_on else None
    if cfg["data"].get("prefetch_gpu", False):
        from src.prefetcher import CudaPrefetcher
        loader = CudaPrefetcher(loader, device)

    amp = bool(cfg["train"].get("amp", False))
    channels_last = bool(cfg["train"].get("channels_last", False))
    model = build_model(cfg["train"]["model"], cfg["data"]["num_classes"]).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"].get("weight_decay", 0.0))
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    n_steps = args.warmup + args.steps
    if len(loaders["train"]) < n_steps:
        sys.exit(f"only {len(loaders['train'])} training batches; lower --warmup/--steps")
    print(f"[profile] config={args.config} model={cfg['train']['model']} batch={cfg['data']['batch_size']} "
          f"gpu_preprocess={gpu_pre_on} amp={amp} channels_last={channels_last} "
          f"warmup={args.warmup} steps={args.steps}", flush=True)

    h2d_bytes = 0
    batches = iter(loader)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 # repeat=1: record ONE cycle. With the default (repeat=0) the schedule starts a new,
                 # empty cycle after the active steps, and the recorded steps are discarded.
                 schedule=schedule(wait=0, warmup=args.warmup, active=args.steps, repeat=1)) as prof:
        for step in range(n_steps):
            if step == args.warmup:                  # the recorded window starts here
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            x, y = next(batches)                     # includes waiting for the DataLoader
            if step >= args.warmup:
                h2d_bytes += x.numel() * x.element_size() + y.numel() * y.element_size()
            y = y.to(device, non_blocking=True)
            x = pre(x) if pre is not None else x.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, enabled=amp):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            prof.step()
        torch.cuda.synchronize()
        wall_us = (time.perf_counter() - t0) * 1e6

    # ---- A7.3: analyse the GPU events of the recorded steps ----
    # GPU events only, without the "ProfilerStep#N" labels: those are markers the profiler puts
    # on the GPU timeline around each step, not real GPU work (counting them inflated the times).
    gpu_events = [e for e in prof.events()
                  if e.device_type == DeviceType.CUDA and not e.name.startswith("ProfilerStep")]
    if not gpu_events:
        print("WARNING: the profiler recorded no GPU events", flush=True)

    def dur(e):
        return e.time_range.end - e.time_range.start          # microseconds

    h2d = [e for e in gpu_events if "HtoD" in e.name]
    d2h = [e for e in gpu_events if "DtoH" in e.name]
    kernels = [e for e in gpu_events if "Memcpy" not in e.name and "Memset" not in e.name]
    busy_us = union_length([(e.time_range.start, e.time_range.end) for e in gpu_events])

    summary = {
        "config": args.config, "model": cfg["train"]["model"],
        "batch_size": cfg["data"]["batch_size"], "gpu": torch.cuda.get_device_name(0),
        "gpu_preprocess": gpu_pre_on, "amp": amp, "channels_last": channels_last,
        "warmup_steps": args.warmup, "profiled_steps": args.steps,
        "wall_ms": wall_us / 1e3,
        "wall_ms_per_step": wall_us / 1e3 / args.steps,
        "gpu_kernel_ms": sum(dur(e) for e in kernels) / 1e3,
        "h2d_memcpy_ms": sum(dur(e) for e in h2d) / 1e3,
        "h2d_bytes": h2d_bytes,
        "d2h_memcpy_ms": sum(dur(e) for e in d2h) / 1e3,
        "gpu_busy_ms": busy_us / 1e3,                 # union of all GPU intervals (no double counting)
        "gpu_idle_fraction": 1 - busy_us / wall_us,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    try:
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    except Exception:                                # newer PyTorch names it device_time_total
        table = prof.key_averages().table(sort_by="device_time_total", row_limit=30)
    with open(os.path.join(args.out, "top_ops.txt"), "w") as f:
        f.write(table)
    prof.export_chrome_trace(os.path.join(args.out, "trace.json"))
    print(json.dumps(summary, indent=2))
    print(f"-> {args.out}/summary.json, top_ops.txt, trace.json")


if __name__ == "__main__":
    main()
