# Pipeline diagram (A1.9)

N = batch size (32), B = images in the whole split. Shapes are written as in the code:
HWC on the CPU side (NumPy), NCHW (PyTorch) once batched. Written from `src/data_loader.py`,
`src/preprocess_cpu.py`, `src/preprocess_gpu.py`, `src/train.py` and `src/evaluate.py`.

## 1. CPU baseline (`configs/baseline.yaml`, `src/train_cpu.py`)

```
ONCE PER RUN
  find_images(root)  ──►  [(path, grade 0-4)] × 2,999 files
  split_by_patient   ──►  train 2,116 / val 425 / test 458 (patient-level, split_seed 42)
  build model (ReferenceCNN), AdamW, CrossEntropyLoss

EVERY EPOCH, for every training image (DataLoader, 2 worker processes, CPU)
  file on disk (PNG/JPEG, full resolution)
     │ loading: decode + resize (PIL)                       ◄── 63.5% of the profiled epoch
     ▼
  224 × 224 × 3 uint8 (HWC)
     │ / 255
     ▼
  224 × 224 × 3 float32 in [0, 1]
     │ augmentation (train split only, if preprocess.augment: random flip + rotation)
     │ preprocessing: CLAHE → circular crop → normalize (ImageNet mean/std)   ◄── 3.4%
     ▼
  3 × 224 × 224 float32 (CHW)
     │ DataLoader collates N images (drop_last on train)
     ▼
  N × 3 × 224 × 224 float32  (19.3 MB per batch of 32)
     │ forward (ReferenceCNN)                               ◄── 14.5%
     ▼
  logits N × 5 float32  ──► CrossEntropyLoss ──► backward  ◄── 17.6%  ──► optimizer step  ◄── 0.2%

  validation (425 images, same loading + preprocessing, no augmentation, no backward)
     └─► val accuracy, val macro-F1 (confusion matrix 5 × 5); best checkpoint saved
         to checkpoints/<run>/best.pt

ONCE, AT THE END (A7.8)
  evaluate.py: best checkpoint on the TEST split (458 images) ──► results/<run>/metrics.json
```

Stand-alone stages, **not** part of training: the filters of Module C (`filters_cpu.py`:
Gaussian, median, green channel, contrast stretch) are only timed in `benchmark_stages.py` and
the A4 sweeps.

Shares from `results/profile_stages.json` (cProfile, one CPU epoch, `num_workers=0`).

## 2. Optimized pipeline (`configs/optimized.yaml`, `src/train_gpu.py`)

```
ONCE (first run only)                                   CPU
  decode + resize every image ──► uint8 cache on disk: data/cache/cache_<md5>_...  (~451 MB)
                                   (A3: loading ~1000× faster afterwards, results/A35_cache.csv)

EVERY EPOCH                                             CPU (DataLoader workers)
  read 224 × 224 × 3 uint8 from the memory-mapped cache
     │ transpose, no float conversion (raw_uint8)
     ▼
  3 × 224 × 224 uint8 ──► batch N × 3 × 224 × 224 uint8 in PINNED memory (4.8 MB per batch of 32)
═════════════════════════ CPU / GPU boundary: PCIe ═══════════════════════════════════
     │ copy uint8, non_blocking, on a side CUDA stream (CudaPrefetcher), overlapped with
     │ the previous step's compute: 4× fewer bytes than float32 (A4.14, A5.10)
     ▼                                                  GPU
  N × 3 × 224 × 224 uint8 ──► float in [0, 1]
     │ augmentation (train only, if on) → CLAHE → circular crop → normalize  (preprocess_batch_gpu)
     ▼
  N × 3 × 224 × 224 float32 ──► channels_last (NHWC memory layout)
     │ forward + backward under AMP (float16 where safe, GradScaler) → optimizer step
     ▼
  logits N × 5; loss and accuracy accumulated ON the GPU, read once per epoch (one sync)
```

- **Stays on the CPU:** reading the cache, batching, pinning.
- **Crosses PCIe:** only the uint8 image batch and its labels.
- **On the GPU:** everything from the conversion to float onwards.
- **On disk:** the decoded uint8 cache, built once.
