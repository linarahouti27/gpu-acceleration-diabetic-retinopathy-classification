# Design proposal

TODO(A3.6): DRAFT by P4 (Lina), 2026-11-13 (day 5), to be completed and agreed by the whole team
at the design meeting; remove this tag once agreed. One row per planned optimization. Every row must
point to a measurement in `results/`.

Written after the baseline profile (P1, `results/profile_stages.json`). Following the compressed
schedule, the implementation of A3 (cache) and A4 (GPU preprocessing and filters) had started in
parallel; the profile then confirmed loading as the first priority and guides the remaining choices.

**Baseline profile (CPU, RTX 3050 laptop, one epoch, num_workers=0):** loading (decode + resize) 63%,
preprocessing 3.4% (CLAHE 1.8%), forward 14.5%, backward 17.6%, optimizer step 0.2%. Amdahl bound if
a stage became instant: 1 / (1 − p) = 2.7× (loading), 1.48× (training), 1.035× (preprocessing).
**GPU profile of the baseline** (Colab T4, draft, `results/torch_profiler_baseline/summary.json`):
the GPU is idle 95% of the time (78 ms of GPU work per 1,677 ms step).

| # | Optimization | Measured bottleneck (file, number) | Hypothesis | Expected gain | Cost / trade-off | Validation |
|---|---|---|---|---|---|---|
| 1 | Decoded-image cache (A3, P2) | `results/profile_stages.json`: loading (decode + resize) = 63% of the epoch | The baseline decodes the same images again every epoch; with a cache, each image is decoded once and later epochs only read the stored pixels | Less than the Amdahl bound of 2.7×, since reading the cache still costs a little time | About 450 MB on disk (2,999 × 224 × 224 × 3 bytes); the first run is slower, since it builds the cache | Correct: `test_cache_matches_direct_decode` (P2). Faster: A3.5, P1 measures loading with / without the cache vs the baseline (first run, warm cache, cold OS cache) |
| 2 | More DataLoader workers, pinned memory, CUDA prefetcher (A5.1, A5.2, A5.8, P2) | `profile_stages.json`: loading = 63%, done by only 2 workers; GPU profile: copy = 11 ms per step; A4.14: pinned memory copies 1.9× faster | More workers decode images in parallel; pinned memory speeds up copies; the prefetcher reduces GPU idle time by preparing the next batch in advance | Before the cache, more workers matter most (loading 63% vs copy 0.7% of a step). After the cache, loading becomes cheap: workers help less, and pinned memory + prefetcher matter more | Each worker uses RAM, and the laptop's RAM is already 7.4 / 8 GB (`results/baseline/resources.csv`): too many workers could cause swapping. Pinned memory also takes RAM | Correct: `test_prefetcher_yields_same_batches_on_device`. Faster: A5.8 sweep (0–8 workers, 3 repeats, img/s, RAM watched), run by P1 |
| 3 | GPU preprocessing (CLAHE, crop, normalize) + copy uint8 (A4.1–A4.6 P3, A5.3 P2) | `profile_stages.json`: preprocessing = 3.4% (CLAHE 1.8%); A4.14: uint8 + pinned copies 6.8× faster than float32 from normal memory | CLAHE is data-parallel, so the GPU can process a whole batch at once; converting to float on the GPU allows copying uint8 (4× fewer bytes) | Small directly: at most 1.035× (Amdahl). Worth more indirectly: it enables the uint8 copy, and frees the CPU workers, which matters more once the cache removes decoding | GPU CLAHE is not bit-identical to OpenCV (correlation > 0.95 only): it could slightly change what the model learns | Correct: P3's equivalence tests. Faster: P3's CLAHE sweep (13–15× on Colab, draft; official run by P1). Quality: same model with CPU vs GPU CLAHE, val macro-F1 vs seed-to-seed variation |
| 4 | Train on the GPU (same model and settings, device cpu → cuda) | `profile_stages.json`: forward + backward + optimizer = 32% of the CPU epoch; `results/timing.csv`: about 25 ms of CPU training per image | Convolutions are data-parallel, so the GPU computes them much faster | At most 1.48× (Amdahl). Small before the cache, because the GPU mostly waits for decoding (it also frees CPU cores for loading). Larger once loading is fixed | GPU memory (868 MB for the reference CNN at batch 32, of 6 GB); copies (~0.7 s per epoch, much less than the ~50 s of CPU training they replace); results not bit-identical to the CPU's (summation order, cuDNN, A1.5) | Speed: baseline on CPU vs GPU, same config, both on P1's laptop (A6.10). Quality: val macro-F1 difference smaller than the seed-to-seed variation |
| 5 | Mixed precision (AMP) + `channels_last` (A6.1, A6.2, P3) | `profile_stages.json`: training = 32% of the epoch | 16-bit numbers are half the size and run on the GPU's Tensor Cores; `channels_last` may suit Tensor Cores better | Little today: the GPU computes only ~5% of each step and waits for loading. It should matter once loading is fixed. Half the memory also allows bigger batches | Less precise numbers could lower F1 and accuracy (`GradScaler` protects small gradients). `channels_last` may not help: measure it, don't assume | Speed: AMP vs no AMP, then `channels_last` vs not, one change at a time. Quality: the F1 difference must be smaller than the seed-to-seed variation |
| 6 | ResNet-18 pretrained on ImageNet, lr 0.0001 (A6.6, A6.9, P4) | Quality, not speed: the baseline reaches only 0.35 val macro-F1 and underfits (train accuracy 0.34) (`results/baseline/history.json`) | A pretrained model should perform better than the reference CNN trained from scratch, especially with only ~2,100 training images | Measured in drafts: val macro-F1 0.35 → 0.55 (Colab, `results/exp_resnet18_lr1e-4/`); but it overfits (train 0.96 vs val 0.55; augmentation halves the gap) | 3.6× more FLOPs per image (`results/model_size.csv`): once loading is fixed, its training will be slower than the reference CNN's (only +4% today); a bit more GPU memory (921 vs 868 MB) | Both models trained on P1's laptop, same splits, same budget, same checkpoint rule (val F1), ideally several seeds; test set used once per model (A7.8) |

The GPU filters (A4.7–A4.11: Gaussian, median, green channel, contrast stretch, and the fused Numba
median) are not part of the training pipeline: they are measured as stand-alone stages (A4.13, A4.15)
and do not change the end-to-end time.

## Revised architecture

```
                 CPU (DataLoader workers)                         GPU
first run:  decode + resize every image ──► uint8 cache on disk (data/cache, ~450 MB; file
            name depends on the image list and size)
every epoch:
  read uint8 images from the cache (memory-mapped)
  batch (N,3,224,224) uint8 in pinned memory ──PCIe──► copy, non-blocking, overlapped with the
             (4.8 MB per batch of 32,                   previous step (CudaPrefetcher)
              4× less than float32)                          │
                                                        to float [0,1]
                                                        augmentation (train only)
                                                        CLAHE, circular crop, normalize
                                                        model forward + backward (AMP,
                                                        channels_last if it helps)
                                                        optimizer step
                                                        loss / accuracy kept on the GPU,
                                                        read once per epoch
```

- **Stays on the CPU:** reading the cached images and building batches (workers), pinning.
- **Crosses the PCIe bus:** only the uint8 image batch and its labels (no float32 images, no
  per-batch results back to the CPU).
- **On the GPU:** everything from the conversion to float onwards.
- **On disk:** the decoded uint8 cache, built once.

## Order of experiments

One factor per experiment, starting with the largest share of the measured time (Amdahl): the
later optimizations only show in the epoch time once the earlier bottlenecks are removed, so each
step is followed by a re-profile (`benchmarks/profile_gpu.py`, `benchmarks/benchmark_stages.py`).

0. **References** (P1's laptop): CPU baseline (done) and the same baseline on the GPU (device effect,
   A6.10), with the GPU profile (idle fraction).
1. **Cache** (loading = 63%): A3.5, then re-profile.
2. **Workers and pinned memory** (A5.8 sweep), on top of the cache.
3. **GPU preprocessing + uint8 copy + prefetcher** (A5.3–A5.5), then re-profile: the GPU idle
   fraction should drop.
4. **AMP**, then **`channels_last`**, then **batch size** (A6.1, A6.2, A6.7), one at a time, each
   with a quality check (F1 vs seed variation).
5. **Model:** ResNet-18 (+ augmentation if it helps) vs the reference CNN on the final pipeline
   (A6.9, A6.10).
6. **Final:** `configs/optimized.yaml`, ablation (switch off one optimization at a time, A7.6),
   re-profile (A7.7), test-set evaluation once per final model (A7.8).

## Risks

- **RAM (8 GB, already 7.4 GB used in the baseline):** more workers and pinned memory could cause
  swapping, which would slow everything down.
- **GPU memory (6 GB):** ResNet-18 with larger batches may not fit; AMP halves activation memory.
- **Windows:** DataLoader workers use `spawn` (slower start-up, everything must be picklable);
  `torch.compile` (A6.3) is not available without Triton.
- **Laptop power and heat:** battery mode or thermal throttling change the speed; always measure
  plugged in, on the same power mode, with repeats (mean ± std).
- **Disk and OS file cache:** the cache's speed depends on whether its file is already in memory
  (warm) or read from disk (cold): A3.5 measures both.
- **PCIe bandwidth:** the laptop's pinned copies are capped at about 9.4 GB/s (vs about 12 GB/s on
  Colab), so copies are a larger share there.
- **Quality:** GPU CLAHE (not bit-identical to OpenCV) and AMP (lower precision) could change the
  model's quality; only one seed per experiment so far, so small differences are within noise; the
  classes are not balanced (504–701 images per grade), so macro-F1 is the main metric.
- **Amdahl:** once the cache removes decoding, another stage becomes the bottleneck; GPU
  preprocessing alone cannot speed up the epoch much (preprocessing = 3.4%).
