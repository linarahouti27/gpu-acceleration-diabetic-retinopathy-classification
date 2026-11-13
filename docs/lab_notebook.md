# Lab notebook

One entry per experiment. Write the prediction BEFORE you run anything. A wrong prediction with a good explanation is worth more than a right one you cannot explain.

## Entry template

```
### <assignment id> <short title>          date: <yyyy-mm-dd>
Config / commit:      <config file and git commit hash>
Question:             <what do you want to find out?>
Prediction:           <number or ranking, and the reason, written before the run>
Command:              <exact command line>
Result:               <measured numbers: mean +/- std, n repeats, cold vs steady>
Prediction vs result: <what was different?>
Explanation:          <why? what evidence supports it (profiler output, a second measurement)?>
Decision:             <keep / drop the change, and what you do next>
```

## A1.3 Baseline configuration review

TODO(A1.3): answers to the questions listed at the top of `configs/baseline.yaml`.

## Entries

(add yours below)

### A4.12 Filter test tolerances (P4: Lina) — 2026-11-09

- **Config / commit:** none (tests only); filters at commit `30071bd` (A4.7–A4.10).
- **Question:** are the tolerances of the three filter tests in `tests/test_cpu_gpu_equivalence.py` appropriate? (The preprocessing tests are covered by P3.) 1 gray level = 1/255 ≈ 0.004.
- **Prediction:** n/a: a review of the test specification, not a timed experiment. No prediction was written before reading the tests.
- **Command:** `python -m pytest tests/test_cpu_gpu_equivalence.py -k "gaussian or median or contrast" -v`
- **Result:** GPU run on Colab, Tesla T4, Python 3.13.16, pytest 8.4.2: 3 passed, 6 deselected in 3.29 s.

  | Test | Tolerance | Result | Largest difference (NumPy check vs `filters_cpu.py`) |
  |---|---|---|---|
  | `test_gaussian_filter_equivalence` | every pixel within 0.02 | PASSED | 3e-7 |
  | `test_median_filter_equivalence` | ≤ 2% of pixels off by > 1 gray level | PASSED | 1 gray level, 0% of pixels beyond it |
  | `test_contrast_stretch_equivalence` | every pixel within 0.001 | PASSED | 6e-8 |

- **Prediction vs result:** n/a (no prediction).
- **Explanation:**
  - **Gaussian blur (0.02): too loose.** 0.02 is about 5 gray levels, while my blur differs from OpenCV by only 3e-7, so a slightly wrong blur (wrong sigma, wrong border mode) could still pass. Some margin is needed because the GPU computes convolutions in lower precision (TF32), but about 1 gray level (0.004) would be enough.
  - **Median filter (2%): too generous.** The 1-gray-level allowance is necessary: the CPU version converts to uint8, which rounds every value down to a whole gray level, so it can never be more than 1 gray level away from an exact float result. But 2% (about 1,000 pixels of a 224×224 image) is too much: my version had 0% of pixels beyond 1 gray level. Also, the test image is black at the borders, so a wrong padding mode (reflect instead of replicate) would still pass.
  - **Contrast stretch (0.001): about right.** 0.001 is a quarter of a gray level, so invisible, and CPU and GPU compute the percentiles the same way (linear interpolation). It could be tightened slightly (e.g. 1e-4), but not much more: a tolerance too close to zero could fail a correct GPU version just because of floating-point rounding differences.
  - **Green channel: no test.** Nothing would warn us if the function returned the wrong channel (e.g. `t[:, 0:1]`, the red one).
- **Decision:** keep the tests (all pass), but propose: a Gaussian tolerance of about 1 gray level; for the median, at most about 0.1% of pixels beyond 1 gray level, with a test image that is not black at the borders; and a green-channel test checking the shape `(N,1,H,W)`, the values (equal to `t[:, 1]`) and that the result is a view (same memory).

### A4.13 Median filter: GPU vs CPU time per image across batch sizes (P4: Lina) — 2026-11-10, official run 2026-11-12

- **Config / commit:** none (script defaults). Official run: `benchmarks/sweep_median.py` at commit `9faff69` (on `main`), run by P1 on the RTX 3050 laptop (branch `p1-baseline`, `results/sweep_median.csv`); Numba was not installed, so no Numba column (see A4.11). Colab draft: commit `9ce6840` (that version reported the std per batch; per image since `1954c73`); its CSV was not committed, numbers below copied from the console output.
- **Question:** at which batch size (1, 8, 32, 64, 128) does the GPU median filter become faster per image than the CPU (OpenCV) one, if ever?
- **Prediction:** at batch size 1 the GPU is slower than the CPU, because the fixed kernel launch overhead dominates. As the batch grows, the GPU time per image drops (the overhead is shared by more images) and goes below the CPU time at about batch size 64. The CPU time per image stays roughly constant.
- **Command:** `python benchmarks/sweep_median.py` (batch sizes 1 8 32 64 128, 5 repeats, 2 warm-up runs, CUDA synchronized around every timed run; the batch is already on the GPU, so the host-to-device copy is excluded).
- **Result:**
  - **Official (P1, RTX 3050 Laptop 6 GB, i5-13420H), ms per image:**

  | Batch | CPU (OpenCV) | GPU (torch) |
  |---|---|---|
  | 1 | 0.47 | 4.40 |
  | 8 | 0.68 | 3.69 |
  | 32 | 0.77 | 2.91 |
  | 64 | 0.71 | 2.91 |
  | 128 | 0.68 | 2.91 |

  - **Draft (Colab, Tesla T4, 2 shared CPU cores), ms per image:** CPU 0.73 / 0.82 / 0.84 / 1.11 / 1.10; GPU 3.58 / 2.76 / 2.44 / 2.44 / 2.45 (batch 1 / 8 / 32 / 64 / 128). Colab's CPU is shared, so its CPU times are noisy.
- **Prediction vs result:** I predicted the GPU would overtake the CPU at batch size 64, but the CPU won at every batch size, on both machines (about 4× faster at batch 128 on the laptop, about 2× on Colab). The first part of the prediction was right: the GPU time per image drops as the launch overhead is shared by more images (4.40 → 2.91 ms on the laptop). But it becomes flat from batch 32 on: the GPU is then fully busy and that time is the cost of the work itself, so bigger batches cannot help.
- **Explanation:**
  - **The two versions do not do the same amount of work.** OpenCV's `medianBlur` is a specialized algorithm tuned for 8-bit images that avoids sorting most values. My GPU version uses general operations: `F.unfold` copies each pixel's 25 neighbours into memory (25× the input size, about 1.9 GB for 128 images), then `torch.median` sorts those 25 values once for every pixel. The extra memory traffic and generic sorting cost more than the GPU's parallelism gains.
  - **The gap depends on the hardware:** it is bigger on the laptop (4× vs 2×) because its CPU is faster (newer, 8 cores / 12 threads that OpenCV can use, vs 2 shared cores on Colab: 0.68 vs 1.10 ms at batch 128) and its GPU slower (the RTX 3050 Laptop has 20 SMs vs 40 for the T4, so about half as many threads at once: 2.91 vs 2.45 ms).
- **Decision:** do not use this GPU median filter as is (it is not used in training anyway). To make it faster, fuse the steps into one custom kernel (Numba CUDA, A4.11): each thread reads its 25 neighbours directly and sorts them in registers, which removes the 25× copy. Tuning blocks / threads per block is a secondary improvement. Since the CPU–GPU balance depends on the machine, a speed-up measured on one machine cannot be generalized to another: the official comparison is the laptop's.

### A4.14 Host-to-device copy: uint8 vs float32 batch (P4: Lina) — 2026-11-10, official run 2026-11-12

- **Config / commit:** none (script defaults). Official run: `benchmarks/h2d_copy.py` at commit `a59a759` (on `main`), run by P1 on the RTX 3050 laptop (branch `p1-baseline`, `results/h2d_copy.csv`). Colab draft: same commit; its CSV was not committed, numbers below copied from the console output.
- **Question:** how long does copying one batch of images from CPU memory to the GPU take as uint8 vs as float32? (The second part, how much of the GPU preprocessing time is the copy, needs P3's `preprocess_batch_gpu`.)
- **Prediction:** uint8 should be about 4× faster to copy, because it has 4× fewer bytes to move (1 byte per value instead of 4).
- **Command:** `python benchmarks/h2d_copy.py` (batch sizes 1 8 32 128, 10 repeats, 2 warm-up runs, CUDA synchronized; pinning is done before timing, only the copy is timed).
- **Result:**
  - **Official (P1, RTX 3050 Laptop 6 GB), copy time in ms:**

  | Batch | uint8, normal | float32, normal | Ratio | uint8, pinned | float32, pinned | Ratio |
  |---|---|---|---|---|---|---|
  | 1 | 0.054 | 0.132 | 2.4× | 0.042 | 0.136 | 3.2× |
  | 8 | 0.332 | 1.064 | 3.2× | 0.203 | 0.549 | 2.7× |
  | 32 | 0.754 | 3.798 | 5.0× | 0.556 | 2.001 | 3.6× |
  | 128 | 3.399 | 17.313 | 5.1× | 2.062 | 8.700 | 4.2× |

  Bandwidth for large batches: about 4.5–6.4 GB/s (normal) vs about 9.4 GB/s (pinned).
  - **Draft (Colab, Tesla T4), copy time in ms:** uint8 normal 0.083 / 0.381 / 1.185 / 4.145; float32 normal 0.214 / 1.102 / 4.090 / 16.959; uint8 pinned 0.041 / 0.133 / 0.427 / 1.632; float32 pinned 0.079 / 0.443 / 1.629 / 6.328 (batch 1 / 8 / 32 / 128). Bandwidth for large batches about 4.6 GB/s (normal) vs about 12 GB/s (pinned).
  - **Batch 32, float32 normal → uint8 pinned:** 3.80 → 0.56 ms (6.8×) on the laptop; 4.09 → 0.43 ms (9.5×) on Colab.
- **Prediction vs result:** uint8 is faster than float32 at every batch size, on both machines. The predicted 4× holds for large batches (3.6–5.1× at 32–128 on the laptop, 3.9–4.1× at 128 on Colab), but not for small ones (2–3× at batch 1): every copy has a fixed start-up cost that dominates when the batch is tiny.
- **Explanation:**
  - **The copy time depends on the number of bytes, not on the type:** on Colab, uint8 at batch 128 and float32 at batch 32 are both 19.27 MB and take the same time (4.15 vs 4.09 ms normal, 1.63 vs 1.63 ms pinned).
  - **Pinned memory is faster** because the driver does not need an extra copy into a fixed staging area first: about 1.9× for float32 on the laptop, about 2.6× on Colab.
  - **The laptop's copies are capped lower** (about 9.4 vs 12 GB/s pinned), probably because its PCIe connection to the GPU is narrower or older than the T4's (to confirm: `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv` on P1's laptop). The pinned-memory gain is also smaller on the laptop because its faster CPU does the extra staging copy of normal memory faster.
- **Decision:** the conclusion holds on both machines: copy the batches as uint8 from pinned memory and convert to float on the GPU (P3's design, with P2's `pin_memory`): 6.8× faster than float32 from normal memory on the laptop (batch 32: 3.80 → 0.56 ms). The pinned uint8 batch is small (19 MB at batch 128), so pinning is cheap (PyTorch's `pin_memory` already reuses pinned blocks). Keep it in perspective: the copy was 14% of the GPU's busy time in the baseline profile (A7.2–A7.3), and the GPU is idle 95% of the time, so this matters only once the loading bottleneck is fixed (Amdahl). Next: the second part of A4.14 needs P3's `preprocess_batch_gpu`.

### A4.11 Second GPU median filter: fused custom Numba CUDA kernel (P4: Lina) — 2026-11-10

- **Config / commit:** none (script defaults); `src/filters_numba.py` and `benchmarks/sweep_median.py` at commit `9faff69`.
- **Question:** A4.13 showed the PyTorch median (pad + `F.unfold` + `torch.median`) is slower than OpenCV at every batch size. Can a second GPU version, a fused custom Numba CUDA kernel, beat OpenCV and the PyTorch version while staying correct?
- **Prediction:** no prediction was written before the run. Stated after the run: I expected it to be only slightly faster than the CPU, and only at larger batch sizes.
- **Command:** `python -m pytest tests/test_cpu_gpu_equivalence.py -k numba -v`, then `python benchmarks/sweep_median.py`
- **Result:**
  - **Correctness:** `test_median_filter_numba_equivalence` PASSED on Colab T4 (same rule as the PyTorch median test); also 0% of pixels beyond 1 gray level in Numba's CUDA simulator on 20×20 and 23×17 images (23×17 exercises out-of-image threads).
  - **Speed:** DRAFT run on Colab, Tesla T4 (official run: P1, RTX 3050), ms per image:

  | Batch | CPU (OpenCV) | PyTorch (A4.8) | Numba (A4.11) |
  |---|---|---|---|
  | 1 | 0.76 | 4.80 | 1.33 |
  | 8 | 0.80 | 3.79 | 0.51 |
  | 32 | 0.84 | 2.42 | 0.56 |
  | 64 | 0.84 | 2.43 | 0.56 |
  | 128 | 1.00 | 2.44 | 0.55 |

- **Prediction vs result:** better than expected. The Numba kernel beats OpenCV from batch size 8 on (1.5–1.8× faster) and is about 4.4× faster than the PyTorch version from batch 32 on. At batch size 1 the CPU still wins (launch overhead).
- **Explanation:**
  - **Implementation** (`src/filters_numba.py`, wrapper `filters_gpu.median_filter_numba`): one thread per output pixel, 16×16 blocks, one grid layer (`blockIdx.z`) per image-channel slice. Each thread reads its 5×5 neighbours with coordinates clamped to the image (= replicate padding), sorts them in a private `cuda.local.array` (insertion sort) and writes the middle value. Window fixed at 5×5.
  - **Why it is faster:** fusion avoids the 25× memory copy. Each thread reads its neighbours straight into its private array, instead of `F.unfold` writing a 25× larger tensor to GPU memory and `torch.median` reading it back, and there is one kernel launch instead of three. Same GPU and same idea (sort 25 values per pixel), so the gain comes from less memory traffic. The GPU is saturated from batch 8 on (flat curve), earlier than the PyTorch version (batch 32), since there is less work per image. It wins over OpenCV despite using a simple insertion sort.
- **Decision:** the fused kernel is the GPU median filter worth keeping. The median filter is not used in training, so this does not speed up the pipeline itself; it shows why kernel fusion matters. Possible next step: a faster sort (e.g. only partially sorting up to the 13th value). Re-run on P1's RTX 3050.
- **Prediction for the official rerun (P1, RTX 3050 laptop, with Numba installed), written before it:** the Numba kernel won't be much faster than the laptop's CPU, maybe about a tie, because the laptop's CPU is newer and has more cores than Colab's (OpenCV: 0.68 vs 1.10 ms at batch 128) while its GPU has half the SMs (the PyTorch median is about 19% slower on it: 2.91 vs 2.45 ms; the same slowdown would put Numba at about 0.65 ms vs OpenCV 0.68 ms).

### A6.6 Model size: parameters and FLOPs, reference CNN vs ResNet-18 (P4: Lina) — 2026-11-11

- **Config / commit:** none (script defaults); `src/model.py` and `benchmarks/model_size.py` at commit `55b9a33`. ResNet-18 uses `ResNet18_Weights.DEFAULT` (= `IMAGENET1K_V1` in this torchvision version), last layer replaced by `nn.Linear(512, 5)`.
- **Question:** how many parameters and how many FLOPs per 224×224 image do the reference CNN and the pretrained ResNet-18 (5-class last layer) have?
- **Prediction:** ResNet-18 has about 12× more parameters than the reference CNN, and needs more FLOPs per image too. (Whether the FLOP gap is bigger or smaller than the parameter gap: not predicted.)
- **Command:** `python benchmarks/model_size.py` (Colab, CPU; one 224×224 image; `torch.utils.flop_counter.FlopCounterMode`, which counts 2 FLOPs per multiply-add)
- **Result:**

  | Model | Parameters | GFLOPs per image |
  |---|---|---|
  | Reference CNN | 0.62 million | 1.01 |
  | ResNet-18 | 11.18 million | 3.63 |
  | Ratio | 18× | 3.6× |

- **Prediction vs result:** ResNet-18 has 18× more parameters (predicted 12×: right direction and order of magnitude) and 3.6× more FLOPs. The FLOP gap is much smaller than the parameter gap.
- **Explanation:**
  - **A weight's cost depends on how many times it is used.** Convolution weights are reused at every position of the feature map, so they are expensive on large maps and cheap on small ones.
  - **ResNet-18:** as we go through the layers, it runs on smaller and smaller maps (56 → 28 → 14 → 7). Stage 4 (512 channels, 7×7 maps) holds about 8.4 M of the 11.2 M parameters (75%) but only about 0.8 GFLOPs (23%). Its stem also shrinks the image to 56×56 right away.
  - **Reference CNN:** about 85% of its parameters are in one fully connected layer (`Linear(2048, 256)`, 524,544 weights), applied once per image, so almost free. Its convolutions have few weights but run on large maps: the first one (896 parameters) runs on the full 224×224 image.
- **Decision:** keep ResNet-18 as the justified alternative model. Expected costs: about 3.6× more compute per image for training and inference, and 45 MB of weights (11.2 M × 4 bytes), which fits easily in GPU memory. 18× more parameters on 3,000 images could overfit; pretraining on ImageNet should limit that. Next: compare both models in A6.9 with the same splits and training budget, measuring epoch time, GPU memory and validation macro-F1.

### A6.9 Redesign experiments: model, input representation, training strategy (P4: Lina) — 2026-11-11

- **Config / commit:** `configs/baseline.yaml` and the 7 `configs/exp_*.yaml` (A6.8) at commit `4354816`. All use `split_seed: 42` (identical train/val/test images: 2116 / 425 / 458) and 5 epochs. Dataset: 2,999 images (image `2T2SALA702` is missing from the Zenodo archive), class counts 504 / 701 / 583 / 632 / 579, so **not balanced**, contrary to the brief.
- **Question:** which single change to the model, the input representation or the training strategy improves validation macro-F1, and what does each change cost in epoch time?

  | Run | One change | Compared with |
  |---|---|---|
  | `baseline_colab` | (reference) | — |
  | `exp_baseline_seed43` | seed 42 → 43 (control: run-to-run variation) | baseline |
  | `exp_resnet18` | model → resnet18 | baseline |
  | `exp_resnet18_lr1e-4` | lr 0.001 → 0.0001 | exp_resnet18 |
  | `exp_no_clahe` | clahe → false | baseline |
  | `exp_res160` | image size 224 → 160 | baseline |
  | `exp_augment` | augment → true | baseline |
  | `exp_cosine` | scheduler → cosine | baseline |

- **Prediction:** (written before the 7 experiment runs, after the baseline run)
  - **Best macro-F1:** `exp_resnet18`.
  - **Fastest per epoch:** `exp_no_clahe`.
  - **Slowest per epoch:** `exp_augment`, because the baseline run showed the pipeline is CPU-bound: augmentation adds CPU work, while ResNet-18's extra work is on the GPU, which is already waiting. (My first guess was the baseline; I revised it before any experiment ran, after noticing that several runs do strictly more work than the baseline.)
  - **Worst macro-F1:** `exp_no_clahe` and `exp_res160`, since less contrast and smaller images mean fewer visible details.
- **Command:** `python src/train.py --config configs/<config>.yaml --device cuda --run <run>` on Colab, Tesla T4.
- **Result:** DRAFT runs on Colab T4 (unoptimized pipeline; official runs later on P1's RTX 3050). `baseline_colab` ran on 2026-11-11, all the others in one session on 2026-11-12, so for speed the reference is `exp_baseline_seed43` (same work as the baseline). "Best val F1" is the best epoch; `train.py` saves the epoch with the best val **accuracy**, shown separately when it differs.

  | Run | One change | Best val F1 (epoch) | Saved checkpoint F1 (epoch) | Train time / epoch | GPU memory | Final train loss / acc |
  |---|---|---|---|---|---|---|
  | `baseline_colab` | — | 0.353 (4) | same | 120 s | 868 MB | 1.48 / 0.34 |
  | `exp_baseline_seed43` | seed 43 | 0.354 (4) | same | 125 s | 868 MB | 1.46 / 0.37 |
  | `exp_resnet18` | model → resnet18 | 0.514 (0) | same | 130 s | 921 MB | 0.90 / 0.61 |
  | `exp_resnet18_lr1e-4` | lr → 0.0001 (vs exp_resnet18) | 0.550 (4) | 0.527 (0) | 126 s | 921 MB | 0.16 / 0.96 |
  | `exp_no_clahe` | clahe → false | 0.332 (4) | same | 121 s | 868 MB | 1.48 / 0.34 |
  | `exp_res160` | image size → 160 | 0.321 (1) | same | 114 s | 423 MB | 1.46 / 0.35 |
  | `exp_augment` | augment → true | 0.296 (1) | 0.291 (2) | 128 s | 868 MB | 1.48 / 0.35 |
  | `exp_cosine` | scheduler → cosine | 0.342 (4) | same | 126 s | 868 MB | 1.48 / 0.34 |

  Validation (425 images) takes about 26 s per epoch in every run. Run-to-run variation (baseline vs seed 43, same settings): best F1 almost identical (0.353 vs 0.354), but epoch-by-epoch F1 differs by up to 0.08 (0.21 → 0.29 → 0.30 → 0.30 → 0.35 vs 0.18 → 0.20 → 0.32 → 0.32 → 0.35). With lr 0.001, ResNet-18 is unstable (F1 0.51 → 0.29 → 0.45 → 0.41 → 0.33).
- **Prediction vs result:**
  - **Best F1: ResNet-18:** right, by far (0.51–0.55 vs 0.35).
  - **Fastest per epoch: no CLAHE:** wrong. `exp_res160` is the fastest (114 s); no CLAHE is second (121 s).
  - **Slowest per epoch: augmentation:** about a tie with ResNet-18 (128 s vs 130 s).
  - **Worst F1: no CLAHE and res160:** partly wrong. Augmentation is the worst (0.30), then res160 (0.32), then no CLAHE (0.33), but these differences are within the run-to-run variation.
- **Explanation:**
  - **Only ResNet-18's difference is clearly real;** the rest could be luck. The seed-43 control shows epoch-to-epoch swings of up to 0.08 with identical settings, so differences of 0.01–0.03 (no CLAHE, res160, cosine) are not conclusive with one seed each, while ResNet-18's gain (+0.16 to +0.20) is far beyond it.
  - **Image loading is what takes most of the time.** Every run decodes the full-size original photos before resizing, identically, so the epoch time barely changes (114–130 s): no CLAHE saves only 3%, 160×160 images only 9%. ResNet-18 needs 3.6× more FLOPs but only +4% time, because the GPU was waiting for the CPU anyway: its extra work fills part of the GPU's idle time (to be measured with `profile_gpu.py`).
  - **The reference CNN is not overfitting, it is underfitting:** its final training accuracy (0.34) is even below its validation accuracy (0.39). Augmentation makes training harder on purpose, which slows down a model that is still struggling to learn, so it hurt here. ResNet-18 with lr 0.0001 overfits (training accuracy 0.96 vs validation 0.55), so that is where augmentation could help.
  - **lr 0.001 is too large for the pretrained ResNet-18:** it is best after the first epoch and then gets worse, partly destroying the ImageNet weights; lr 0.0001 keeps it stable.
- **Decision:**
  - **Best model so far: ResNet-18 with lr 0.0001** (val macro-F1 0.55), with two caveats: one seed only, and its saved checkpoint is epoch 0 (F1 0.527) because `train.py` selects by val accuracy. Propose to the team to select the checkpoint by val macro-F1 (our main metric, since the classes are not balanced), for all runs.
  - **Next experiment:** `exp_resnet18_lr1e-4_augment` (one change vs `exp_resnet18_lr1e-4`: augment → true), to fight its overfitting.
  - **For the team:** decoding is the main cost of every epoch, which supports P2's decoded-image cache (A3).

#### A6.9 follow-up: `exp_resnet18_lr1e-4_augment` (P4: Lina) — 2026-11-12

- **Config / commit:** `configs/exp_resnet18_lr1e-4_augment.yaml` at commit `9564746`; one change vs `exp_resnet18_lr1e-4`: `preprocess.augment: false -> true`.
- **Question:** does augmentation reduce ResNet-18's overfitting (lr 0.0001: final train acc 0.96 vs val acc 0.55), and does it improve val macro-F1?
- **Prediction:** val macro-F1 above 0.55; epoch time about +2% (extra CPU work); lower training accuracy (training with augmentation is harder); a smaller train–val gap.
- **Command:** `python src/train.py --config configs/exp_resnet18_lr1e-4_augment.yaml --device cuda --run exp_resnet18_lr1e-4_augment` (Colab T4, new session).
- **Result:** DRAFT run on Colab T4.

  | | `exp_resnet18_lr1e-4` | `exp_resnet18_lr1e-4_augment` |
  |---|---|---|
  | Best val macro-F1 (epoch) | 0.550 (4) | 0.553 (3) |
  | Final train accuracy | 0.955 | 0.775 |
  | Final val accuracy | 0.553 | 0.569 |
  | Train–val accuracy gap | 0.40 | 0.21 |
  | Train time / epoch | 126 s | 123 s (different session) |

  Per epoch with augmentation: val F1 0.48 → 0.50 → 0.50 → 0.55 → 0.55; train accuracy 0.49 → 0.61 → 0.68 → 0.72 → 0.78.
- **Prediction vs result:** lower training accuracy and smaller gap: right (the gap is halved). Val macro-F1 0.553 is above 0.55, but +0.003 is within the run-to-run variation, so no real improvement. Epoch time: −2% instead of +2%, but the run was in a different Colab session, so the times are not comparable.
- **Explanation:** augmentation did what it is for: the model memorizes much less (train accuracy 0.78 instead of 0.96), but within 5 epochs this did not turn into better validation F1. The model was still improving at the end (train accuracy rising, best val accuracy at epoch 4): with augmentation, a model usually needs more epochs to benefit.
- **Decision:** keep augmentation as an option for the final ResNet-18 recipe (it halves overfitting at almost no cost). Hypothesis to test if time allows: with a longer training budget for both runs, augmentation should improve val macro-F1.

### A7.2–A7.3 GPU profiler: test run on the baseline pipeline (P4: Lina) — 2026-11-12

- **Config / commit:** `configs/baseline.yaml`; `benchmarks/profile_gpu.py` at commit `8960169` (after two fixes, see Explanation).
- **Question:** during steady-state training of the unoptimized baseline, what fraction of the time is the GPU idle, and how much of its busy time is host-to-device copies?
- **Prediction:** a batch of 32 arrives about every 1,900 ms (about 17 img/s); one training step of the reference CNN is about 100 GFLOPs, about 33 ms on a T4 (about 3,000 GFLOP/s in practice). So the GPU idle fraction should be about 0.98. Copies should be a small part of the busy time (about 4 ms of copy per batch from A4.14, vs about 33 ms of kernels: about 10%).
- **Command:** `python benchmarks/profile_gpu.py --config configs/baseline.yaml --out results/torch_profiler_baseline` (Colab T4; 5 warm-up + 15 profiled steps). `trace.json` is not committed (`.gitignore`: large file).
- **Result:** DRAFT on Colab T4 (`results/torch_profiler_baseline/summary.json`):

  | | Prediction | Measured, per step |
  |---|---|---|
  | Time per step | ~1,900 ms | 1,677 ms |
  | GPU kernels | ~33 ms | 68 ms |
  | Host-to-device copy | ~4 ms | 11 ms (float32 from pageable memory, 1.76 GB/s) |
  | GPU busy (union of all GPU intervals) | — | 78 ms |
  | Copies' share of busy time | ~10% | 14% |
  | **GPU idle fraction** | **~0.98** | **0.953** |

  Kernels + copies (1,187 ms over 15 steps) slightly exceed the busy time (1,174 ms): 13 ms where a copy and a kernel overlapped, counted once by `union_length`.
- **Prediction vs result:** the main prediction holds: the GPU is idle 95% of the time. The kernels take about 2× the estimate (68 vs 33 ms): 3,000 GFLOP/s was optimistic for a small network whose layers don't fill the GPU. The copy is about 3× slower than in the A4.14 benchmark (11 ms vs 4 ms; 1.76 vs 4.7 GB/s from pageable memory).
- **Explanation:**
  - **The GPU waits for the CPU:** the CPU takes time decoding the images, so the GPU finishes its 78 ms of work and then stays idle while waiting for the next batch (about 1.6 s). This directly confirms the A6.9 experiments, where changing the GPU's workload barely changed the epoch time.
  - **The copy is slower during training** than in the isolated A4.14 benchmark, because a copy from pageable memory needs the CPU (an extra staging copy), and the 2 CPU cores are busy loading data.
  - **Two bugs fixed in the script before this result** (both in the profiler setup, not in the measurement): the default `schedule` repeats (`repeat=0`), so the recorded steps were discarded (first run: idle 1.0) → `repeat=1`; and the `ProfilerStep#N` labels on the GPU timeline were counted as kernels (second run: kernel time 1,783 ms > busy time 1,353 ms, impossible) → excluded.
- **Decision:**
  - **P2's work should help the most:** the decoded-image cache (A3) removes the repeated decoding, the main cost.
  - **For the copy:** copy uint8 batches from pinned memory, as A4.14 showed: pinned memory is P2's DataLoader option (A5.1), and copying uint8 needs P3's GPU preprocessing plugged in by P2's hook (A5.3).
  - **Next:** run the same script on `configs/optimized.yaml` once the team's work is merged (official runs on P1's RTX 3050, baseline included), to see whether the idle fraction drops and the bottleneck moves (A7.7).
