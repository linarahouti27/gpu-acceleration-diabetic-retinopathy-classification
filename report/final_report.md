# GPU Acceleration of a Diabetic Retinopathy Classification Pipeline - Final Report

> Skeleton following the required outline (section 8.B of the brief). Replace every
> `<...>` with measured values. Target length 10-15 pages.

## 1. Abstract and objectives

<Problem, methods, headline results, main conclusion. Example sentence: "The
principal bottleneck was <stage> (p = <x>% of baseline runtime); the optimized
pipeline achieves <S>x end-to-end speedup at <acc> test accuracy vs <acc_base>
for the baseline.">

## 2. Problem and pipeline description

Dataset: 3,000 fundus images, 5 balanced DR classes; patient-safe splits.
Pipeline stages: loading -> preprocessing (CLAHE, circular crop, normalize) ->
filters (Gaussian, median, green channel, contrast stretch) -> augmentation ->
CNN training -> evaluation. Reference model: 3-block CNN (see `src/model.py`).

## 3. Experimental environment (Table 1)

| Item | Value |
|---|---|
| CPU | <model, cores, freq> |
| GPU | <model, VRAM> |
| RAM | <GB> |
| OS | <distro/version> |
| Python | <version> |
| PyTorch / CUDA / cuDNN | <versions> |
| Image resolution / batch size / splits | 224x224 / <B> / 70-15-15 patient-level |

## 4. Baseline profiling

<Methodology: cProfile + PyTorch Profiler + manual stage timers. Insert
Figure 1 (runtime breakdown) and the ranked bottleneck table. State the
principal bottleneck and the evidence.>

| Rank | Stage | Time (s) | % of total | Evidence |
|---|---|---|---|---|
| 1 | <stage> |  |  |  |
| 2 |  |  |  |  |

## 5. Redesign and optimization decisions

For each change: motivation (linked to section 4 measurement), implementation,
expected effect, trade-off.

| # | Change | Motivation (measured) | Expected | Trade-off |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

## 6. Benchmarking methodology

Warm-up: <n> iterations. Timing boundary: <start/end>. Synchronization:
`torch.cuda.synchronize` before/after. Repeats: <n>, reported as mean +/- std.
Transfers: included in end-to-end timing.

## 7. Results

Table 2 - Stage-level performance (fill from `results/timing.csv`):

| Pipeline stage | CPU baseline (s) | Optimized (s) | Speedup |
|---|---|---|---|
| Image loading |  |  |  |
| Preprocessing |  |  |  |
| Filtering (Gaussian/median/green/contrast) |  |  |  |
| Augmentation |  |  |  |
| Training per epoch |  |  |  |
| Evaluation |  |  |  |
| Total pipeline |  |  |  |

Table 3 - Classification quality (fill from `results/<run>/metrics.json`):

| Metric | Baseline model | Final model |
|---|---|---|
| Accuracy |  |  |
| Macro-precision |  |  |
| Macro-recall |  |  |
| Macro-F1 |  |  |
| Confusion matrix |  |  |

Figures: Fig 1 baseline breakdown; Fig 2 speedup comparison; Fig 3 accuracy vs
execution time; Fig 4 ablation; Fig 5 CPU/GPU utilization + memory.

Amdahl analysis: with p = <fraction> of runtime accelerated by s = <x>,
S_total = 1 / ((1 - p) + p/s) = <x>. <Discuss why the measured end-to-end
speedup does or does not match.>

## 8. Discussion and critical analysis

<Why did some stages benefit more? Transfer/launch overhead? Did the bottleneck
shift? Were isolated gains reflected end-to-end? Algorithmic vs hardware gains.
What changed in macro-F1 / per-class recall? Was final eval untouched by model
selection? Memory/portability trade-offs? Would this transfer to other GPUs?>

## 9. Limits and threats to validity

*Draft by P4 (Lina), 2026-11-24. Official numbers come from P1's RTX 3050 laptop; numbers marked
"Colab" are exploratory drafts (`results/colab_t4/`).*

### 9.1 Dataset

- **Not the dataset described in the brief.** The FDR26 archive contains **2,999 images**, not
  3,000 (image `2T2SALA702` is listed in the metadata but missing from the archive), and the five
  grades are **not balanced** (504 / 701 / 583 / 632 / 579 images), whereas the brief describes a
  balanced dataset. We report **macro-F1** as the main quality metric and select checkpoints on it,
  so that the most frequent grade cannot dominate the result.
- **Small size.** After the patient-level split, only 2,116 images are used for training, i.e.
  about 400 to 700 per grade, and 425 for validation. This is enough for an educational benchmark
  but not to establish how a model would generalize.
- **A single source.** All images come from one dataset; image quality, cameras, acquisition
  conditions and patient populations in other hospitals could lead to different results.
- **Leakage is controlled, not excluded.** Images of the same patient always stay in the same
  split (verified in A1.4), which prevents the most common source of inflated scores; other
  correlations between images (e.g. the same camera session) cannot be ruled out.

### 9.2 Model and clinical use

- **This is not a diagnostic tool.** The best model so far (ResNet-18 pretrained on ImageNet,
  lr 0.0001) reached a validation macro-F1 of 0.55 in the Colab drafts, against 0.35 for the
  reference CNN: a clear improvement, but far from what a clinical tool would require.
- **Overfitting.** The ResNet-18 reaches 96% training accuracy against about 55% on validation;
  augmentation halves this gap but did not improve macro-F1 within the 5-epoch budget (A6.9).
- **Scores are not clinical utility.** A higher macro-F1 does not by itself mean better clinical
  usefulness; we did not study per-grade sensitivity or the cost of specific errors (e.g. missing
  a severe case versus over-grading a mild one).

### 9.3 GPU and performance measurements

- **Results depend on the hardware.** All official numbers come from one laptop (i5-13420H,
  RTX 3050 Laptop 6 GB, Windows 11). The balance between CPU and GPU changes from one machine to
  another: our GPU median filter was about **2× slower** than OpenCV on Colab (Tesla T4, 2 shared
  CPU cores) but about **4× slower** on the laptop (faster CPU, GPU with half the SMs) (A4.13).
  A different, e.g. more recent, GPU would shift the balance again; our speedups cannot be
  transferred as such to another machine.
- **Measurement conditions matter.** A second run of the same median sweep on the laptop, in
  different power or thermal conditions, was about 60% slower on the GPU; it was discarded and all
  official runs follow the same protocol (plugged in, best performance mode, nothing else running).
- **Laptop constraints.** The laptop's RAM (8 GB, already 7.4 GB in use during the baseline, A2.8)
  limits the number of DataLoader workers and the amount of pinned memory, and its 6 GB of GPU
  memory limits the batch size. `torch.compile` could not be used because it requires Triton,
  which is not available on Windows (A6.3).
- **Not every component ran everywhere.** The custom Numba CUDA median kernel (A4.11) could not run
  on the laptop: the available numba-cuda (0.30.4) is incompatible with NumPy 2.5.4, and NumPy was
  not downgraded to keep the official environment unchanged. Its results come from Colab only.
- **Numerical differences.** GPU and CPU results are not bit-identical: the GPU CLAHE uses a
  different color-space approximation than OpenCV (tested for a luminance correlation above 0.95),
  mixed precision (AMP) computes in 16-bit, and GPU convolutions may use TF32. We check that
  results stay within the test tolerances and that quality differences stay within the
  seed-to-seed variation, not that they are identical.

### 9.4 Methodology

- **One run per configuration.** Each experiment was run with a single seed. A control run with a
  different seed (A6.9) showed epoch-to-epoch differences of up to 0.08 in validation macro-F1 for
  identical settings, so differences smaller than that (CLAHE off, 160×160 images, cosine schedule)
  cannot be distinguished from chance; only the ResNet-18 gain (+0.16 to +0.20) is clearly above it.
- **Short training budget.** All models were trained for 5 epochs and had not converged; longer
  training could change the ranking of some configurations (e.g. augmentation).
- **Checkpoint selection is optimistic.** The saved model is the epoch with the best validation
  macro-F1, which slightly overestimates its validation score; the test set is therefore used only
  once per final model (A7.8), never for selection.
- **Profiling changes what it measures.** cProfile requires `num_workers=0` and adds overhead: the
  profiled epoch took 167.7 s against 112.3 s for the real baseline epoch, so the profile gives
  shares of time, not absolute durations.
- **Drafts and official numbers.** Part of the exploration ran on Colab (shared CPU, random GPU
  allocation); these drafts guided the decisions but only the laptop runs are used for the final
  comparisons.
- **Predictions were not always written before the run.** Most predictions in the lab notebook were
  recorded before measuring, but some were not (e.g. A4.11, P4's A2.3 ranking, A2.6); the notebook
  states it explicitly in each case.

### 9.5 Scope

Within the project's duration we did not: compare several GPU architectures on the final pipeline,
evaluate more models or longer training budgets, repeat each configuration with several seeds,
optimize inference separately from training, try GPU image decoding (e.g. NVIDIA DALI), or measure
energy consumption. These are natural next steps (section 10).

## 10. Conclusion and future work

<Contribution, strongest evidence, lessons learned. Future work: model
compilation/operator fusion, inference-only optimization, energy measurement,
performance model for CPU-vs-GPU dispatch.>
