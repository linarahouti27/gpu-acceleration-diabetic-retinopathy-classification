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
Add-Content -Path docs\lab_notebook.md -Value @"

### A1.4 Data split verification (P1: Fatima Zahrae )          date: 2026-11-12
Config / commit:      configs/baseline.yaml, commit $commit
Question:             Are the classes still balanced inside each split? Does any patient
                      appear in two splits?
Prediction:           (none written before the run)
Command:              python check_split.py (the brief's script + an overlap test)
Result:               split  images  patients   class 0 / 1 / 2 / 3 / 4
                      train   2116     1342     343 / 489 / 418 / 453 / 413
                      val      425      287      83 / 104 /  76 /  78 /  84
                      test     458      289      78 / 108 /  89 / 101 /  82
                      Shared patients: train & val = 0, train & test = 0, val & test = 0.
                      Total: 2116 + 425 + 458 = 2999 images, as expected.
Prediction vs result: -
Explanation:          Balance: the full dataset is almost balanced (504 / 701 / 583 / 632 / 579).
                      The share of each class stays close across the three splits (class 1:
                      23.1% in train, 24.5% in val, 23.6% in test). The largest gaps are class 0
                      (16.2% in train, 19.5% in val) and class 3 (21.4% in train, 18.4% in val),
                      probably because val is small (425 images). The splits are comparable.
                      Data leakage: no patient id is in two splits.
                      Why patient-level: images of the same patient look alike (same person,
                      same camera, same anatomy). With an image-level split, near-duplicates
                      would be in both training and validation. The model could recognize the
                      patient instead of learning the disease, and the validation accuracy
                      would be too optimistic. With a patient-level split, it better estimates
                      the performance on new patients. I did not measure by how much it would
                      be inflated.
Decision:             Keep this split for all comparisons (data.split_seed fixed).
Add-Content -Path docs\lab_notebook.md -Value @"

### A1.5 torch.backends.cudnn.benchmark setting (P1: Fatima zahrae)          date: 2026-11-12
Config / commit:      configs/baseline.yaml, commit $commit
Question:             How should torch.backends.cudnn.benchmark be set for the baseline and for
                      the benchmarks, and why?
Prediction:           (none written before the decision)
Command:              Select-String -Path src\*.py -Pattern "cudnn"
Result:               The setting only matters on the GPU: it makes cuDNN try several
                      convolution algorithms the first time it sees an input shape, then keep the
                      fastest one.
Prediction vs result: -
Explanation:          Baseline (CPU): the baseline runs on the CPU, where cuDNN is not used, so
                      the flag has no effect. Keep the default (False) so the baseline stays
                      simple and reproducible.
                      GPU benchmarks: use True, because the input shape is fixed (224x224, same
                      batch size). Two precautions: (1) the first pass is slower because the
                      algorithms are tested, so a warm-up is needed and the cold first epoch is
                      reported separately; (2) the chosen algorithm can change between runs, so
                      results are not always identical to the last bit. For correctness tests
                      (CPU/GPU equivalence) use a tolerance or set it to False.
                      A different last batch size triggers a new search, which is one more
                      reason to use drop_last=True on the train loader.
Decision:             False for the CPU baseline, True for the GPU benchmarks. The same value
                      must be used in every comparison, and it is written in the config file.

### A1.7 / A1.8 CPU baseline run (P1: Fatima Zahrae)          date: 2026-11-12
Config / commit:      configs/baseline.yaml, commit dd9e6c7
Question:             What are the validation accuracy and the time per epoch of the CPU baseline?
Prediction:           (none written before the run)
Command:              python src/train_cpu.py
Result:               RTX 3050 laptop, CPU i5-13420H, charger plugged in, Best performance.
                      5 epochs, best val_acc = 0.3859 (val_f1 at the last epoch = 0.3356),
                      wall time = 564.4 s, about 23 img/s.
                      train_time per epoch: 93.41, 93.55, 91.72, 90.65, 95.72 s.
                      Steady epochs (1 to 4): mean 92.9 s +/- 2.2 s. Cold epoch (0): 93.41 s.
                      setup_time_s = 2.515521000023 s
Prediction vs result: -
Explanation:          One-time costs: setup (2.52 s) and the cold first epoch. The first epoch is
                      not slower than the others (93.41 s vs 92.9 s average), so there is no
                      visible cold-start cost on CPU. Repeated cost: about 93 s per epoch, paid
                      at every epoch.
                      The 5 train_time values add up to 465 s but the wall time is 564 s. Setup
                      explains only 2.5 s of the 99 s gap. The rest is probably validation after
                      each epoch (about 19 s per epoch), not confirmed yet.
Decision:             Keep this run as the baseline everything is compared against. Test set not
                      used (evaluate.py is for A7).
Add-Content -Path docs\lab_notebook.md -Value @"

### A1.10 Checkpoint questions (P1: $name)          date: 2026-11-15
Config / commit:      configs/baseline.yaml, commit $commit

Q1. Which stages run once, and which run every epoch?
  Run once:           listing the images and splitting by patient, building the model and the
                      loaders. Measured setup_time_s = 2.52 s.
  Run every epoch:    loading (decode + resize), preprocessing (circular crop, CLAHE,
                      normalize), forward, backward, optimizer step, and the validation pass
                      (val_acc is printed after every epoch). Loading is 63.5% of an epoch in
                      the profile, because the baseline has no cache yet. Augmentation is off
                      in the baseline (augment=False).
  In the optimized design: decoding becomes a one-time cost thanks to the decoded-image cache
  (A3).

Q2. How many bytes does one batch occupy as uint8 and as float32 at 224x224?
  One image has 224 x 224 x 3 = 150528 values.
  uint8 (1 byte each):    150528 bytes per image.
  float32 (4 bytes each): 602112 bytes per image.
  With batch size $B:     uint8 = $u8 bytes ($u8mb MB), float32 = $f32 bytes ($f32mb MB).
  float32 is always 4 times larger than uint8, so copying uint8 to the GPU and converting to
  float32 there moves 4 times fewer bytes. This assumes 3 channels.

Q3. What does drop_last=True on the train loader change in the throughput numbers?
  With 2116 training images and batch size $B there are $full full batches. drop_last=True
  removes the last incomplete batch, so $drop images are skipped in an epoch ($pct% of 2116).
  The effect on images per second is tiny: it is at most about that percentage. All batches
  have the same shape, which avoids a new cudnn.benchmark search and torch.compile
  recompilation, and avoids a batch of size 1 for BatchNorm. If the shuffle changes each
  epoch, the skipped images differ every epoch, so no image is lost for good.

### A4.12 Filter test tolerances (P4: Lina)          date: 2026-11-09
Scope:            the three filter tests in `tests/test_cpu_gpu_equivalence.py` (the preprocessing
                  tests are covered by P3). 1 gray level = 1/255 ≈ 0.004.
Measured so far:  checked with a NumPy re-implementation of each GPU function against
                  `src/filters_cpu.py` (no CUDA machine available yet).

Gaussian blur (every pixel within 0.02):
    Too loose. 0.02 is about 5 gray levels, while my blur differs from OpenCV by only 3e-7.
    A slightly wrong blur (wrong sigma, wrong border mode) could still pass. Some margin is
    needed because the GPU computes convolutions in lower precision (TF32), but about
    1 gray level (0.004) would be enough.

Median filter (at most 2% of pixels may differ by more than 1 gray level):
    The 1-gray-level allowance is necessary: the CPU version converts to uint8, which rounds
    every value down to a whole gray level, so it can never be more than 1 gray level away
    from an exact float result. But 2% (about 1,000 pixels of a 224x224 image) is too generous:
    my version had 0% of pixels beyond 1 gray level. A tiny margin for floating-point rounding
    is fair (e.g. 0.1%). Also, the test image is black at the borders, so a wrong padding
    mode (reflect instead of replicate) would still pass.

Contrast stretch (every pixel within 0.001):
    About right. 0.001 is a quarter of a gray level, so invisible, and my version differs by
    only 6e-8 because CPU and GPU compute the percentiles the same way (linear interpolation).
    It could be tightened slightly (e.g. 1e-4), but not much more: a tolerance too close to
    zero could fail a correct GPU version just because of floating-point rounding differences.

Green channel (no test):
    This is a problem: nothing would warn us if the function returned the wrong channel
    (e.g. t[:, 0:1], the red one). A test should check the shape (N,1,H,W), that the values
    equal the green channel t[:, 1], and that the result is a view (same memory, no copy).

GPU test run:     Colab, Tesla T4, Python 3.13.16, pytest 8.4.2
                  `python -m pytest tests/test_cpu_gpu_equivalence.py -k "gaussian or median or contrast" -v`
                  test_gaussian_filter_equivalence   PASSED
                  test_median_filter_equivalence     PASSED
                  test_contrast_stretch_equivalence  PASSED
                  3 passed, 6 deselected in 3.29s

### A4.13 Median filter: GPU vs CPU time per image across batch sizes (P4: Lina)   date: 2026-11-10
Question:             At which batch size (1, 8, 32, 64, 128) does the GPU median filter become
                      faster per image than the CPU (OpenCV) one, if ever?
Prediction:           At batch size 1 the GPU is slower than the CPU, because the fixed kernel
                      launch overhead dominates. As the batch grows, the GPU time per image drops
                      (the overhead is shared by more images) and goes below the CPU time at
                      about batch size 64. The CPU time per image stays roughly constant.
Command:              python benchmarks/sweep_median.py   (default: batch sizes 1 8 32 64 128,
                      5 repeats, 2 warm-up runs, CUDA synchronized around every timed run;
                      the batch is already on the GPU, so the host-to-device copy is excluded)
Result:               DRAFT run on Colab, Tesla T4 (official run: P1, RTX 3050). ms per image:
                        batch    CPU (OpenCV)   GPU (torch)
                          1         0.73           3.58
                          8         0.82           2.76
                         32         0.84           2.44
                         64         1.11           2.44
                        128         1.10           2.45
                      Colab's CPU is shared, so the CPU times are noisy.
Prediction vs result: I predicted the GPU would overtake the CPU at batch size 64, but the CPU
                      won at every batch size (about 2x faster at 128). The first part of the
                      prediction was right: the GPU time per image drops from 3.58 to 2.44 ms as
                      the launch overhead is shared by more images. But it becomes flat from
                      batch 32 on: the GPU is then fully busy and 2.44 ms is the cost of the work
                      itself, so bigger batches cannot help.
Explanation:          The two versions do not do the same amount of work. OpenCV's medianBlur is a
                      specialized algorithm tuned for 8-bit images that avoids sorting most values.
                      My GPU version uses general operations: F.unfold copies each pixel's 25
                      neighbours into memory (25x the input size, about 1.9 GB for 128 images),
                      then torch.median sorts those 25 values once for every pixel. The extra memory
                      traffic and generic sorting cost more than the GPU's parallelism gains.
Decision:             Do not use this GPU median filter as is (it is not used in training anyway).
                      To make it faster, fuse the steps into one custom kernel (e.g. Numba CUDA,
                      A4.11): each thread reads its 25 neighbours directly and sorts them in
                      registers, which removes the 25x copy. Tuning blocks / threads per block is a
                      secondary improvement. Re-run on P1's RTX 3050 for the official numbers.

### A4.14 Host-to-device copy: uint8 vs float32 batch (P4: Lina)   date: 2026-11-10
Question:             How long does copying one batch of images from CPU memory to the GPU take
                      as uint8 vs as float32? (Second part, how much of the GPU preprocessing
                      time is the copy, needs P3's preprocess_batch_gpu.)
Prediction:           uint8 should be about 4x faster to copy, because it has 4x fewer bytes to
                      move (1 byte per value instead of 4).
Command:              python benchmarks/h2d_copy.py   (batch sizes 1 8 32 128, 10 repeats, 2 warm-up
                      runs, CUDA synchronized; pinning is done before timing, only the copy is timed)
Result:               DRAFT run on Colab, Tesla T4 (official run: P1, RTX 3050). Copy time in ms:
                        batch | uint8 normal  float32 normal  ratio | uint8 pinned  float32 pinned  ratio
                          1   |    0.083          0.214        2.6x |    0.041          0.079       1.9x
                          8   |    0.381          1.102        2.9x |    0.133          0.443       3.3x
                         32   |    1.185          4.090        3.5x |    0.427          1.629       3.8x
                        128   |    4.145         16.959        4.1x |    1.632          6.328       3.9x
                      Bandwidth for large batches: about 4.6 GB/s (normal) vs about 12 GB/s (pinned).
Prediction vs result: uint8 is faster than float32 at every batch size. The predicted 4x holds for
                      large batches (3.9-4.1x at 128), but not for small ones (1.9-2.6x at batch 1):
                      every copy has a fixed start-up cost that dominates when the batch is tiny.
Explanation:          The copy time depends on the number of bytes, not on the type: uint8 at batch
                      128 and float32 at batch 32 are both 19.27 MB and take the same time (4.15 vs
                      4.09 ms normal, 1.63 vs 1.63 ms pinned). Pinned memory is about 2.6x faster for
                      both types because the driver does not need an extra copy into a fixed staging
                      area first; at about 12 GB/s it is close to the PCIe limit of the T4.
Decision:             Copy the batches as uint8 from pinned memory and convert to float on the GPU
                      (P3's design): about 10x faster than copying float32 from normal memory
                      (batch 32: 4.09 -> 0.43 ms). The pinned uint8 batch is small (19 MB at batch
                      128), so pinning is cheap, especially if the pinned buffer is reused instead of
                      allocated for every batch (PyTorch's pin_memory already caches pinned blocks).
                      Keep it in perspective: at batch 32 this saves about 3.7 ms per batch, about
                      0.25 s per epoch; compare with the epoch time from P1's profiling (Amdahl).
Still to do:          second part of A4.14, the share of the copy in the GPU preprocessing time,
                      needs P3's preprocess_batch_gpu. Re-run on P1's RTX 3050 for official numbers.

### A4.11 Second GPU median filter: fused custom Numba CUDA kernel (P4: Lina)   date: 2026-11-10
Motivation:           A4.13 showed the PyTorch median (pad + F.unfold + torch.median) is slower
                      than OpenCV at every batch size, because it runs 3 kernels and writes a 25x
                      intermediate tensor to GPU memory.
Implementation:       src/filters_numba.py (wrapper: filters_gpu.median_filter_numba). One thread
                      per output pixel, 2D blocks of 16x16 threads, one grid layer (blockIdx.z) per
                      image-channel slice. Each thread reads its 5x5 neighbours directly (coordinates
                      clamped to the image = replicate padding), sorts them in a private
                      cuda.local.array with an insertion sort and writes the middle value.
                      One kernel launch, no padding tensor, no 25x copy. Window fixed at 5x5.
Correctness:          tests/test_cpu_gpu_equivalence.py::test_median_filter_numba_equivalence
                      PASSED on Colab T4 (same rule as the PyTorch median test). Also checked in
                      Numba's CUDA simulator on 20x20 and 23x17 images (0% of pixels off by more
                      than 1 gray level; 23x17 exercises the out-of-image threads).
Prediction:           No prediction was written before the run. Stated after the run: I expected
                      it to be only slightly faster than the CPU, and only at larger batch sizes.
Command:              python -m pytest tests/test_cpu_gpu_equivalence.py -k numba -v
                      python benchmarks/sweep_median.py
Result:               DRAFT run on Colab, Tesla T4 (official run: P1, RTX 3050). ms per image:
                        batch    CPU (OpenCV)   PyTorch (A4.8)   Numba (A4.11)
                          1         0.76            4.80            1.33
                          8         0.80            3.79            0.51
                         32         0.84            2.42            0.56
                         64         0.84            2.43            0.56
                        128         1.00            2.44            0.55
Prediction vs result: Better than expected. The Numba kernel beats OpenCV from batch size 8 on
                      (1.5-1.8x faster) and is about 4.4x faster than the PyTorch version from
                      batch 32 on. At batch size 1 the CPU still wins (launch overhead).
Explanation:          Fusion avoids the 25x memory copy: each thread reads its neighbours straight
                      into its private array instead of F.unfold writing a 25x larger tensor to GPU
                      memory and torch.median reading it back, and there is one kernel launch
                      instead of three. Same GPU and same idea (sort 25 values per pixel), so the
                      gain comes from less memory traffic. The GPU is saturated from batch 8 on
                      (flat curve), earlier than the PyTorch version (batch 32), since there is less
                      work per image. It wins over OpenCV despite using a simple insertion sort.
Decision:             The fused kernel is the GPU median filter worth keeping. The median filter is
                      not used in training, so this does not speed up the pipeline itself; it shows
                      why kernel fusion matters. Possible next step: a faster sort (e.g. only
                      partially sorting up to the 13th value). Re-run on P1's RTX 3050.



### A2.3 Prediction: which stage takes the most time in a baseline epoch (P1: toutes)   date: 2026-11-12
Question:             Rank by share of the baseline epoch time: loading, preprocessing (CLAHE, crop,
                      normalize), forward, backward, optimizer step.
Prediction:           1) backward  2) forward  3) loading  4) preprocessing  5) optimizer step
                      Reason: the baseline runs on the CPU at only about 23 img/s (about 43 ms per
                      image), which is slow, so I think the neural network computation dominates.
                      The backward pass costs about twice the forward pass (it computes gradients
                      for every layer), so it should come first. Loading (decoding large retina
                      images every epoch) and preprocessing (CLAHE) come next, because nothing is
                      cached yet. The optimizer step only updates the weights once per batch, so
                      it should be the smallest.
Command:              python benchmarks/profile_baseline.py
                      python benchmarks/benchmark_stages.py --n 64
Result:               Profile of 1 CPU epoch (workers=0 on purpose, cProfile), total 167.7 s:
                        loading (decode+resize)   106.45 s   63.5 %
                        backward                   29.60 s   17.6 %
                        forward                    24.30 s   14.5 %
                        preprocessing (total)       5.64 s    3.4 %  (CLAHE 3.01, crop 1.12, normalize 1.37)
                        optimizer step              0.26 s    0.2 %
                        other                       1.48 s    0.9 %
                      Per-stage timing on 64 images (benchmark_stages.py, mean +/- std):
                        loading         2490.8 +/- 290.2 ms  (25.7 img/s)
                        preprocessing    214.7 +/-  17.6 ms
                        train_step      1631.3 +/- 167.6 ms  (39.2 img/s)
                        inference        413.4 +/-   9.7 ms
Prediction vs result: Predicted: backward > forward > loading > preprocessing > optimizer.
                      Measured: loading > backward > forward > preprocessing > optimizer.
                      Right: preprocessing 4th, optimizer last, backward above forward.
                      Wrong: loading. I put it 3rd, but it is 1st with 63.5 %, about twice
                      forward + backward together (32.1 %).
Explanation:          Loading decodes and resizes large retina images from disk, and the same
                      images are decoded again at every epoch. It costs about 39 ms per image
                      (benchmark) against about 25 ms per image for forward + backward
                      (train_step). Consistency check: forward + backward in the profile is
                      53.9 s for 2116 images = 25.5 ms per image, the same as the train_step
                      benchmark (1631 ms / 64 = 25.5 ms). Preprocessing is small (CLAHE only
                      3.0 s) and the optimizer step is tiny (it updates the weights once per
                      batch). The profile epoch is slower (15 img/s) than the baseline run
                      (23 img/s) because it uses no worker processes, so loading is not
                      overlapped with compute, and cProfile adds some overhead. The shares
                      above are for this serial version.
                      Amdahl bound: loading p = 0.635 gives 1 / (1 - 0.635) = 2.74x at best.
                      Backward p = 0.176 gives 1 / (1 - 0.176) = 1.21x at best.
Decision:             Loading is the first target: build the decoded-image cache (A3). Even
                      with free loading the end-to-end gain is at most 2.74x. Forward + backward
                      (32 %, at best 1.47x alone) come second, with the GPU. Preprocessing
                      (3.4 %, at best 1.04x) is not worth optimizing first. Re-profile after
                      each change, because the shares will move.