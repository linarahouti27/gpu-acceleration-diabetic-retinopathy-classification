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

### A2.3: Prediction of Time Distribution (CPU Baseline)

- **Prediction:** Preprocessing (CLAHE, cropping, and normalization) is expected to account for the largest share of the execution time per epoch, due to CPU color space conversions and the initial absence of caching.
- **Justification:** Sequential OpenCV operations block the pipeline before the forward pass even begins.

### A2.7: Fine-Grained Profiling with line_profiler

- **Result:** Profiling the `apply_clahe` function revealed that color space conversions account for a significant proportion of its execution time.
- **Comparison:** A standard profiling tool such as `cProfile` would only indicate that the `apply_clahe` function takes approximately 0.1 seconds overall, without providing details about the execution time of individual operations.
- **Conclusion:** This profiling demonstrates that accelerating the pipeline on GPU requires more than simply rewriting the CLAHE algorithm. The entire processing pipeline must be reconsidered to avoid costly CPU color space conversions and to take advantage of a fully GPU-based tensor pipeline.

- **Conclusion of Profiling (Average over 7 iterations)**

**Moyenne des 7 exécutions du profilage**

**Temps total moyen :** ~0.002238 s (2 238 337 ns)

| Line # | Hits | Time (ns) | Per Hit (ns) | % Time | Line Contents |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 23 | | | | | `def apply_clahe(img, clip=2.0, grid=8):` |
| 24 | | | | | `"""CLAHE on the L channel of LAB. Input RGB float [0,1], same range out."""` |
| 25 | 1 | 207,140.6 | 207,140.6 | 9.3% | `u8 = (img * 255).astype(np.uint8)` |
| 26 | 1 | 480,974.7 | 480,974.7 | 21.5% | `lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)` |
| 27 | 1 | 100,700.0 | 100,700.0 | 4.5% | `l, a, b = cv2.split(lab)` |
| 28 | 1 | 568,260.4 | 568,260.4 | 25.4% | `cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)` |
| 29 | 1 | 746,284.0 | 746,284.0 | 33.3% | `out = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)` |
| 30 | 1 | 149,977.3 | 149,977.3 | 6.7% | `return out.astype(np.float32) / 255.0`

The average execution time is approximately 2.2 ms per image. The analysis of the execution time distribution shows that the histogram equalization algorithm itself is not the main bottleneck:

- **Color space conversion overhead:** The RGB-to-LAB conversion (line 26) accounts for 21.5% of the total execution time, while the reverse LAB-to-RGB conversion (line 29) is the most expensive operation, accounting for 33.3%.
- **Algorithm execution cost:** The actual CLAHE operation (line 28) accounts for only 25.4% of the total execution time.

**Conclusion:**

The main bottleneck of this CPU function comes from the color space conversions, which together account for approximately 54.8% of the total execution time. To optimize this pipeline on GPU (A4.3), it will be essential to perform CLAHE directly on the reconstructed luminance channel or avoid these expensive color space conversions.

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

dd9e6c75b5b8280f1223d1815229da0d7efba42b
