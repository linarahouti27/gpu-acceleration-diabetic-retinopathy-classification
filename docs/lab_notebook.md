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

Still to do:      run `python -m pytest tests/test_cpu_gpu_equivalence.py -v` on a CUDA machine
                  (Colab or P1's RTX 3050) and paste the result here.
