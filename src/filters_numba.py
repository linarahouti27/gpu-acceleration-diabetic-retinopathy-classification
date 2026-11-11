"""A4.11: fused median filter as a custom Numba CUDA kernel (second GPU version of the median).

One thread computes one output pixel: it reads its 5x5 neighbourhood directly from the image
(coordinates clamped to the border = replicate padding), sorts the 25 values in a private
array with an insertion sort, and writes the middle one. Compared with median_filter_gpu
(pad + F.unfold + torch.median), there is one kernel launch and no 25x intermediate tensor.

Limitation: the window is fixed at 5x5 (cuda.local.array needs a compile-time size).
Kept in its own file so that only this module needs Numba.
"""
import math
import numba
import torch
from numba import cuda


@cuda.jit
def median5_kernel(src, dst):
    """One thread = one output pixel. src, dst: (N*C, H, W) float32 on the GPU."""
    x, y = cuda.grid(2)              # this thread's column x and row y
    img = cuda.blockIdx.z            # which image-channel slice
    H, W = src.shape[1], src.shape[2]
    if x >= W or y >= H:             # threads that fall outside the image do nothing
        return
    window = cuda.local.array(25, dtype=numba.float32)   # this thread's private array
    i = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            yy = min(max(0, y + dy), H - 1)   # clamp to the image = replicate padding
            xx = min(max(0, x + dx), W - 1)
            window[i] = src[img, yy, xx]
            i += 1
    # insertion sort of the 25 values
    for i in range(len(window) - 1):
        key = window[i + 1]
        j = i
        while j > -1 and window[j] > key:
            window[j + 1] = window[j]
            j -= 1
        window[j + 1] = key
    dst[img, y, x] = window[12]      # the middle of 25 sorted values


def median_filter_numba(t):
    """Same interface as median_filter_gpu: (N,C,H,W) float32 GPU tensor, 5x5 window."""
    N, C, H, W = t.shape
    src = t.contiguous().view(N * C, H, W)      # every image-channel is one 2D slice
    dst = torch.empty_like(src)
    threads = (16, 16)                          # 256 threads per block
    blocks = (math.ceil(W / 16), math.ceil(H / 16), N * C)
    median5_kernel[blocks, threads](cuda.as_cuda_array(src), cuda.as_cuda_array(dst))
    return dst.view(N, C, H, W)
