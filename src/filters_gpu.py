"""GPU (torch) equivalents of the reference filters in src/filters_cpu.py (assignment A4).

All functions take a (N,3,H,W) float tensor on the GPU and process the whole
batch at once. Tolerances are defined in tests/test_cpu_gpu_equivalence.py.
"""
import torch
import torch.nn.functional as F


def gaussian_blur_gpu(t, ksize=5, sigma=1.0):
    """Gaussian blur of a batch (N,3,H,W): one 2D kernel applied per channel."""
    k = ksize if ksize % 2 == 1 else ksize + 1
    x = torch.arange(k, device=t.device, dtype=t.dtype) - k // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    t_pad = F.pad(t, (k//2, k//2, k//2, k//2), mode='reflect')
    g_2d = torch.outer(g, g)
    return F.conv2d(t_pad, torch.stack((g_2d, g_2d, g_2d)).unsqueeze(1), groups=3)


def median_filter_gpu(t, ksize=5):
    """Median filter of a batch (N,C,H,W): replicate padding (as cv2.medianBlur), then
    F.unfold gathers every k*k neighbourhood. Memory cost: k*k times the input size."""
    N, C, H, W = t.shape
    k = ksize if ksize % 2 == 1 else ksize + 1
    t_pad = F.pad(t, (k//2, k//2, k//2, k//2), mode="replicate")
    columns = F.unfold(t_pad, kernel_size=k).reshape((N, C, k*k, H*W))
    return torch.median(columns, dim=2).values.reshape((N, C, H, W))


def median_filter_numba(t):
    """A4.11: second GPU version of the median filter (5x5), a fused custom Numba CUDA kernel.
    See src/filters_numba.py. Imported here only when called, so the other filters work
    on machines without Numba."""
    from src.filters_numba import median_filter_numba as _median_numba
    return _median_numba(t)


def green_channel_gpu(t):
    """The green channel as a (N,1,H,W) tensor (a view, no copy)."""
    return t[:, 1:2, :, :]


def contrast_stretch_gpu(t):
    """Per-image contrast stretch between the 1st and 99th percentiles of all its pixels
    (same definition as filters_cpu.contrast_stretch)."""
    N = t.shape[0]
    t_flat = t.reshape(N, -1)
    hi = torch.quantile(t_flat, 0.99, dim=1).reshape((N, 1, 1, 1))
    lo = torch.quantile(t_flat, 0.01, dim=1).reshape((N, 1, 1, 1))
    return torch.clamp((t - lo) / torch.clamp(hi - lo, min=1e-6), min=0, max=1)
