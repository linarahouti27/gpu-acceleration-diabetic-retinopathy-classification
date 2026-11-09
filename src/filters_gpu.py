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
    """TODO(A4.8): median filter of a batch.

    Hint: pad with mode="replicate" (what cv2.medianBlur does), extract all
    ksize*ksize neighbourhoods with F.unfold, take the median along that axis.
    F.unfold on (N,3,H,W) returns (N, 3*k*k, L): reshape it to (N,3,k*k,H,W) first.
    Note the memory cost: ksize^2 times the input size. Report it.
    """
    raise NotImplementedError("TODO(A4.8): median_filter_gpu")


# TODO(A4.11): pick ONE of the filters above and implement a second GPU version with a
# different tool (CuPy cupyx.scipy.ndimage, a Numba CUDA kernel, or separable convolutions).
# Put it in this file under a clear name and compare both versions against the CPU
# reference in speed and in maximum absolute difference. Which would you ship, and why?


def green_channel_gpu(t):
    """The green channel as a (N,1,H,W) tensor (a view, no copy)."""
    return t[:, 1:2, :, :]


def contrast_stretch_gpu(t):
    """TODO(A4.10): per-image contrast stretch using the 1st and 99th percentiles
    of all pixels of that image (same definition as filters_cpu.contrast_stretch).

    Hint: flatten each image and use torch.quantile along dim=1.
    """
    raise NotImplementedError("TODO(A4.10): contrast_stretch_gpu")
