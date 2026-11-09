"""Module C (GPU): Image filters.

GPU counterparts of filters_cpu.py using library kernels (e.g. cupyx.scipy.ndimage,
torch conv2d) or custom kernels for selected bottlenecks.
"""


def gaussian_blur_batch(images, kernel_size=5, sigma=1.0):
    raise NotImplementedError


def median_filter_batch(images, kernel_size=3):
    raise NotImplementedError


def green_channel_batch(images):
    raise NotImplementedError


def apply_filters_batch(images, config):
    raise NotImplementedError
