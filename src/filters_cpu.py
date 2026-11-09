"""Module C (CPU reference): Image filters.

Gaussian blur, median filtering, green-channel extraction, contrast adjustment.

Profiling focus: computational complexity, neighborhood operations, parallel execution,
algorithmic alternatives.
"""


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    raise NotImplementedError


def median_filter(image, kernel_size=3):
    raise NotImplementedError


def green_channel(image):
    raise NotImplementedError


def apply_filters(image, config):
    raise NotImplementedError
