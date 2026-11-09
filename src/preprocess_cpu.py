"""Module B (CPU reference): Image preprocessing.

Normalization, circular cropping, resizing and contrast enhancement (CLAHE).

Profiling focus: per-image processing cost, intermediate arrays, memory allocation,
opportunities for batching.
"""


def circular_crop(image):
    raise NotImplementedError


def apply_clahe(image, clip_limit=2.0, tile_grid_size=8):
    raise NotImplementedError


def normalize(image):
    raise NotImplementedError


def preprocess(image, config):
    """Full CPU preprocessing for one image."""
    raise NotImplementedError
