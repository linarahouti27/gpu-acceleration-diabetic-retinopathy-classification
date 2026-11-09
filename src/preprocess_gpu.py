"""Module B (GPU): Batched image preprocessing.

GPU counterpart of preprocess_cpu.py (PyTorch / CuPy / Numba). Operates on batches
already on the device. Must be validated against the CPU reference
(see tests/test_cpu_gpu_equivalence.py).
"""


def circular_crop_batch(images):
    raise NotImplementedError


def apply_clahe_batch(images, clip_limit=2.0, tile_grid_size=8):
    raise NotImplementedError


def normalize_batch(images):
    raise NotImplementedError


def preprocess_batch(images, config):
    """Full GPU preprocessing for a batch of images."""
    raise NotImplementedError
