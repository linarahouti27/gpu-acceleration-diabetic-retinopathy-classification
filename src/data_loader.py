"""Module A: Image loading.

Read images from disk, decode them, resize them to the target resolution and prepare
batches. Also responsible for patient-grouped train/val/test splitting.

Profiling focus: disk I/O, image decoding, repeated loading, data-loading overhead.
"""


def load_labels(config):
    """Load the labels file (image_id, patient_id, label)."""
    raise NotImplementedError


def make_splits(labels, config):
    """Split into train/val/test, keeping all images of a patient in the same split."""
    raise NotImplementedError


def load_image(path, image_size):
    """Read, decode and resize a single RGB image."""
    raise NotImplementedError


def build_dataloader(split, config):
    """Return a PyTorch DataLoader for the given split."""
    raise NotImplementedError
