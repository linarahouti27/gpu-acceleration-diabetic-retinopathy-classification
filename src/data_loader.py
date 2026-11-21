"""Dataset discovery, patient-safe splitting, and DataLoader construction.

The CPU reference below works as is. Assignments A3 and A5 extend it.
"""
import os, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_images(root):
    """Walk root/<class_id>/*.png and return [(path, label), ...]."""
    items = []
    for dirpath, _, fnames in os.walk(root):
        try:
            label = int(os.path.basename(dirpath))
        except ValueError:
            continue
        if not 0 <= label <= 4:
            continue
        for f in sorted(fnames):
            if f.lower().endswith(IMG_EXTS):
                items.append((os.path.join(dirpath, f), label))
    return items


def patient_id(path):
    """Patient id = filename prefix before the LAST underscore (<patient>_<image>.png)."""
    return os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[0]


def split_by_patient(items, train_frac, val_frac, seed):
    by_pat = {}
    for p, l in items:
        by_pat.setdefault(patient_id(p), []).append((p, l))
    pats = sorted(by_pat)
    random.Random(seed).shuffle(pats)
    n = len(pats)
    n_train, n_val = int(n * train_frac), int(n * val_frac)
    train_p = set(pats[:n_train])
    val_p = set(pats[n_train:n_train + n_val])
    splits = {"train": [], "val": [], "test": []}
    for p in pats:
        key = "train" if p in train_p else "val" if p in val_p else "test"
        splits[key].extend(by_pat[p])
    return splits


class DRDataset(Dataset):
    """Loads, decodes and resizes an image; returns (C,H,W) tensor + label.

    CPU preprocessing is injected through `transform`.
    With `raw_uint8=True` the image is returned untouched (uint8, CHW) so a
    GPU can preprocess the whole batch after a 4x smaller host-to-device copy.
    """

    def __init__(self, items, image_size, transform=None, raw_uint8=False, cache=None):
        self.items = items
        self.image_size = tuple(image_size)
        self.transform = transform
        self.raw_uint8 = raw_uint8
        self.cache = cache

    def __len__(self):
        return len(self.items)

    def load(self, idx):
        """Decode + resize only (the 'loading' stage). Returns HWC uint8."""
        if self.cache is not None:
            return np.asarray(self.cache[idx])
        path, _ = self.items[idx]
        with Image.open(path) as im:
            im = im.convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
            return np.asarray(im, dtype=np.uint8)

    def __getitem__(self, idx):
        u8 = self.load(idx)
        label = int(self.items[idx][1])
        if self.raw_uint8:
            return torch.from_numpy(np.ascontiguousarray(u8.transpose(2, 0, 1))), label
        x = u8.astype(np.float32) / 255.0
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = x.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), label


class CpuTransform:
    """CPU preprocessing for one image: HWC float [0,1] -> CHW float.

    A class (not a lambda) so it pickles into spawned DataLoader workers.
    """

    def __init__(self, cfg, augment=False):
        self.cfg = cfg
        self.augment = augment
        self._rng = None

    def __call__(self, img):
        from src import preprocess_cpu
        if self.augment:
            if self._rng is None:      # created inside the worker, seeded per worker
                self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            img = preprocess_cpu.augment_image(img, self._rng)
        return preprocess_cpu.preprocess_image(img, self.cfg)


def get_preprocess_transform(cfg, augment=False):
    return CpuTransform(cfg, augment)


def build_cache(items, image_size, cache_dir, workers=8):
    import hashlib
    import os
    from concurrent.futures import ThreadPoolExecutor
    
    os.makedirs(cache_dir, exist_ok=True)
    
    hasher = hashlib.md5()
    for path, label in items:
        hasher.update(path.encode("utf-8"))
        hasher.update(str(label).encode("utf-8"))
    hasher.update(str(image_size).encode("utf-8"))
    cache_sig = hasher.hexdigest()[:16]
    
    W, H = image_size
    shape = (len(items), H, W, 3)
    filename = f"cache_{cache_sig}_{len(items)}x{H}x{W}x3.uint8.npy"
    final_path = os.path.join(cache_dir, filename)
    
    if os.path.exists(final_path):
        return np.memmap(final_path, dtype=np.uint8, mode="r", shape=shape)
        
    tmp_path = final_path + ".tmp"
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
            
    tmp_memmap = np.memmap(tmp_path, dtype=np.uint8, mode="w+", shape=shape)
    ds = DRDataset(items, image_size)
    
    def load_and_write(idx):
        tmp_memmap[idx] = ds.load(idx)
        
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(load_and_write, range(len(items))))
        
    tmp_memmap.flush()
    del tmp_memmap
    
    os.replace(tmp_path, final_path)
    return np.memmap(final_path, dtype=np.uint8, mode="r", shape=shape)


def _worker_init(_):
    # one OpenCV thread per worker: N workers x many cv2 threads oversubscribes the CPU
    import cv2
    cv2.setNumThreads(0)


def make_loaders(cfg, device="cpu", gpu_preprocess=False, which=("train", "val", "test")):
    items = find_images(cfg["data"]["root"] )
    if not items:
        raise FileNotFoundError(

            f"No images found under {cfg['data']['root']}/<class>/...  "

            "Run scripts/prepare_fdr26.py or point data.root at the dataset.")
    splits = split_by_patient(

        items, cfg["data"]["train_frac"], cfg["data"]["val_frac"],

        cfg["data"].get("split_seed", cfg["seed"]))
    nw = cfg["data"]["num_workers"]
    augment = bool(cfg["preprocess"].get("augment", False))
    loaders = {}
    for k in which:
        tf = None if gpu_preprocess else get_preprocess_transform(cfg, augment and k == "train")
        cache = None
        if cfg["data"].get("cache_dir"):
            cache = build_cache(splits[k], cfg["data"]["image_size"], cfg["data"]["cache_dir"])
        ds = DRDataset(splits[k], cfg["data"]["image_size"], transform=tf,

                       raw_uint8=gpu_preprocess, cache=cache)
        kw = dict(

            batch_size=cfg["data"]["batch_size"],

            shuffle=(k == "train"),

            num_workers=nw,

            drop_last=(k == "train"),

        )
        if device == "cuda":
            kw["pin_memory"] = True
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = cfg["data"].get("prefetch_factor", 2)
            kw["worker_init_fn"] = _worker_init
        loaders[k] = DataLoader(ds, **kw)
    return loaders, splits
