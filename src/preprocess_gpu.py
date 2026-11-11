"""GPU preprocessing pipeline for batched image processing."""

import torch
import torch.nn.functional as F

_MASK_CACHE = {}
_NORM_CACHE = {}

# ============================================================
# A4.1 - CIRCULAR CROP
# ============================================================
def circular_crop_gpu(t):
    """
    Apply a circular mask to a batch of images.
    Input:  (N, C, H, W)
    Output: (N, C, H, W)
    """
    if t.ndim != 4:
        raise ValueError("Expected a tensor with shape (N,C,H,W).")
    _, _, H, W = t.shape
    key = (H, W, t.device.type, t.device.index, t.dtype)
    if key not in _MASK_CACHE:
        y = torch.arange(H, device=t.device)
        x = torch.arange(W, device=t.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        center_y = H // 2
        center_x = W // 2
        radius = max(min(H, W) // 2 - 2, 1)
        mask = (
            (yy - center_y).square()
            + (xx - center_x).square()
            <= radius ** 2
        )
        _MASK_CACHE[key] = mask.to(dtype=t.dtype)
    return t * _MASK_CACHE[key].unsqueeze(0).unsqueeze(0)

# ============================================================
# A4.2 - NORMALIZATION
# ============================================================
def normalize_gpu(t):
    """
    Apply ImageNet normalization to a batch of images.
    Input: (N, 3, H, W) or (N, 1, H, W)
    """
    channels = t.shape[1]
    if channels not in (1, 3):
        raise ValueError("Expected 1 or 3 image channels.")
    key = (t.device.type, t.device.index, t.dtype, channels)
    if key not in _NORM_CACHE:
        mean_values = (0.485, 0.456, 0.406)
        std_values = (0.229, 0.224, 0.225)
        if channels == 1:
            mean_values = mean_values[:1]
            std_values = std_values[:1]
        mean = torch.tensor(
            mean_values,
            device=t.device,
            dtype=t.dtype
        ).view(1, channels, 1, 1)
        std = torch.tensor(
            std_values,
            device=t.device,
            dtype=t.dtype
        ).view(1, channels, 1, 1)
        _NORM_CACHE[key] = (mean, std)
    mean, std = _NORM_CACHE[key]
    return (t - mean) / std

# ============================================================
# A4.3 - CLAHE
# ============================================================
def clahe_gpu(t, clip=2.0, grid=8):
    """
    Batched CLAHE on image luminance using pure PyTorch, aligned with OpenCV Lab L channel.
    Input:  (N, 3, H, W), float in [0,1]
    Output: (N, 3, H, W), float in [0,1]
    """
    if t.ndim != 4 or t.shape[1] != 3:
        raise ValueError("CLAHE expects RGB input (N,3,H,W).")
    if grid < 1 or clip <= 0:
        raise ValueError("grid and clip must be positive.")
    N, C, H, W = t.shape
    tile_h = H // grid
    tile_w = W // grid
    if tile_h == 0 or tile_w == 0:
        raise ValueError("Grid is too large for image dimensions.")
    Hc = tile_h * grid
    Wc = tile_w * grid

    r, g, b = t[:, 0], t[:, 1], t[:, 2]
    r_lin = torch.where(r <= 0.04045, r / 12.92, ((r + 0.055) / 1.055) ** 2.4)
    g_lin = torch.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4)
    b_lin = torch.where(b <= 0.04045, b / 12.92, ((b + 0.055) / 1.055) ** 2.4)
    
    y_linear = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    delta = 6.0 / 29.0
    f_y = torch.where(y_linear > delta**3, y_linear ** (1.0 / 3.0), (y_linear / (3.0 * delta**2)) + (4.0 / 29.0))
    luminance = (116.0 * f_y - 16.0) / 100.0

    lum_crop = luminance[:, :Hc, :Wc]
    pixels = (lum_crop.clamp(0, 1) * 255.0).long()

    tiles = (
        pixels.contiguous()
        .view(N, grid, tile_h, grid, tile_w)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(N * grid * grid, tile_h * tile_w)
    )
    tile_pixels = tile_h * tile_w
    bins = 256

    hist = torch.zeros(
        N * grid * grid,
        bins,
        device=t.device,
        dtype=torch.float32
    )
    hist.scatter_add_(
        1,
        tiles,
        torch.ones_like(tiles, dtype=torch.float32)
    )

    clip_limit = clip * tile_pixels / bins
    excess = (hist - clip_limit).clamp(min=0).sum(dim=1, keepdim=True)
    hist = hist.clamp(max=clip_limit)
    hist = hist + excess / bins

    cdf = hist.cumsum(dim=1)
    lut = cdf / cdf[:, -1:].clamp(min=1e-12)
    lut = lut.view(N, grid * grid, bins)

    gy = (torch.arange(Hc, device=t.device, dtype=torch.float32) + 0.5) / tile_h - 0.5
    gx = (torch.arange(Wc, device=t.device, dtype=torch.float32) + 0.5) / tile_w - 0.5

    gy_floor = torch.floor(gy)
    gx_floor = torch.floor(gx)

    wy = (gy - gy_floor).view(1, Hc, 1)
    wx = (gx - gx_floor).view(1, 1, Wc)

    y0 = gy_floor.long().clamp(0, grid - 1)
    y1 = (gy_floor.long() + 1).clamp(0, grid - 1)

    x0 = gx_floor.long().clamp(0, grid - 1)
    x1 = (gx_floor.long() + 1).clamp(0, grid - 1)

    idx00 = (y0[:, None] * grid + x0[None, :]).reshape(1, -1)
    idx01 = (y0[:, None] * grid + x1[None, :]).reshape(1, -1)
    idx10 = (y1[:, None] * grid + x0[None, :]).reshape(1, -1)
    idx11 = (y1[:, None] * grid + x1[None, :]).reshape(1, -1)

    idx00 = idx00.expand(N, -1)
    idx01 = idx01.expand(N, -1)
    idx10 = idx10.expand(N, -1)
    idx11 = idx11.expand(N, -1)

    pixel_values = pixels.reshape(N, -1)

    batch_idx = torch.arange(N, device=t.device).view(N, 1).expand_as(pixel_values)

    v00 = lut[batch_idx, idx00, pixel_values]
    v01 = lut[batch_idx, idx01, pixel_values]
    v10 = lut[batch_idx, idx10, pixel_values]
    v11 = lut[batch_idx, idx11, pixel_values]

    v00 = v00.view(N, Hc, Wc)
    v01 = v01.view(N, Hc, Wc)
    v10 = v10.view(N, Hc, Wc)
    v11 = v11.view(N, Hc, Wc)

    top = v00 * (1 - wx) + v01 * wx
    bottom = v10 * (1 - wx) + v11 * wx

    equalized = top * (1 - wy) + bottom * wy
    equalized = equalized * 255.0

    scale = (equalized + 1.0) / (lum_crop * 255.0 + 1.0)
    scale_full = torch.ones_like(luminance)
    scale_full[:, :Hc, :Wc] = scale

    output = t.float() * scale_full.unsqueeze(1)
    return output.clamp(0, 1).to(dtype=t.dtype)

# ============================================================
# A4.4 - DATA AUGMENTATION
# ============================================================
def augment_gpu(t):
    """
    Apply independent horizontal flips and rotations per image.
    Rotation range: [-10, +10] degrees.
    """
    N, C, H, W = t.shape

    flip_mask = torch.rand(N, device=t.device) < 0.5
    flipped = torch.flip(t, dims=[3])
    t = torch.where(flip_mask.view(N, 1, 1, 1), flipped, t)

    angles = torch.empty(N, device=t.device, dtype=t.dtype).uniform_(-10.0, 10.0)
    angles = angles * (torch.pi / 180.0)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    theta = torch.zeros(N, 2, 3, device=t.device, dtype=t.dtype)

    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    grid = F.affine_grid(theta, t.size(), align_corners=False)
    output = F.grid_sample(t, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return output

# ============================================================
# A4.5 - UINT8 TO FLOAT CONVERSION
# ============================================================
def to_float_bchw(x, device=None):
    """
    Convert uint8 or float input to float32 NCHW in [0,1].
    """
    x = torch.as_tensor(x)
    if device is not None:
        x = x.to(device=device, non_blocking=True)
    elif x.device.type == 'cpu' and torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    if x.ndim != 4:
        raise ValueError("Expected a batch with 4 dimensions.")

    if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
        x = x.permute(0, 3, 1, 2)

    if x.shape[1] not in (1, 3):
        raise ValueError("Expected 1 or 3 channels.")

    if x.dtype == torch.uint8:
        x = x.to(torch.float32).div(255.0)
    else:
        x = x.to(torch.float32)

    return x.contiguous()

# ============================================================
# A4.6 - FULL GPU BATCH PIPELINE
# ============================================================
def preprocess_batch_gpu(images, cfg, augment=False, device="cuda"):
    """
    Full batched GPU preprocessing pipeline.
    """
    x = to_float_bchw(images, device)
    preprocess_cfg = cfg.get("preprocess", {})

    if augment:
        x = augment_gpu(x)

    gamma = float(preprocess_cfg.get("gamma", 1.0))
    if gamma <= 0:
        raise ValueError("Gamma must be positive.")
    if gamma != 1.0:
        x = x.clamp(0, 1).pow(gamma)

    if preprocess_cfg.get("clahe", False):
        x = clahe_gpu(
            x,
            clip=preprocess_cfg.get("clahe_clip", 2.0),
            grid=preprocess_cfg.get("clahe_grid", 8)
        )

    if preprocess_cfg.get("circular_crop", True):
        x = circular_crop_gpu(x)

    if preprocess_cfg.get("normalize", True):
        x = normalize_gpu(x)

    return x
