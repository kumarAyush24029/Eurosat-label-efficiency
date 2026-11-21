"""
data.py  -  load EuroSAT images into memory and prepare them for the network.

EuroSAT is small (27,000 images of 64x64 pixels), so instead of reading files from
disk every epoch we load EVERYTHING into memory once as a big tensor.
This makes training fast, especially on a free Kaggle/Colab GPU.

Two versions of the dataset exist:
  * RGB   : normal colour images (.jpg), 3 channels.
  * MS    : multispectral (.tif), 13 Sentinel-2 bands (visible + near-infrared +
            red-edge + short-wave infrared ...). Extra bands can reveal things the eye
            can't see (e.g. healthy vegetation reflects strongly in near-infrared).
"""
import json
from pathlib import Path

import numpy as np
import torch

# The 13 Sentinel-2 bands in the order they are stored in the EuroSAT_MS .tif files.
#
# IMPORTANT: the narrow NIR band B8A is stored LAST, not between B08 and B09.
# We found this from the data itself (see the spectral-signature figure): the channel at
# position 9 is ~0 for every class (the cirrus band B10 has almost no signal), and the channel
# at position 12 follows the near-infrared band B08 almost exactly (very bright for crops,
# very dark for water), which is what B8A looks like. Position 10/11 also fit B11 > B12.
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]

# Which bands to feed the network in each experiment ("band ablation").
# Numbers are positions in BANDS above. NOTE: for RGB the order must be R,G,B
# = B04, B03, B02, which is why it reads [3, 2, 1].
BAND_GROUPS = {
    "rgb": [3, 2, 1],                                  # red, green, blue only
    "rgb_ms": [3, 2, 1],                               # same bands, but taken from the 13-band files
                                                       # (a like-for-like reference for the ablation)
    "rgb+nir": [3, 2, 1, 7],                           # + near-infrared (B08)
    "rgb+nir+rededge": [3, 2, 1, 7, 4, 5, 6, 12],      # + red-edge bands (B05,B06,B07,B8A)
    "rgb+nir+swir": [3, 2, 1, 7, 10, 11],              # + short-wave infrared (B11,B12)
    "all13": list(range(13)),                          # everything
    # --- groups of the FIRST Kaggle run, which used the wrong band order (positions 8, 11, 12
    # were assumed to be B8A, B11, B12). What they really contained is kept under honest names:
    "rgb+nir+b05b06b07b09": [3, 2, 1, 7, 4, 5, 6, 8],  # + B05,B06,B07 (red-edge) and B09 (water vapour)
    "rgb+nir+b12+b8a": [3, 2, 1, 7, 11, 12],           # + B12 (SWIR-2) and B8A (narrow NIR)
}


def _read(path):
    """Read ONE image file and return an array shaped (channels, height, width)."""
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile                                 # only needed for the 13-band version
        a = tifffile.imread(path)                       # shape (H, W, 13), 16-bit integers
        return np.ascontiguousarray(a.transpose(2, 0, 1))   # -> (13, H, W)
    from PIL import Image
    # .convert("RGB") guarantees 3 channels; transpose (H,W,3) -> (3,H,W) as PyTorch expects.
    return np.asarray(Image.open(path).convert("RGB")).transpose(2, 0, 1).copy()


def load_split_arrays(root, splits_json, cache_dir=None, tag="rgb"):
    """Load train/val/test images according to splits.json.

    Returns:
        out     : {"train": (X, y), "val": (X, y), "test": (X, y)}
                  X = uint tensor [N, channels, 64, 64] (raw pixel values), y = class numbers
        classes : list of class names
    The first call is slow (reads ~27k files); the result is cached as .npz so
    later calls take a second. `tag` ("rgb"/"ms") keeps the two caches apart.
    """
    root = Path(root)
    sp = json.loads(Path(splits_json).read_text())
    classes = sp["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    out = {}
    for split in ("train", "val", "test"):
        cache = Path(cache_dir) / f"{tag}_{split}.npz" if cache_dir else None
        if cache is not None and cache.exists():
            # Fast path: already converted before.
            z = np.load(cache)
            X, y = z["X"], z["y"]
        else:
            stems = sp[split]
            # Find out whether this folder holds .jpg or .tif files.
            ext = None
            for e in (".jpg", ".tif", ".png", ".tiff", ".jpeg"):
                if (root / f"{stems[0]}{e}").exists():
                    ext = e
                    break
            assert ext, f"cannot find files for {stems[0]} under {root}"
            X = np.stack([_read(root / f"{s}{ext}") for s in stems])
            y = np.array([class_to_idx[s.split("/")[0]] for s in stems])   # class from folder name
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, X=X, y=y)
        out[split] = (torch.from_numpy(X), torch.from_numpy(y).long())
    return out, classes


def channel_stats(X):
    """Mean and standard deviation of every channel (computed on TRAIN images only).

    Used to standardise the input: (pixel - mean) / std. Networks train much better
    when inputs are roughly zero-mean / unit-variance. Computing it from train only
    avoids leaking information from val/test.
    """
    x = X.float()
    return x.mean((0, 2, 3)), x.std((0, 2, 3)).clamp_min(1e-6)   # clamp avoids dividing by 0


def normalize(xb, mean, std):
    """Standardise a batch of images: (x - mean) / std, per channel."""
    return (xb.float() - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)


def dihedral_aug(x):
    """Data augmentation: randomly rotate (0/90/180/270 deg) and flip each image.

    Why: a satellite photo has no "correct" up direction - a forest seen rotated is
    still a forest. These transforms create new, valid training examples for free,
    which matters a lot when we only have ~200 labelled images. Done on the GPU,
    per image, so it costs almost nothing.
    """
    B = x.shape[0]
    k = torch.randint(0, 4, (B,), device=x.device)       # number of 90-degree turns per image
    f = torch.rand(B, device=x.device) < 0.5              # whether to flip each image
    out = x.clone()
    for r in range(1, 4):                                  # apply each rotation to its group of images
        m = k == r
        if m.any():
            out[m] = torch.rot90(x[m], r, (2, 3))          # rotate over the height/width axes
    if f.any():
        out[f] = out[f].flip(3)                            # horizontal flip
    return out
