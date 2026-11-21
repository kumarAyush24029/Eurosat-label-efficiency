"""
splits.py  -  decide WHICH images are used for training, validation and testing.

Why this file exists
--------------------
A fair experiment needs three things:
  1. A fixed TEST set that no model ever trains on or is tuned on (used once, at the end).
  2. A VALIDATION set used to pick the best checkpoint while training.
  3. A TRAIN set, from which we can take small "label budgets" (1%, 10%, ...)
     to simulate having very few labelled images.

We create the split ONCE and save it to splits.json, so every experiment
(every model, every seed) is compared on exactly the same images.

How images are identified
-------------------------
Each image gets a "stem" such as "Forest/Forest_123"  (class folder + file name
without extension). The RGB version (.jpg) and the 13-band version (.tif) of EuroSAT
use identical stems, so ONE splits.json works for both. That makes the
"RGB vs 13-band" comparison fair: same images, only the input channels differ.

Usage
-----
    python src/splits.py --root data/raw_rgb/x/EuroSAT_RGB --out data/splits.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

# File types we treat as images.
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def scan(root):
    """Walk the dataset folder (one sub-folder per class) and list every image.

    Returns:
        classes : sorted list of class names, e.g. ["AnnualCrop", "Forest", ...]
        stems   : list of "Class/filename_without_extension" strings
        labels  : numpy array, labels[i] = class index of stems[i]
    """
    root = Path(root)
    # Class names = folder names. Sorting makes the class->number mapping deterministic.
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    stems, labels = [], []
    for class_index, class_name in enumerate(classes):
        for path in sorted((root / class_name).iterdir()):   # sorted => same order every run
            if path.suffix.lower() in EXTS:
                stems.append(f"{class_name}/{path.stem}")
                labels.append(class_index)
    return classes, stems, np.array(labels)


def make_splits(labels, val=0.1, test=0.1, seed=0):
    """Split image indices into train / val / test, keeping class proportions equal.

    "Stratified": we split each class separately, so every split contains
    the same mix of classes (e.g. 10% of Forest goes to test, 10% of River, ...).

    Returns a dict {"train": [...], "val": [...], "test": [...]} of indices into the scanned list.
    """
    rng = np.random.default_rng(seed)        # seeded => the split is reproducible
    out = {"train": [], "val": [], "test": []}
    for c in np.unique(labels):
        # All images of class c, in random order.
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_val, n_test = round(len(idx) * val), round(len(idx) * test)
        out["test"] += idx[:n_test].tolist()                   # first chunk  -> test
        out["val"] += idx[n_test:n_test + n_val].tolist()      # second chunk -> val
        out["train"] += idx[n_test + n_val:].tolist()          # the rest     -> train
    return {k: sorted(v) for k, v in out.items()}


def label_subset(train_idx, labels, frac, seed):
    """Pick a small, class-balanced "labelled" subset of the training set.

    This simulates a low-label situation: with frac=0.01 we pretend only 1% of the
    training images have labels (about 21 per class).

    Key property - the subsets are NESTED for a given seed:
        subset(1%)  is contained in  subset(10%)  is contained in  subset(100%)
    because for every class we shuffle once (using the seed) and just take a longer
    prefix for a bigger fraction. So the learning curve only changes the AMOUNT of
    data, not which images happened to be drawn - this makes the curve much smoother.

    Different seeds give different random subsets, which lets us report mean +/- std.
    """
    rng = np.random.default_rng(10_000 + seed)
    train_idx = np.asarray(train_idx)
    picked = []
    for c in np.unique(labels[train_idx]):
        # Training images of class c, shuffled. Same shuffle for any frac (same seed).
        cls = rng.permutation(train_idx[labels[train_idx] == c])
        # Take the first ceil(frac * n) images (at least 1 per class).
        picked += cls[: max(1, math.ceil(frac * len(cls)))].tolist()
    return np.array(sorted(picked))


def main():
    """Command-line entry: scan the dataset, make the split, save splits.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="EuroSAT folder with one sub-folder per class")
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--val", type=float, default=0.1, help="fraction for validation")
    ap.add_argument("--test", type=float, default=0.1, help="fraction for test")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    classes, stems, labels = scan(a.root)
    sp = make_splits(labels, a.val, a.test, a.seed)
    # Save names (not indices) so the file is valid for any copy of the dataset.
    payload = {"classes": classes, **{k: [stems[i] for i in v] for k, v in sp.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload))
    print(f"{len(stems)} images, {len(classes)} classes -> "
          + ", ".join(f"{k}={len(v)}" for k, v in sp.items()))


if __name__ == "__main__":
    main()
