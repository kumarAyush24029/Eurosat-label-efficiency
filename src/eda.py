"""
eda.py  -  Exploratory Data Analysis: look at the data BEFORE modelling.

Produces three figures for the report:
  1. class_counts.png      - how many images per class in train / val / test
                             (EuroSAT is nearly balanced, so accuracy is a fair metric)
  2. sample_grid.png       - random example images of each class (get a feel for the task)
  3. spectral_signatures.png (only if --ms_root is given)
                           - average reflectance of each class in each of the 13 Sentinel-2 bands.
                             Classes whose curves differ in the infrared bands are exactly the ones
                             where extra bands can help beyond what RGB shows.

Usage:
  python src/eda.py --rgb_root data/raw_rgb/x/EuroSAT_RGB --splits data/splits.json --out figures
  python src/eda.py --ms_root  data/raw_ms/EuroSAT_MS     --splits data/splits.json --out figures   (13-band plot)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

import data as D


def class_counts(sp, out):
    """Bar chart: images per class, per split (read straight from splits.json)."""
    classes = sp["classes"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(classes))
    for split, color in zip(("train", "val", "test"), ("#4c72b0", "#dd8452", "#55a868")):
        cnt = Counter(s.split("/")[0] for s in sp[split])
        vals = np.array([cnt[c] for c in classes])
        ax.bar(classes, vals, bottom=bottom, label=f"{split} ({vals.sum()})", color=color)
        bottom += vals
    ax.set_ylabel("number of images"); ax.set_title("EuroSAT: images per class and split")
    plt.xticks(rotation=40, ha="right"); ax.legend(); plt.tight_layout()
    plt.savefig(out / "class_counts.png", dpi=200); plt.close()


def sample_grid(arrays, classes, out, per_class=8, seed=0):
    """Grid of random training images: one row per class."""
    X, y = arrays["train"]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.2, len(classes) * 1.2))
    for r, c in enumerate(classes):
        idx = rng.choice(np.flatnonzero(y.numpy() == r), per_class, replace=False)
        for k, i in enumerate(idx):
            axes[r, k].imshow(X[i].permute(1, 2, 0).numpy())        # (C,H,W) -> (H,W,C)
            axes[r, k].axis("off")
        axes[r, 0].text(-0.1, 0.5, c, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(out / "sample_grid.png", dpi=200, bbox_inches="tight"); plt.close()


def spectral_signatures(arrays, classes, out):
    """Mean value of every band for every class (13-band data only)."""
    X, y = arrays["train"]
    X = X.float()
    plt.figure(figsize=(9, 5))
    for r, c in enumerate(classes):
        m = X[y == r].mean((0, 2, 3)).numpy()                       # mean per band for class r
        plt.plot(D.BANDS, m, marker="o", ms=3, label=c)
    plt.xlabel("Sentinel-2 band"); plt.ylabel("mean pixel value (digital number)")
    plt.title("Spectral signature of each land-use class")
    plt.legend(fontsize=7, ncol=2); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(out / "spectral_signatures.png", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_root"); ap.add_argument("--ms_root")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sp = json.loads(Path(a.splits).read_text())
    class_counts(sp, out)
    if a.rgb_root:
        arrays, classes = D.load_split_arrays(a.rgb_root, a.splits, a.cache, "rgb")
        sample_grid(arrays, classes, out)
    if a.ms_root:
        arrays, classes = D.load_split_arrays(a.ms_root, a.splits, a.cache, "ms")
        spectral_signatures(arrays, classes, out)
    print("saved figures to", out)


if __name__ == "__main__":
    main()
