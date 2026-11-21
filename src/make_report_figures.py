"""
make_report_figures.py  -  produce the final, publication-style figures used in the LaTeX report.

Reads : results/runs_*.csv (all runs), results/logits/*.npz (test predictions), figures/*.png (EDA)
Writes: report/latex/fig_*.png  (flat folder: easy to upload to Overleaf)

Figures
  fig_samples.png     - example images per class (downscaled copy of the EDA grid)
  fig_label_eff.png   - label-efficiency curves for the RGB models (main result)
  fig_bands.png       - accuracy GAIN of extra spectral bands over RGB, paired by seed
  fig_spectral.png    - spectral signature of every class (copy of the EDA figure)
  fig_confusion.png   - confusion matrices at 1% labels: small CNN vs ResNet-50 vs ViT-S

Usage:  python src/make_report_figures.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path("report/latex")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})

# read EVERY results file (RGB runs, first band run a+b, corrected band rerun c+d), so new runs are never missed
df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("results/runs_*.csv"))], ignore_index=True)
df["cfg"] = df["model"] + "/" + df["init"] + "/" + df["input"]
classes = json.loads(Path("data/splits.json").read_text())["classes"]
SHORT = ["AnnualCrop", "Forest", "Herbaceous", "Highway", "Industrial", "Pasture", "PermCrop",
         "Residential", "River", "SeaLake"]


def curve(ax, cfg, label, **kw):
    """Plot mean +/- std over seeds of test accuracy vs % of labels for one setting."""
    s = df[df.cfg == cfg].groupby("frac").test_acc.agg(["mean", "std"]).reset_index()
    ax.errorbar(s.frac * 100, s["mean"] * 100, yerr=s["std"].fillna(0) * 100, marker="o", ms=4,
                capsize=2, lw=1.5, label=label, **kw)


# ---------------------------------------------------------------- 1. samples + spectral (copies)
im = Image.open("figures/sample_grid.png")
im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
im.save(OUT / "fig_samples.png", optimize=True)
Image.open("figures/spectral_signatures.png").save(OUT / "fig_spectral.png", optimize=True)

# ---------------------------------------------------------------- 2. label efficiency
fig, ax = plt.subplots(figsize=(6.2, 4.0))
curve(ax, "cnn/scratch/rgb", "Small CNN (scratch)", color="#1f77b4")
curve(ax, "resnet50/scratch/rgb", "ResNet-50 (scratch)", color="#2ca02c")
curve(ax, "resnet50/imagenet/rgb", "ResNet-50 (ImageNet)", color="#ff7f0e")
curve(ax, "vit_s/imagenet/rgb", "ViT-S/16 (ImageNet-21k)", color="#d62728")
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 25, 100]); ax.set_xticklabels(["1", "2", "5", "10", "25", "100"])
ax.set_xlabel("Labelled training data (% of the 21,600 training images, log scale)")
ax.set_ylabel("Test accuracy (%)")
ax.grid(alpha=.3); ax.legend(loc="lower right", frameon=True)
plt.tight_layout(); plt.savefig(OUT / "fig_label_eff.png", dpi=200); plt.close()

# ---------------------------------------------------------------- 3. band gain (paired by seed)
base = df[df.cfg == "resnet50/imagenet/rgb_ms"].set_index(["frac", "seed"]).test_acc
fig, ax = plt.subplots(figsize=(6.2, 3.8))
styles = [("resnet50/imagenet/rgb+nir", "RGB + NIR", "#ff7f0e"),
          ("resnet50/imagenet/rgb+nir+rededge", "RGB + NIR + red-edge (B05-B07,B8A)", "#2ca02c"),
          ("resnet50/imagenet/rgb+nir+swir", "RGB + NIR + SWIR (B11,B12)", "#d62728"),
          ("resnet50/imagenet/all13", "All 13 bands", "#1f77b4")]
for cfg, lab, col in styles:
    d = ((df[df.cfg == cfg].set_index(["frac", "seed"]).test_acc - base) * 100).groupby("frac").agg(["mean", "std"])
    ax.errorbar(d.index * 100, d["mean"], yerr=d["std"], marker="o", ms=4, capsize=2, lw=1.5, label=lab, color=col)
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 25, 100]); ax.set_xticklabels(["1", "2", "5", "10", "25", "100"])
ax.set_xlabel("Labelled training data (%, log scale)")
ax.set_ylabel("Accuracy gain over R,G,B bands (points)")
ax.grid(alpha=.3); ax.legend(frameon=True, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_bands.png", dpi=200); plt.close()

# ---------------------------------------------------------------- 4. confusion matrices at 1% labels
def conf(model, init, frac):
    cm = np.zeros((10, 10))
    for f in sorted(glob.glob(f"results/logits/{model}_{init}_rgb_{frac}_*.npz")):
        z = np.load(f)
        cm += np.bincount(z["y"] * 10 + z["logits"].argmax(1), minlength=100).reshape(10, 10)
    return cm / cm.sum(1, keepdims=True)


panels = [("cnn", "scratch", "Small CNN (scratch)"), ("resnet50", "imagenet", "ResNet-50 (ImageNet)"),
          ("vit_s", "imagenet", "ViT-S/16 (ImageNet-21k)")]
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9))
for ax, (m, i, t) in zip(axes, panels):
    cm = conf(m, i, 0.01)
    im_ = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_xticklabels(SHORT, rotation=70, ha="right", fontsize=6.5)
    ax.set_yticks(range(10)); ax.set_yticklabels(SHORT if ax is axes[0] else [], fontsize=6.5)
    for a in range(10):
        for b in range(10):
            if a != b and cm[a, b] >= 0.05:
                ax.text(b, a, f"{cm[a, b] * 100:.0f}", ha="center", va="center", fontsize=6, color="black")
    ax.set_title(t, fontsize=9); ax.set_xlabel("predicted", fontsize=8)
axes[0].set_ylabel("true", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_confusion.png", dpi=200); plt.close()
print("wrote", sorted(p.name for p in OUT.glob("fig_*.png")))
