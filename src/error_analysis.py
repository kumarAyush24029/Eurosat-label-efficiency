"""
error_analysis.py  -  WHERE do the models make mistakes, and how trustworthy are their scores?

Needs the predictions saved by train.py --save_logits  (results/logits/*.npz).
Each file name is  <model>_<init>_<input>_<frac>_<seed>.npz  and holds the raw class
scores (logits) for the test set plus the true labels.

For one chosen setting (e.g. resnet50 / imagenet / rgb) and each label fraction asked for,
this script (averaging over all seeds found) produces:
  1. confusion_<tag>_<frac>.png  - row-normalised confusion matrix: row = true class,
                                   column = predicted class. Off-diagonal cells = typical mix-ups.
  2. per_class_f1_<tag>.png/.csv - F1 score per class vs label fraction: which classes
                                   are hard when labels are scarce?
  3. calibration_<tag>.csv       - Expected Calibration Error (ECE): how well the model's
                                   confidence matches its real accuracy (0 = perfectly calibrated).

Usage:
  python src/error_analysis.py --model resnet50 --init imagenet --input rgb --fracs 0.01 0.1 1.0
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def softmax(z):
    z = z - z.max(1, keepdims=True)                 # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, bins=15):
    """Expected Calibration Error: average gap between confidence and accuracy over confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def load_runs(logits_dir, model, init, inp, frac):
    """All seeds for one setting -> list of (logits, y)."""
    pattern = f"{model}_{init}_{inp}_{frac}_*.npz"
    return [(np.load(f)["logits"], np.load(f)["y"]) for f in sorted(Path(logits_dir).glob(pattern))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits_dir", default="results/logits")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--model", required=True); ap.add_argument("--init", required=True)
    ap.add_argument("--input", default="rgb")
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    classes = json.loads(Path(a.splits).read_text())["classes"]
    tag = f"{a.model}_{a.init}_{a.input}"

    f1_rows, ece_rows = {}, []
    for frac in a.fracs:
        runs = load_runs(a.logits_dir, a.model, a.init, a.input, frac)
        if not runs:
            print("no logits found for frac", frac); continue
        # Sum the confusion matrices of all seeds, then normalise each row to sum to 1.
        cm = sum(confusion_matrix(y, lg.argmax(1), labels=range(len(classes))) for lg, y in runs)
        cm = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] >= 0.02:                                  # print only non-trivial cells
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{tag}, {frac * 100:g}% labels ({len(runs)} seeds)")
        plt.colorbar(im, fraction=0.046); plt.tight_layout()
        plt.savefig(out / f"confusion_{tag}_{frac}.png", dpi=200); plt.close()

        # Per-class F1, averaged over seeds.
        f1_rows[frac] = np.mean([f1_score(y, lg.argmax(1), average=None, labels=range(len(classes)))
                                 for lg, y in runs], axis=0)
        # Calibration, averaged over seeds.
        ece_rows.append({"frac": frac, "ece": float(np.mean([ece(softmax(lg), y) for lg, y in runs])),
                         "seeds": len(runs)})

    if f1_rows:
        df = pd.DataFrame(f1_rows, index=classes).round(3)
        df.columns = [f"{c * 100:g}%" for c in df.columns]
        df.to_csv(out / f"per_class_f1_{tag}.csv")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(df.values, cmap="viridis", vmin=0.3, vmax=1, aspect="auto")
        ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
        ax.set_xlabel("% labelled data"); ax.set_title(f"Per-class F1 - {tag}")
        plt.colorbar(im); plt.tight_layout()
        plt.savefig(out / f"per_class_f1_{tag}.png", dpi=200); plt.close()
        print(df.to_string())
    if ece_rows:
        pd.DataFrame(ece_rows).to_csv(out / f"calibration_{tag}.csv", index=False)
        print(pd.DataFrame(ece_rows).to_string(index=False))


if __name__ == "__main__":
    main()
