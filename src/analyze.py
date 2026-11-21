"""
analyze.py  -  turn results/runs.csv into tables and plots for the report.

Main output: the LABEL-EFFICIENCY CURVE
    x-axis : how many labelled training images we used (% of the train set, log scale)
    y-axis : test accuracy
    one line per (model / init / input) setting, with error bars = standard deviation
    over the random seeds.
A curve that stays high on the left (few labels) is "label-efficient".
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=["results/runs.csv"],
                    help="one or more results files (e.g. runs_rgb.csv runs_ms.csv); they are merged")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.read_csv(p) for p in a.csv], ignore_index=True)
    # One readable name per setting, e.g. "resnet50 / imagenet / rgb".
    df["cfg"] = df["model"] + " / " + df["init"] + " / " + df["input"]

    # Average over seeds: mean accuracy, its std (error bar), mean F1, number of seeds.
    g = df.groupby(["cfg", "frac"]).agg(acc=("test_acc", "mean"), acc_sd=("test_acc", "std"),
                                         f1=("test_f1", "mean"), n=("seed", "count")).reset_index()
    g.to_csv(out / "summary.csv", index=False)
    # Print a compact table: rows = settings, columns = label fractions.
    print(g.pivot(index="cfg", columns="frac", values="acc").round(4).to_string())

    def curve(cfgs, name, title):
        """Draw one plot with one line per setting in `cfgs`."""
        plt.figure(figsize=(7, 4.5))
        for c in cfgs:
            s = g[g.cfg == c].sort_values("frac")
            plt.errorbar(s.frac * 100, s.acc * 100, yerr=s.acc_sd.fillna(0) * 100,
                         marker="o", capsize=3, label=c)
        plt.xscale("log")
        plt.xlabel("% of labelled training data (log scale)")
        plt.ylabel("Test accuracy (%)")
        plt.title(title)
        plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(out / name, dpi=200); plt.close()

    # Figure 1: all RGB models (CNN vs ResNet vs ViT, scratch vs pretrained).
    rgb = [c for c in g.cfg.unique() if c.endswith("/ rgb")]
    if rgb:
        curve(rgb, "label_efficiency_rgb.png", "Label efficiency on EuroSAT (RGB), mean ± std over seeds")
    # Figure 2: effect of extra spectral bands (same model, different inputs).
    # (the jpg-based "/ rgb" line is excluded: the ablation compares inputs from the same 13-band files)
    ms = [c for c in g.cfg.unique() if c.startswith("resnet50 / imagenet") and not c.endswith("/ rgb")]
    if len(ms) > 1:
        curve(ms, "band_ablation.png", "Effect of input bands (ResNet-50, ImageNet init)")


if __name__ == "__main__":
    main()
