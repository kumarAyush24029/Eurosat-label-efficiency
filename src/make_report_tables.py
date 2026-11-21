"""
make_report_tables.py  -  write the LaTeX tables of the report straight from the results.

Reads : results/runs_*.csv, results/logits/*.npz, data/splits.json
Writes: report/latex/tab_main.tex   accuracy of the 4 RGB models vs % of labels
        report/latex/tab_bands.tex  band ablation (missing settings are shown as TBD)
        report/latex/tab_f1.tex     per-class F1 at 1% of the labels
        report/latex/tab_ece.tex    calibration (ECE, mean confidence, accuracy)

Re-run after new results arrive:  python src/make_report_tables.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OUT = Path("report/latex")
OUT.mkdir(parents=True, exist_ok=True)
FRACS = [0.01, 0.02, 0.05, 0.1, 0.25, 1.0]
HEAD = " & ".join(["1\\%", "2\\%", "5\\%", "10\\%", "25\\%", "100\\%"])

files = sorted(glob.glob("results/runs_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["cfg"] = df["model"] + "/" + df["init"] + "/" + df["input"]
stat = df.groupby(["cfg", "frac"]).test_acc.agg(["mean", "std"]).mul(100)


def cell(cfg, frac, best=None):
    """'98.7 $\\pm$ 0.1' (bold if it is the best of its column); TBD if there are no runs yet."""
    if (cfg, frac) not in stat.index:
        return "\\textbf{TBD}"
    m, s = stat.loc[(cfg, frac)]
    t = f"{m:.1f} $\\pm$ {s:.1f}"
    return f"\\textbf{{{t}}}" if best is not None and abs(m - best) < 1e-9 else t


def table(rows, caption, label, first_col="Model / input"):
    """rows = [(cfg, display name)]; the best mean of every column is printed in bold."""
    best = {}
    for f in FRACS:
        vals = [stat.loc[(c, f)]["mean"] for c, _ in rows if (c, f) in stat.index]
        best[f] = max(vals) if vals else None
    lines = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{|l|c|c|c|c|c|c|}", "\\hline",
             f"{first_col} & {HEAD} \\\\ \\hline \\hline"]
    for cfg, name in rows:
        lines.append(name + " & " + " & ".join(cell(cfg, f, best[f]) for f in FRACS) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ main table
main_rows = [("cnn/scratch/rgb", "Small CNN (scratch)"),
             ("resnet50/scratch/rgb", "ResNet-50 (scratch)"),
             ("resnet50/imagenet/rgb", "ResNet-50 (ImageNet)"),
             ("vit_s/imagenet/rgb", "ViT-S/16 (ImageNet-21k)")]
(OUT / "tab_main.tex").write_text(table(
    main_rows, "Test accuracy (\\%, mean $\\pm$ std over 3 seeds) on EuroSAT RGB for different percentages of the "
    "21,600 labelled training images (1\\% = 216 images). Best value per column in bold.", "tab:main"))

# ------------------------------------------------------------------ band ablation
band_rows = [("resnet50/imagenet/rgb_ms", "R,G,B"),
             ("resnet50/imagenet/rgb+nir", "R,G,B + B08 (NIR)"),
             ("resnet50/imagenet/rgb+nir+rededge", "R,G,B + B08 + B05,B06,B07,B8A (red-edge)"),
             ("resnet50/imagenet/rgb+nir+swir", "R,G,B + B08 + B11,B12 (SWIR)"),
             ("resnet50/imagenet/rgb+nir+b12+b8a", "R,G,B + B08 + B12,B8A (first run)"),
             ("resnet50/imagenet/rgb+nir+b05b06b07b09", "R,G,B + B08 + B05--B07,B09 (first run)"),
             ("resnet50/imagenet/all13", "All 13 bands")]
(OUT / "tab_bands.tex").write_text(table(
    band_rows, "Band ablation with an ImageNet-pre-trained ResNet-50 (test accuracy \\%, mean $\\pm$ std over 3 seeds). "
    "All inputs come from the 13-band files. B02, B03, B04 = blue, green, red; B08 = near-infrared; "
    "B05--B07 and B8A = red-edge/narrow NIR; B11, B12 = short-wave infrared. "
    "The two ``first run'' rows are band sets that were run before we discovered that the band order in the files differs "
    "from the documentation (see text).", "tab:bands", first_col="Input bands"))

# ------------------------------------------------------------------ per-class F1 at 1% of labels
classes = json.loads(Path("data/splits.json").read_text())["classes"]
settings = [("cnn", "scratch", "CNN"), ("resnet50", "scratch", "R50 scratch"),
            ("resnet50", "imagenet", "R50 ImageNet"), ("vit_s", "imagenet", "ViT-S")]
f1 = {}
for m, i, name in settings:
    per_seed = []
    for p in sorted(glob.glob(f"results/logits/{m}_{i}_rgb_0.01_*.npz")):
        z = np.load(p)
        per_seed.append(f1_score(z["y"], z["logits"].argmax(1), average=None, labels=range(10)))
    f1[name] = np.mean(per_seed, axis=0) * 100
lines = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{5pt}",
         "\\begin{tabular}{|l|c|c|c|c|}", "\\hline",
         "Class & " + " & ".join(n for _, _, n in settings) + " \\\\ \\hline \\hline"]
for k, c in enumerate(classes):
    best = max(f1[n][k] for _, _, n in settings)
    cells = [(f"\\textbf{{{f1[n][k]:.1f}}}" if abs(f1[n][k] - best) < 1e-9 else f"{f1[n][k]:.1f}") for _, _, n in settings]
    lines.append(c + " & " + " & ".join(cells) + " \\\\")
lines.append("\\hline Macro average & " + " & ".join(f"{f1[n].mean():.1f}" for _, _, n in settings) + " \\\\ \\hline")
lines += ["\\end{tabular}",
          "\\caption{Per-class F1 score (\\%) on the test set when only 1\\% of the labels (about 22 images per class) are "
          "used, averaged over 3 seeds. Best value per class in bold.}", "\\label{tab:f1}", "\\end{table}"]
(OUT / "tab_f1.tex").write_text("\n".join(lines))


# ------------------------------------------------------------------ calibration
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return tot


lines = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{|l|ccc|ccc|}", "\\hline",
         " & \\multicolumn{3}{c|}{1\\% of labels} & \\multicolumn{3}{c|}{100\\% of labels} \\\\",
         "Model & acc. & conf. & ECE & acc. & conf. & ECE \\\\ \\hline \\hline"]
for m, i, name in settings:
    row = []
    for fr in ("0.01", "1.0"):
        R = [np.load(p) for p in sorted(glob.glob(f"results/logits/{m}_{i}_rgb_{fr}_*.npz"))]
        acc = np.mean([(r["logits"].argmax(1) == r["y"]).mean() for r in R]) * 100
        conf = np.mean([softmax(r["logits"]).max(1).mean() for r in R]) * 100
        e = np.mean([ece(softmax(r["logits"]), r["y"]) for r in R]) * 100
        row += [f"{acc:.1f}", f"{conf:.1f}", f"{e:.1f}"]
    lines.append(name + " & " + " & ".join(row) + " \\\\")
lines += ["\\hline", "\\end{tabular}",
          "\\caption{Calibration on the test set: accuracy, mean confidence and Expected Calibration Error (all in \\%, 15 bins), "
          "averaged over 3 seeds. Training used label smoothing 0.1; no post-hoc calibration was applied.}",
          "\\label{tab:ece}", "\\end{table}"]
(OUT / "tab_ece.tex").write_text("\n".join(lines))
print("wrote", sorted(p.name for p in OUT.glob("tab_*.tex")), "from", [Path(f).name for f in files])
