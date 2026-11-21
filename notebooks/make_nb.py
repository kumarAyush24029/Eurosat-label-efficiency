"""Generates the two Kaggle notebooks (run this file, then commit the .ipynb files).

  run_kaggle_rgb.ipynb : RGB core grid  (CNN, ResNet-50, ViT-S; scratch vs ImageNet)
  run_kaggle_ms.ipynb  : multispectral / band-ablation grid (ResNet-50, ImageNet init)

They are separate so they can run at the same time in two Kaggle sessions.
Both use the committed data/splits.json, so the train/val/test images are identical.
"""
import json

REPO = "https://github.com/kumarAyush24029/Eurosat-label-efficiency.git"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)}


def save(name, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(f"notebooks/{name}", "w"), indent=1)


CLONE = code(f"""# 1) get the code from GitHub (public repo)
!git clone {REPO} /kaggle/working/proj
%cd /kaggle/working/proj
!pip install -q timm tifffile imagecodecs""")

# ---------------------------------------------------------------- RGB notebook
save("run_kaggle_rgb.ipynb", [
    md("# Team 21 - EuroSAT label-efficiency: RGB grid\n"
       "Settings (right panel): **Accelerator = GPU**, **Internet = On**. Run cells top to bottom.\n"
       "Results are appended to `results/runs_rgb.csv`; finished runs are skipped, so restarting is safe."),
    CLONE,
    code("""# 2) download RGB EuroSAT (~90 MB, official Zenodo mirror)
!mkdir -p data && cd data && wget -q -O rgb.zip 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1' && unzip -q -n rgb.zip -d raw_rgb
!ls data/raw_rgb"""),
    code("""# 3) quick GPU sanity run (should take under a minute)
!python src/train.py --rgb_root data/raw_rgb/EuroSAT_RGB --splits data/splits.json --out results/smoke.csv --models cnn --fracs 0.01 --seeds 0 --epochs 2"""),
    md("## Main grid (CNN, ResNet-50 scratch/ImageNet, ViT-S ImageNet) x 6 label fractions x 3 seeds"),
    code("""COMMON = "--rgb_root data/raw_rgb/EuroSAT_RGB --splits data/splits.json --out results/runs_rgb.csv --fracs 0.01 0.02 0.05 0.1 0.25 1.0 --seeds 0 1 2 --save_logits"
!python src/train.py $COMMON --models cnn --inits scratch --inputs rgb --epochs 40
!python src/train.py $COMMON --models resnet50 --inits imagenet scratch --inputs rgb --epochs 20
!python src/train.py $COMMON --models vit_s --inits imagenet --inputs rgb --epochs 20"""),
    code("""# 4) plots + bundle to download (Output tab -> results_rgb.zip)
!python src/analyze.py --csv results/runs_rgb.csv --out figures
# error analysis: confusion matrices, per-class F1, calibration (uses the saved predictions)
!python src/error_analysis.py --model resnet50 --init imagenet --input rgb --fracs 0.01 0.1 1.0
!python src/error_analysis.py --model resnet50 --init scratch --input rgb --fracs 0.01 0.1 1.0
!python src/error_analysis.py --model vit_s --init imagenet --input rgb --fracs 0.01 0.1 1.0
!python src/eda.py --rgb_root data/raw_rgb/EuroSAT_RGB --out figures
!zip -qr results_rgb.zip results figures && ls -la results_rgb.zip"""),
])

# ---------------------------------------------------------------- MS notebook
save("run_kaggle_ms.ipynb", [
    md("# Team 21 - EuroSAT label-efficiency: multispectral / band ablation\n"
       "Settings: **Accelerator = GPU**, **Internet = On**. Results go to `results/runs_ms.csv`.\n"
       "Inputs compared (all from the 13-band files): `rgb_ms`, `rgb+nir`, `rgb+nir+rededge`, `rgb+nir+swir`, `all13`."),
    CLONE,
    code("""# 2) download 13-band EuroSAT (~2 GB; takes a few minutes)
!mkdir -p data && cd data && wget -q -O ms.zip 'https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1' && unzip -q -n ms.zip -d raw_ms
!ls data/raw_ms
!ls data/raw_ms/* | head"""),
    code("""# 3) sanity run on the 13-band loader (the folder name below must match the `ls` output above)
!python src/train.py --ms_root data/raw_ms/EuroSAT_MS --splits data/splits.json --out results/smoke_ms.csv --models cnn --inputs all13 --fracs 0.01 --seeds 0 --epochs 2"""),
    md("## Band-ablation grid (ResNet-50, ImageNet init) x 6 label fractions x 3 seeds\n"
       "Kaggle gives 2 GPUs per session but one training process uses only one, so we start TWO processes at once "
       "(one per GPU), each with its own results file. Cell 3 above must have run first: it builds the data cache that both processes read."),
    code("""%%bash
ARGS="--ms_root data/raw_ms/EuroSAT_MS --splits data/splits.json --fracs 0.01 0.02 0.05 0.1 0.25 1.0 --seeds 0 1 2 --models resnet50 --inits imagenet --epochs 20"
# GPU 0: three input settings; GPU 1: the other two. `sed -u` tags each log line with its GPU.
CUDA_VISIBLE_DEVICES=0 python src/train.py $ARGS --out results/runs_ms_a.csv --inputs rgb_ms all13 rgb+nir 2>&1 | sed -u 's/^/[GPU0] /' &
CUDA_VISIBLE_DEVICES=1 python src/train.py $ARGS --out results/runs_ms_b.csv --inputs rgb+nir+rededge rgb+nir+swir 2>&1 | sed -u 's/^/[GPU1] /' &
wait
echo ALL DONE"""),
    code("""# 4) bundle to download (Output tab -> results_ms.zip)
!python src/analyze.py --csv results/runs_ms_a.csv results/runs_ms_b.csv --out figures
!python src/eda.py --ms_root data/raw_ms/EuroSAT_MS --out figures
!zip -qr results_ms.zip results figures && ls -la results_ms.zip"""),
])


# ------------------------------------------------- MS notebook, corrected band groups (rerun)
save("run_kaggle_ms_fix.ipynb", [
    md("# Team 21 - EuroSAT: corrected band-ablation groups\n"
       "The first 13-band run assumed the wrong band order (B8A is stored LAST in the files). "
       "Here we rerun the two affected groups with the corrected indices: `rgb+nir+rededge` (B04,B03,B02,B08,B05,B06,B07,B8A) "
       "and `rgb+nir+swir` (B04,B03,B02,B08,B11,B12). Settings: **GPU T4 x2**, **Internet On**."),
    CLONE,
    code("""# 2) download 13-band EuroSAT (~2 GB)
!mkdir -p data && cd data && wget -q -O ms.zip 'https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1' && unzip -q -n ms.zip -d raw_ms
!ls data/raw_ms"""),
    code("""# 3) sanity run; also builds the data cache that the two processes below share
!python src/train.py --ms_root data/raw_ms/EuroSAT_MS --splits data/splits.json --out results/smoke_ms.csv --models cnn --inputs all13 --fracs 0.01 --seeds 0 --epochs 2"""),
    md("## Corrected groups: one training process per GPU"),
    code("""%%bash
ARGS="--ms_root data/raw_ms/EuroSAT_MS --splits data/splits.json --fracs 0.01 0.02 0.05 0.1 0.25 1.0 --seeds 0 1 2 --models resnet50 --inits imagenet --epochs 20"
CUDA_VISIBLE_DEVICES=0 python src/train.py $ARGS --out results/runs_ms_c.csv --inputs rgb+nir+rededge 2>&1 | sed -u 's/^/[GPU0] /' &
CUDA_VISIBLE_DEVICES=1 python src/train.py $ARGS --out results/runs_ms_d.csv --inputs rgb+nir+swir 2>&1 | sed -u 's/^/[GPU1] /' &
wait
echo ALL DONE"""),
    code("""# 4) spectral-signature figure (with corrected band names) + bundle
!python src/eda.py --ms_root data/raw_ms/EuroSAT_MS --out figures
!python src/analyze.py --csv results/runs_ms_c.csv results/runs_ms_d.csv --out figures
!zip -qr results_ms_fix.zip results figures && ls -la results_ms_fix.zip"""),
])
