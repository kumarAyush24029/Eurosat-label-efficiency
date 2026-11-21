import json, sys
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(True)}
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(True)}
cells = [
md("# Team 21 - EuroSAT label-efficiency runs (Kaggle)\nSettings: **Accelerator = GPU T4/P100**, **Internet = On**. Run top to bottom. Results append to `results/runs.csv` (re-running skips finished runs, so it is safe to restart)."),
code("""!git clone https://github.com/kumarAyush24029/Eurosat-label-efficiency.git /kaggle/working/proj 2>/dev/null || echo 'clone failed - upload src/ instead'
%cd /kaggle/working/proj"""),
code("""# data (official Zenodo mirror): RGB ~90MB, 13-band ~2GB
!mkdir -p data && cd data && \
 wget -q -nc -O rgb.zip 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1' && unzip -q -n rgb.zip -d raw_rgb && \
 wget -q -nc -O ms.zip  'https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1'  && unzip -q -n ms.zip  -d raw_ms
!ls data/raw_rgb data/raw_ms
!python src/splits.py --root data/raw_rgb/EuroSAT_RGB --out data/splits.json"""),
code("""# sanity: tiny run on GPU
!python src/train.py --rgb_root data/raw_rgb/EuroSAT_RGB --splits data/splits.json --out results/smoke.csv --models cnn --fracs 0.01 --seeds 0 --epochs 2"""),
md("## Stage 1 - RGB core grid (run first; this alone gives a complete project)"),
code("""COMMON = "--rgb_root data/raw_rgb/EuroSAT_RGB --ms_root data/raw_ms/EuroSAT_MS --splits data/splits.json --out results/runs.csv --fracs 0.01 0.02 0.05 0.1 0.25 1.0 --seeds 0 1 2 --save_logits"
!python src/train.py $COMMON --models cnn --inits scratch --inputs rgb --epochs 40
!python src/train.py $COMMON --models resnet50 --inits imagenet scratch --inputs rgb --epochs 20
!python src/train.py $COMMON --models vit_s --inits imagenet --inputs rgb --epochs 20"""),
md("## Stage 2 - multispectral / band ablation (research angle)"),
code("""!python src/train.py $COMMON --models resnet50 --inits imagenet --inputs rgb+nir rgb+nir+rededge rgb+nir+swir all13 --epochs 20"""),
code("""!python src/analyze.py --csv results/runs.csv --out figures
!zip -qr results_bundle.zip results figures && ls -la results_bundle.zip   # download this file from the Output tab"""),
]
nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
json.dump(nb, open("notebooks/run_kaggle.ipynb","w"), indent=1)
