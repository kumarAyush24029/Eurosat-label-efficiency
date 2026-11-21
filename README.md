# Label-Efficient Land-Use Classification from Satellite Imagery (EuroSAT)

Team 21 · Deep Learning · Project P21 · Dataset: EuroSAT (Sentinel-2, 10 classes, 27,000 patches of 64x64, RGB + 13 bands)

> New here? Read **[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)** (plain-language walkthrough). The report is in **[report/latex/Elcvia2e.tex](report/latex/Elcvia2e.tex)** (ELCVIA template).

## Research question
**How much labelled data do we really need, and which model family, pre-training strategy and input bands buy the most label-efficiency?**

## Protocol (fixed)
- One stratified 80/10/10 split (seed 0): 21,600 train / 2,700 val / 2,700 test. Same image ids for RGB and 13-band (`data/splits.json`).
- Label fractions 1, 2, 5, 10, 25, 100 % of the *train* set; class-balanced, nested, 3 seeds each.
- Test set used once per run; checkpoint chosen on validation accuracy.
- Same recipe for all models: AdamW, cosine schedule, label smoothing 0.1, flips/rotations, >= 600 updates.

## Main results (test accuracy %, mean over 3 seeds)
| Model | 1% | 5% | 10% | 25% | 100% |
|---|---|---|---|---|---|
| Small CNN (scratch) | 76.3 | 90.5 | 93.0 | 96.5 | 98.5 |
| ResNet-50 (scratch) | 73.5 | 87.6 | 90.1 | 92.8 | 97.2 |
| ResNet-50 (ImageNet) | 88.3 | 96.0 | 96.8 | 98.0 | 98.7 |
| ViT-S/16 (ImageNet-21k) | **90.6** | **97.1** | **97.5** | **98.3** | **98.8** |

Pre-training is the strongest lever (worth ~3.6-5x more labels at 1%). Near-infrared adds ~2 points at 1% labels; feeding all 13 bands to a pre-trained network does not help.
Full tables, band ablation, error and calibration analysis: see the report.

**Band-order pitfall:** in the EuroSAT multispectral GeoTIFFs the narrow NIR band B8A is stored *last* (order B01-B08, B09, B10, B11, B12, B8A), unlike the documentation. `src/data.py` has the verified order. The first band-ablation run used the documented order; its two extended groups are kept in `results/runs_ms_a/b.csv` under their true band lists, and the intended groups were re-run (`notebooks/run_kaggle_ms_fix.ipynb`).

## Layout
```
data/splits.json      fixed split (raw data and caches are git-ignored)
src/splits.py         split + nested label subsets      src/train.py     train/evaluate a grid of runs
src/data.py           loading, normalisation, augment   src/analyze.py   curves from results CSVs
src/models.py         CNN / ResNet-50 / ViT-S           src/eda.py       class counts, samples, spectral signatures
src/error_analysis.py confusion, per-class F1, ECE      src/make_report_figures.py, make_report_tables.py
notebooks/            Kaggle notebooks (RGB grid, 13-band grid, corrected band groups)
results/              runs_*.csv (one row per run) and test-set logits
figures/              plots      report/latex/  LaTeX report + figures + tables
```

## Reproduce
- Splits: `python src/splits.py --root <EuroSAT_RGB folder> --out data/splits.json` (identical output, seed 0).
- Runs: import the notebooks in `notebooks/` into Kaggle (GPU T4 x2, Internet on) and "Save & Run All".
- Figures/tables for the report: `python src/make_report_figures.py` and `python src/make_report_tables.py`.
- The laptop has no GPU: use it only for EDA and smoke tests (`python src/train.py ... --smoke`).
