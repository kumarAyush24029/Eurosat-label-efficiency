# Land-Use Classification from Satellite Imagery with Limited Labels

Team 21 · Deep Learning · Project P21 · Dataset: EuroSAT (Sentinel-2, 10 classes, 27,000 patches of 64x64, RGB + 13 bands)

> New here? Read **[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)** first: what the project is, why each model is used, and how to run it.

## Research question
**How much labelled data do we really need — and which model family, pretraining strategy and
input bands buy the most label-efficiency?**

## Fixed protocol (do not change after the first experiment)
- Split: stratified 80/10/10 train/val/test, seed 0 (`src/splits.py` -> `data/splits.json`). Same image IDs for RGB and 13-band.
- Label fractions: 1%, 2%, 5%, 10%, 25%, 100% of the *train* set; class-balanced, nested, 3-5 seeds each.
- Test set is touched only for final numbers. Model selection uses val only.
- Metrics: accuracy, macro-F1, per-class F1; report mean ± std (or 95% CI) over seeds.

## Experiment matrix
| # | Model | Init | Input | Labels |
|---|-------|------|-------|--------|
| A | Small CNN | scratch | RGB | all fractions |
| B | ResNet-50 | scratch / ImageNet | RGB | all fractions |
| C | ViT-B/16 (or ViT-S) | scratch / ImageNet | RGB | all fractions |
| D | best of B/C | ImageNet (first layer adapted) | 13-band | all fractions |
| E | ResNet-50 | self-supervised on unlabeled EuroSAT train (SimCLR/MAE) -> fine-tune | RGB (+13-band) | all fractions |
| F | best of above | FixMatch (semi-supervised: labelled fraction + rest unlabeled) | RGB | 1%, 5%, 10% |
| G | optional | remote-sensing pretrained weights (e.g. SSL4EO-S12) | 13-band | all fractions |

Analysis: label-efficiency curves; RGB vs 13-band gap by label fraction; band ablation
(drop/keep band groups: RGB, +NIR, +red-edge, +SWIR); linear probe vs full fine-tune;
Grad-CAM; confusion matrix (Highway/River/PermanentCrop are the hard classes); calibration (ECE).
Discussion point: random patch splits leak spatial context, so reported accuracies are optimistic.

## Roadmap
1. **Setup** - repo, env, download data, `splits.py`.            
2. **EDA** - class counts, sample grids, per-band statistics.
3. **Baselines** - A, B at 1/10/100% on RGB; confirm the pipeline end to end.
4. **Full grid** - B, C, D across all fractions and seeds.
5. **Advanced** - E (SSL), F (FixMatch), optional G.
6. **Analysis** - curves, ablations, Grad-CAM, error analysis.
7. **Write-up** - report + slides + clean notebook.

## Layout
```
data/        raw data + splits.json   (git-ignored)
src/         splits.py, (data.py, models.py, train.py, eval.py)
notebooks/   Kaggle/Colab run notebooks, EDA
results/     one CSV row per run (model, init, input, frac, seed, val/test acc, macro-F1)
figures/     final plots
report/      write-up
```

## Where things run
- Write code locally (VS Code) in this folder; push to a private GitHub repo.
- Train on **Kaggle** (free T4/P100, ~30 GPU-h/week) or Colab: clone the repo, point at the data, save `results/*.csv` back.
- This laptop has no NVIDIA GPU (CPU-only PyTorch): use it for EDA, debugging and tiny smoke tests only.
