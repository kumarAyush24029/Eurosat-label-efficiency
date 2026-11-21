# Label-Efficient Land-Use Classification from Sentinel-2 Imagery (EuroSAT)

**Team 21 · Deep Learning · Project P21**
<!-- TODO: add team member names/emails exactly as the course requires -->

> Items marked **[TODO]** are filled in once the Kaggle runs finish. Everything else is already true of the code in this repo.

## Abstract
[TODO - 150 words at the end] Land-use maps need labelled satellite imagery, which is expensive. We study how much labelled data
is needed on EuroSAT (27,000 Sentinel-2 patches, 10 classes) by training a small CNN, ResNet-50 and ViT-Small, from scratch
and from ImageNet weights, with 1%-100% of the training labels (3 seeds each), and by comparing RGB input with up to 13 spectral
bands. Main findings: [TODO: accuracy at 1% labels for each model; gain from pre-training; gain from extra bands; hardest classes].

## 1. Introduction
- Task: classify a 64x64 satellite patch into one of 10 land-use classes.
- Problem: labels need experts, so real projects have few. Question: **which model family and which input bands give the most accuracy per label?**
- Contributions: (i) a controlled label-efficiency study (same split, nested label subsets, 3 seeds); (ii) comparison of CNN vs ResNet vs ViT, scratch vs pre-trained;
  (iii) a band ablation (RGB, +NIR, +red-edge, +SWIR, all 13); (iv) error and calibration analysis.

## 2. Dataset
EuroSAT [1]: 27,000 georeferenced patches (64x64 px, 10 m resolution) from Sentinel-2, 10 classes (AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake), roughly 2,000-3,000 per class.
Two versions: RGB (.jpg) and all 13 spectral bands (.tif).
Figures: `figures/class_counts.png`, `figures/sample_grid.png`, `figures/spectral_signatures.png` [TODO after MS run].

**Split.** One stratified 80/10/10 split (seed 0): 21,600 train / 2,700 validation / 2,700 test, shared by the RGB and 13-band versions (`data/splits.json`).
**Label budgets.** Class-balanced subsets of the *train* set with 1%, 2%, 5%, 10%, 25%, 100% of labels (216 to 21,600 images in total, i.e. about 22 to 2,160 per class),
nested for each seed, so the learning curve changes only the amount of data. Seeds 0, 1, 2 give different subsets and initialisations.
The test set is used once per run, for the checkpoint chosen on validation accuracy.

## 3. Methods
| Model | Initialisation | Role |
|---|---|---|
| Small CNN (4 stages, ~1.2M params) | scratch | baseline with no outside knowledge |
| ResNet-50 [2] | scratch / ImageNet | effect of capacity and of transfer learning |
| ViT-Small/16 [3] | ImageNet-21k (fine-tuned on ImageNet-1k) | attention-based architecture, strong when pre-trained |

**Input.** Per-channel standardisation with train statistics. ResNet/ViT receive images upsampled from 64 to 112 px; the CNN uses 64 px.
For pre-trained models with more than 3 channels, `timm` adapts the first layer by re-using the RGB filters.
**Augmentation.** Random 90-degree rotations and flips (satellite imagery has no canonical orientation).
**Training (identical for all models).** AdamW (weight decay 0.05), lr 1e-3 (scratch) / 3e-4 (pre-trained), 5% warm-up then cosine decay, label smoothing 0.1,
mixed precision, batch size 128, at least 600 weight updates per run (so tiny label sets are trained to convergence), best checkpoint chosen on validation accuracy.
**Metrics.** Test accuracy and macro-F1 (mean +/- std over 3 seeds); per-class F1; Expected Calibration Error.

**Band ablation** (ResNet-50, ImageNet init): `rgb_ms` (R,G,B from the 13-band files), `rgb+nir`, `rgb+nir+rededge`, `rgb+nir+swir`, `all13`.

## 4. Results
### 4.1 Label efficiency (RGB)
Figure `figures/label_efficiency_rgb.png` - [TODO insert]. Table of mean +/- std accuracy per model and label fraction: `figures/summary.csv`.
[TODO: 3-4 sentences: ranking of models at 1% / 10% / 100%; how many labels the best scratch model needs to match the pre-trained model at 1%.]

### 4.2 Effect of spectral bands
Figure `figures/band_ablation.png` - [TODO]. [TODO: gain of each band group over `rgb_ms` at each label fraction; is the gain larger when labels are scarce?]

### 4.3 Error analysis
Confusion matrices `figures/confusion_*.png`, per-class F1 `figures/per_class_f1_*.png`, calibration `figures/calibration_*.csv` - [TODO].
[TODO: hardest classes (expected: Highway, River, PermanentCrop, HerbaceousVegetation vs Pasture); which classes improve most with more labels or bands; over/under-confidence.]

## 5. Discussion
[TODO] Answer: how many labels are enough? Is pre-training or extra bands the bigger lever? Does ViT beat ResNet at very low labels?

## 6. Limitations
- **Random patch split.** Patches from the same scene can fall in train and test, so absolute accuracies are optimistic; relative comparisons are still valid because all methods share the split.
- 3 seeds only; differences under about 1 point are within noise.
- Same hyper-parameters for all models (fair, but not individually tuned).
- Not explored: self-supervised pre-training on unlabeled EuroSAT, semi-supervised learning (FixMatch), remote-sensing-specific pre-trained models.

## 7. Conclusion
[TODO - 4 sentences]

## References
1. P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019. arXiv:1709.00029.
2. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
3. A. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
4. A. Steiner et al. *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers.* TMLR 2022 (source of the augreg ViT weights).
5. J. Deng et al. *ImageNet: A large-scale hierarchical image database.* CVPR 2009.
6. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization (AdamW).* ICLR 2019.
7. R. Wightman. *PyTorch Image Models (timm).* 2019. github.com/huggingface/pytorch-image-models
