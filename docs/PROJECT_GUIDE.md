# Project Guide (plain-language)

## 1. What is this project?
**Task:** look at a small satellite photo (64x64 pixels, from the Sentinel-2 satellite) and say what the
land is used for: forest, river, highway, industrial area, residential area, farmland, etc. (10 classes).
This is called **land-use classification** and it is used for mapping cities, farming, deforestation and so on.

**Dataset:** [EuroSAT](https://arxiv.org/abs/1709.00029) - 27,000 labelled images, 10 classes, 2,700 per class (roughly).
It comes in two versions: normal colour (**RGB**) and **13 spectral bands** (colours plus infrared that human eyes cannot see).

**The twist ("limited labels"):** in real life, getting labelled satellite images is expensive because an expert must look at each one.
So the real question is:

> **If we only have a few labelled images (1%, 5%, 10% ...), which model and which training strategy still works well?
> And do the extra non-RGB bands help when labels are scarce?**

We do not just train one model. We run a controlled study and draw **label-efficiency curves**
(accuracy vs. number of labels) for several models.

## 2. What we are going to do (the experiments)
| Step | What | Why |
|------|------|-----|
| 1 | Make ONE fixed train/val/test split (80/10/10) | every model is compared on the same images |
| 2 | Pretend only 1%, 2%, 5%, 10%, 25%, 100% of train labels exist | simulates label scarcity |
| 3 | Train 3 model families (see section 3), from scratch and pre-trained | measures the value of pre-training |
| 4 | Repeat with 3 random seeds, report mean +/- std | one run can be lucky; this shows the result is reliable |
| 5 | Re-run the best model with extra bands (RGB, +NIR, +red-edge, +SWIR, all 13) | the research angle: do non-RGB bands help, and when? |
| 6 | Analyse errors (confusion matrix, which classes are hard) | understand *why* the model fails |

The test set is touched only for the final number. Model selection uses the validation set.

## 3. Which models and why (short)
| Model | What it is | Why we include it |
|-------|-----------|-------------------|
| **Small CNN (scratch)** | tiny 4-stage convolutional net, random weights | **Baseline.** Shows what you get with no outside knowledge. |
| **ResNet-50 (scratch)** | deep CNN with skip connections, random weights | Shows if a *bigger* model helps when labels are few (spoiler: often it overfits). |
| **ResNet-50 (ImageNet pre-trained)** | same net, but starting from weights learned on 1.2M everyday photos | **Transfer learning:** it already knows edges/textures/shapes, so it needs far fewer labels. This is the main "label-efficiency" tool. |
| **ViT-Small (ImageNet-21k pre-trained)** | Vision Transformer: cuts the image into patches and uses attention instead of convolutions | A different idea from CNNs. Transformers are data-hungry from scratch but excellent when pre-trained, so it is an interesting contrast in the low-label regime. |

Why not more models? We have a short deadline, and these four settings already answer the question cleanly:
*no knowledge* (CNN) vs *more capacity* (ResNet scratch) vs *transfer learning* (ResNet/ViT pre-trained) vs *new architecture* (ViT).

## 4. How the code works (read in this order)
```
src/splits.py   -> makes data/splits.json (which image is train / val / test) and the 1%,10%... subsets
src/data.py     -> loads all images into memory, normalises them, random flips/rotations (augmentation)
src/models.py   -> the networks (SmallCNN, ResNet-50, ViT-S)
src/train.py    -> trains one model per "run" and appends a row to results/runs.csv. Loops over a grid.
src/analyze.py  -> reads runs.csv and draws the label-efficiency curves in figures/
notebooks/run_kaggle.ipynb -> runs all of the above on a free Kaggle GPU
```
Training recipe (same for all models, for fairness): AdamW optimiser, cosine learning-rate decay,
label smoothing 0.1, random flips/rotations, best checkpoint chosen on the validation set.

## 5. How to run it
**On your laptop (no GPU) - only for checking the code works:**
```
python src/splits.py --root data/raw_rgb/x/EuroSAT_RGB --out data/splits.json
python src/train.py --rgb_root data/raw_rgb/x/EuroSAT_RGB --splits data/splits.json --models cnn --fracs 0.01 --smoke
```
**On Kaggle (real experiments):** open `notebooks/run_kaggle.ipynb`, turn on GPU + Internet, run the cells top to bottom.
Results are written to `results/runs.csv`; the last cell makes the plots and a zip to download.

## 6. How to read the results
- `results/runs.csv`: one row per run (model, init, input bands, label fraction, seed, accuracy, macro-F1).
- `figures/label_efficiency_rgb.png`: the main figure. A line that is **high on the left** means the method works with few labels.
- `figures/band_ablation.png`: does adding bands (NIR, red-edge, SWIR) improve accuracy, especially at low label fractions?

## 7. Glossary
- **Transfer learning / pre-training:** start from weights trained on another big dataset instead of random numbers.
- **Fine-tuning:** continue training those weights on our small dataset.
- **Overfitting:** the model memorises the few training images and does badly on new ones; worst with few labels.
- **Augmentation:** random flips/rotations that create extra training examples for free.
- **Seed:** a number that fixes randomness; several seeds show how stable a result is.
- **Macro-F1:** average F1 score over classes, so rare/hard classes count as much as easy ones.
- **Multispectral:** images with more colour bands than RGB (e.g. near-infrared, where healthy plants are very bright).
