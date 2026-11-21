# Handoff - state at the end of session 1 (21 Nov 2026, night)

**Deadline: 22 Nov 2026, 23:59.** Ayush is doing the whole project alone for Team 21.
Repo: https://github.com/kumarAyush24029/Eurosat-label-efficiency (public, branch `main`). Local folder: `C:\Projects\Team21-DL Project` (CPU only, no LaTeX installed).

## What is finished
- Pipeline in `src/` (splits, data, models, train, analyze, eda, error_analysis, make_report_figures, make_report_tables). All files are commented.
- Fixed split `data/splits.json` (21,600 / 2,700 / 2,700).
- **72 RGB runs** (`results/runs_rgb.csv`, test logits in `results/logits/`): small CNN, ResNet-50 (scratch / ImageNet), ViT-S/16 (ImageNet-21k) x 6 label fractions x 3 seeds.
- **90 first-run band-ablation runs** (`results/runs_ms_a.csv`, `runs_ms_b.csv`), ResNet-50 ImageNet init.
- EDA figures, confusion matrices, per-class F1, calibration, final report figures and tables (`report/latex/`).
- LaTeX report in the professor's ELCVIA template: `report/latex/Elcvia2e.tex` (+ `fig_*.png`, `tab_*.tex`). It compiles to 10 pages in an editable Overleaf copy: https://www.overleaf.com/project/6b01c0ecaf1e33b1fad82a62 (the professor's template, view-only: https://www.overleaf.com/project/6af2d7b5d1cff5d7464c3829).

## Key results (test accuracy %, mean of 3 seeds; 1% / 5% / 10% / 25% / 100% of labels)
- Small CNN scratch: 76.3 / 90.5 / 93.0 / 96.5 / 98.5
- ResNet-50 scratch: 73.5 / 87.6 / 90.1 / 92.8 / 97.2
- ResNet-50 ImageNet: 88.3 / 96.0 / 96.8 / 98.0 / 98.7
- ViT-S ImageNet-21k: 90.6 / 97.1 / 97.5 / 98.3 / 98.8
- Pre-training is the biggest lever (CNN needs ~3.6x / ~5.2x more labels to match ResNet-50 / ViT-S at 1%). With all labels everything is within noise (+-0.4 pt on 2,700 test images).
- Bands (ResNet-50 ImageNet, vs R,G,B from the 13-band files): +B08 gives +2.0 pt at 1% labels, +1.0 at 2%, ~0 beyond 5%. All 13 bands is slightly WORSE than R,G,B at 1-10% labels. JPEG RGB is ~1 pt worse than RGB taken from the GeoTIFFs at 1% labels.

## The band-order bug (my mistake, already disclosed in the report)
In the EuroSAT-MS `.tif` files the narrow NIR band **B8A is stored last**: order is B01-B08, B09, B10, B11, B12, B8A (found from the spectral-signature plot: channel 9 ~ 0 for all classes = cirrus B10; channel 12 follows B08). The first run used the documented order, so:
- old `rgb+nir+swir` really = R,G,B + B08 + B12 + B8A -> stored as `rgb+nir+b12+b8a` (best of the first run: 92.3% at 1%, 99.2% at 100%)
- old `rgb+nir+rededge` really = R,G,B + B08 + B05-B07 + B09 -> stored as `rgb+nir+b05b06b07b09`
`src/data.py` is fixed (`BANDS` order, `BAND_GROUPS`). The two groups as originally intended (`rgb+nir+rededge` = B05,B06,B07,B8A; `rgb+nir+swir` = B11,B12) were **re-run** in Kaggle `notebook15cd3e6dee` (Version 2; notebook `notebooks/run_kaggle_ms_fix.ipynb`). The user says it has FINISHED, but **its results are not yet downloaded or merged**.

## Remaining steps (in order)
1. **Get the results.** Kaggle -> notebook15cd3e6dee -> Version 2 -> Output tab -> `proj/results_ms_fix.zip` (~1-3 MB; do NOT click "Download All", it holds ~4 GB). Extract into `kaggle_out/ms_fix/` (git-ignored). Copy `results/runs_ms_c.csv` and `results/runs_ms_d.csv` into the project `results/`, and `figures/spectral_signatures.png` (now with the corrected band names) into `figures/`. If the run did not finish or failed, read its log (view-only `/log` page) and resubmit.
2. `python src/make_report_tables.py` (fills the two TBD rows of the band table automatically) and `python src/make_report_figures.py` (band-gain figure now uses the corrected groups; also copies the spectral figure). Open `figures/` PNGs and check them.
3. **Edit `report/latex/Elcvia2e.tex`:**
   - replace the one `\tbd{...}` in section 5.2 with real numbers for the corrected red-edge and SWIR groups;
   - the sentence "The set B08+B12+B8A reached the highest accuracy of the ablation" must be re-checked against the new numbers;
   - re-check every statement that depends on the band results: abstract, "Near-infrared helps..." paragraph, Discussion ("gain of up to about 3 points"), Conclusion ("NIR and SWIR information adds up to about 3 points");
   - `\nruns` is 198 (72 + 90 + 36): check the corrected run really produced 36 rows;
   - the spectral figure caption must match the corrected figure.
4. **Compile in Overleaf** (copy project above). Upload the changed files with the Chrome `file_upload` tool on the hidden input of Add > Upload (Elcvia2e.tex, tab_bands.tex, fig_bands.png, fig_spectral.png; choose Overwrite) and Recompile. Check page count (header uses `\finpag`, currently 10) and that there are no `TBD` left.
5. `git add -A && git commit && git push`; update README numbers if the band conclusions change.
6. Ask the user to: confirm affiliation/department/city ("Sitare University, India" is a guess from the email domain), author names (Ishant Bhoyar, Ayush Kumar, Gaurav Kumar, Jagriti Pandey), correspondence email, and download the final PDF from Overleaf. Ask whether the course also needs slides (not built yet) or another submission format. Optional extras only if time remains: slide outline; self-supervised/semi-supervised ideas are listed as future work in the report, not implemented.

## Practical notes
- Kaggle: never open a notebook's /edit page in two tabs (causes ConcurrencyViolation / "Failed to save draft"); monitor via view-only `/log` and `/output?scriptVersionId=` pages. Import resets the accelerator to None: set GPU T4 x2 and check "Run with GPU" in Save Version > Advanced. A commit stuck in Queued for 30+ min: cancel via Active Events > ... > Stop Session (ask the user first) and resubmit. Import by URL: File > Import Notebook > Link > `https://raw.githubusercontent.com/kumarAyush24029/Eurosat-label-efficiency/main/notebooks/<file>.ipynb`.
- Use the claude-in-chrome tools (the user is signed in there) for Kaggle/Overleaf; the built-in browser is not signed in.
- No local LaTeX: Overleaf is the compile check. `texcheck`-style static checks (braces/environments) are cheap but not a compile.
- Style the user wants: commented simple code, plain-language explanations, honesty about mistakes, ask before risky/outward actions.
