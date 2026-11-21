"""
train.py  -  train and evaluate models; one "run" = one row in results/runs.csv.

A RUN is defined by 5 settings:
    model  : cnn / resnet50 / vit_s ...
    init   : "scratch" (random weights) or "imagenet" (pre-trained weights)
    input  : which bands the network sees (rgb, rgb+nir, ..., all13)
    frac   : fraction of the labelled training set used (0.01 = 1% of labels)
    seed   : random seed (different seed = different labelled subset + different init)

This script loops over a GRID of such settings. Finished runs are skipped, so if
Kaggle/Colab disconnects you can simply start it again and it continues.

Example (on Kaggle/Colab):
  python src/train.py --rgb_root data/raw_rgb/EuroSAT_RGB --splits data/splits.json \
      --models cnn resnet50 vit_s --inits scratch imagenet --inputs rgb \
      --fracs 0.01 0.1 1.0 --seeds 0 1 2

Quick check on a CPU laptop:  add  --smoke  (1 epoch only, just verifies the code runs).
"""
import argparse
import copy
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import data as D
import models as M
import splits as S

# Columns of the results file.
FIELDS = ["model", "init", "input", "frac", "seed", "n_train",
          "val_acc", "test_acc", "test_f1", "epochs", "steps", "sec"]


# ----------------------------------------------------------------------------
# Small helpers for the results file
# ----------------------------------------------------------------------------
def done_keys(csv_path):
    """Return the set of runs already stored in the CSV (so we can skip them)."""
    if not Path(csv_path).exists():
        return set()
    with open(csv_path) as f:
        return {(r["model"], r["init"], r["input"], r["frac"], r["seed"]) for r in csv.DictReader(f)}


def append_row(csv_path, row):
    """Add one finished run to the CSV (writes the header the first time)."""
    new = not Path(csv_path).exists()
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
@torch.no_grad()                                   # no gradients needed when just predicting
def predict(model, X, mean, std, dev, size, bs=512, amp=False):
    """Return the raw class scores (logits) for every image in X, in batches."""
    model.eval()                                   # switch off training-only behaviour (BatchNorm updates etc.)
    outs = []
    for i in range(0, len(X), bs):
        xb = D.normalize(X[i:i + bs].to(dev), mean.to(dev), std.to(dev))
        if size != xb.shape[-1]:                   # resize 64x64 -> e.g. 112x112 for ResNet/ViT
            xb = F.interpolate(xb, size=size, mode="bilinear", align_corners=False)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):   # mixed precision = faster on GPU
            outs.append(model(xb).float().cpu())
    return torch.cat(outs)


# ----------------------------------------------------------------------------
# One full training run
# ----------------------------------------------------------------------------
def run_one(cfg, arrays, classes, dev, args):
    """Train one model with the settings in `cfg`; return a dict (one results row).

    cfg    : dict(model, init, input, frac, seed)
    arrays : {"train": (X, y), "val": (X, y), "test": (X, y)} already loaded in memory
    """
    # Make the run reproducible.
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    Xtr, ytr = arrays["train"]; Xva, yva = arrays["val"]; Xte, yte = arrays["test"]

    # --- 1. choose input bands -------------------------------------------------
    # If we loaded the 13-band data, keep only the bands this experiment asks for.
    if Xtr.shape[1] == 13:
        cols = D.BAND_GROUPS[cfg["input"]]
        Xtr, Xva, Xte = Xtr[:, cols], Xva[:, cols], Xte[:, cols]

    # --- 2. simulate limited labels --------------------------------------------
    # Take a class-balanced subset of the training set (nested across fractions).
    # frac = 1.0 means "use all labels".
    if cfg["frac"] < 1:
        idx = S.label_subset(np.arange(len(ytr)), ytr.numpy(), cfg["frac"], cfg["seed"])
    else:
        idx = np.arange(len(ytr))
    Xs, ys = Xtr[idx], ytr[idx]                    # the labelled images this run may use

    # Per-channel mean/std for normalisation (from the labelled subset; if it is tiny,
    # use the full train images for a stabler estimate - pixels only, no labels used).
    mean, std = D.channel_stats(Xs if len(idx) > 500 else Xtr)

    # --- 3. build the model ----------------------------------------------------
    pre = cfg["init"] == "imagenet"
    size = args.img if cfg["model"] != "cnn" else 64   # pretrained nets like larger inputs; our CNN uses 64
    model = M.build_model(cfg["model"], pre, in_chans=Xs.shape[1], img_size=size).to(dev)

    # --- 4. training schedule --------------------------------------------------
    bs = min(args.bs, len(idx))                    # batch size can't exceed the number of images
    # With only ~200 images one epoch is just 1-2 updates, so we make sure every run
    # gets at least `min_steps` weight updates (more epochs when data is small).
    ep = max(args.epochs, math.ceil(args.min_steps * bs / len(idx)))
    if args.smoke:
        ep = 1
    spe = max(1, len(idx) // bs)                   # steps per epoch
    total = ep * spe
    lr = args.lr_pre if pre else args.lr_scratch   # smaller learning rate when fine-tuning pretrained weights
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    # Learning-rate schedule: short linear warm-up, then smooth cosine decay to 0.
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp = dev.type == "cuda"                       # mixed precision only on GPU
    scaler = torch.amp.GradScaler(enabled=amp)

    # --- 5. training loop ------------------------------------------------------
    t0 = time.time(); best, best_state = -1, None
    eval_every = max(1, ep // 20)                  # check validation accuracy ~20 times per run
    for e in range(ep):
        model.train()
        perm = torch.randperm(len(idx))            # new random order each epoch
        for i in range(spe):
            b = perm[i * bs:(i + 1) * bs]
            # normalise -> random flips/rotations -> (optional) resize
            xb = D.dihedral_aug(D.normalize(Xs[b].to(dev), mean.to(dev), std.to(dev)))
            if size != xb.shape[-1]:
                xb = F.interpolate(xb, size=size, mode="bilinear", align_corners=False)
            yb = ys[b].to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                # label smoothing = don't be 100% sure of labels; helps when data is small
                loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()          # compute gradients
            scaler.step(opt); scaler.update()      # update weights
            sched.step()                           # update learning rate
        # Periodically measure validation accuracy and remember the best weights.
        if (e + 1) % eval_every == 0 or e == ep - 1:
            acc = (predict(model, Xva, mean, std, dev, size, amp=amp).argmax(1) == yva).float().mean().item()
            if acc > best:
                best, best_state = acc, copy.deepcopy(model.state_dict())

    # --- 6. final evaluation on the TEST set -----------------------------------
    # Use the weights that were best on VALIDATION (never pick using test data!).
    model.load_state_dict(best_state)
    logits = predict(model, Xte, mean, std, dev, size, amp=amp)
    pred = logits.argmax(1).numpy()
    row = dict(cfg, n_train=len(idx), val_acc=round(best, 4),
               test_acc=round(float((pred == yte.numpy()).mean()), 4),
               test_f1=round(float(f1_score(yte.numpy(), pred, average="macro")), 4),   # macro-F1: every class counts equally
               epochs=ep, steps=total, sec=round(time.time() - t0, 1))
    # Optionally keep the raw predictions for later analysis (confusion matrix, calibration...).
    if args.save_logits:
        name = "_".join(str(cfg[k]) for k in cfg)
        np.savez_compressed(Path(args.out).parent / "logits" / name, logits=logits.numpy(), y=yte.numpy())
    return row


# ----------------------------------------------------------------------------
# Command line: loop over the whole grid of settings
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_root", help="folder with the RGB EuroSAT class folders")
    ap.add_argument("--ms_root", help="folder with the 13-band EuroSAT class folders")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--cache", default="data/cache", help="where decoded arrays are cached")
    ap.add_argument("--out", default="results/runs.csv")
    ap.add_argument("--models", nargs="+", default=["cnn"])
    ap.add_argument("--inits", nargs="+", default=["scratch"], help="scratch and/or imagenet")
    ap.add_argument("--inputs", nargs="+", default=["rgb"],
                    help="rgb | rgb+nir | rgb+nir+rededge | rgb+nir+swir | all13")
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.01, 0.1, 1.0], help="label fractions")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min_steps", type=int, default=600, help="minimum weight updates per run")
    ap.add_argument("--bs", type=int, default=128, help="batch size")
    ap.add_argument("--img", type=int, default=112, help="input size for resnet/vit (cnn always uses 64)")
    ap.add_argument("--lr_scratch", type=float, default=1e-3)
    ap.add_argument("--lr_pre", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true", help="1 epoch only, to test the code")
    ap.add_argument("--save_logits", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    # Load only the data version(s) we need (13-band data is ~2 GB, so only if requested).
    need_ms = any(i != "rgb" for i in args.inputs)
    sources = {}
    if "rgb" in args.inputs:
        sources["rgb"] = D.load_split_arrays(args.rgb_root, args.splits, args.cache, "rgb")
    if need_ms:
        sources["ms"] = D.load_split_arrays(args.ms_root, args.splits, args.cache, "ms")
    classes = next(iter(sources.values()))[1]
    if args.save_logits:
        (Path(args.out).parent / "logits").mkdir(parents=True, exist_ok=True)

    finished = done_keys(args.out)
    # Nested loops = the grid. Order: input -> model -> init -> label fraction -> seed.
    for inp in args.inputs:
        arrays = sources["rgb" if inp == "rgb" and "rgb" in sources else "ms"][0]
        for model in args.models:
            for init in args.inits:
                for frac in args.fracs:
                    for seed in args.seeds:
                        cfg = dict(model=model, init=init, input=inp, frac=frac, seed=seed)
                        if (model, init, inp, str(frac), str(seed)) in finished:
                            continue                      # already done earlier -> skip
                        row = run_one(cfg, arrays, classes, dev, args)
                        append_row(args.out, row)         # save immediately (safe if the session dies)
                        print(row, flush=True)


if __name__ == "__main__":
    main()
