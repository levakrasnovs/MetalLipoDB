"""
MetalLipoDB modeling — train_chemprop.py
========================================
Chemprop v2 (D-MPNN) trained from scratch in every outer fold of the frozen
SMILES and DOI splits and of LOMO/LOGO (common.evaluation_folds). The shared descriptors of features.py enter as extra
per-molecule features (x_d), so the metal is described exactly as for
LightGBM and TabPFN: the default atom featurizer knows atomic numbers 1-36 and
53 only, and Ru, Pt, Ir, Au, ... all fall into one "other" atom type.

Graph representations of the same complex:
  DOT     ligand set without the metal (canonical_ligands)
  DATIVE  whole complex with dative metal-ligand bonds
  SINGLE  whole complex, dative bonds replaced by single bonds

Hyperparameters are fixed a priori and identical for all splits. Inside an
outer fold, VALIDATION_SHARE of the training rows, grouped by the key of the
split (ligand set for SMILES, article otherwise), is the early-stopping set;
the model trains on the rest, i.e. on ~90% of what LightGBM sees.

Outputs (results/): metrics/chemprop_{fold_metrics,summary}.csv,
oof_predictions/chemprop_oof.csv, figures/chemprop_representations.png.

--pretrained chemeleon initialises the message passing with the CheMeleon
weights (Burns et al., JCIM 2026; 8.7M parameters, d_h = 2048) and fine-tunes
the whole model end to end, as recommended by its authors (their SI S5-S6:
default Chemprop settings; a permanently frozen encoder underperforms in
regression). Everything else is identical to Chemprop from scratch; outputs
are written as chemeleon_ft_*.
Finished (representation, split, fold) jobs are kept in the fold metrics and
skipped on restart; --fresh discards them.

Usage:
    python train_chemprop.py [--representations DATIVE] [--splits SMILES DOI LOMO LOGO] [--fresh]
                             [--pretrained none|chemeleon] [--accelerator cpu|mps]
"""

import argparse
import gc
import os
import shutil
import warnings
from pathlib import Path
from time import perf_counter

import chemprop
import lightning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from chemprop import data as cpdata, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from rdkit import Chem

from common import (FIGURE_DIR, METAL_SPLITS, METRICS_DIR, OOF_DIR, REPRESENTATIONS, RESULTS_DIR, SPLITS,
                    environment, evaluation_folds, fold_seed, load_modeling_data, molecules,
                    regression_metrics, shared_features, split_validation)
from features import COMMON_CATEGORICAL, COMMON_NUMERIC

warnings.filterwarnings("ignore")
torch.set_num_threads(min(8, os.cpu_count() or 1))

MODEL = f"Chemprop v{chemprop.__version__}"
CHEMELEON_CHECKPOINT = Path.home() / ".cache" / "chemeleon" / "chemeleon_mp.pt"
PRETRAINED = "none"
ACCELERATOR = "cpu"
ALL_SPLITS = [s for s, _ in SPLITS] + METAL_SPLITS
CHECKPOINT_ROOT = RESULTS_DIR / "_chemprop_checkpoints"
FOLD_METRICS_PATH = METRICS_DIR / "chemprop_fold_metrics.csv"
OOF_PATH = OOF_DIR / "chemprop_oof.csv"
OUT = "chemprop"

BATCH_SIZE, MAX_EPOCHS, PATIENCE = 128, 60, 6
MESSAGE_PASSING = dict(d_h=300, depth=3, dropout=0.0)
# one head for every split, fixed before training (the notebook tuned it per split)
HEAD = dict(hidden_dim=300, n_layers=1, dropout=0.10, criterion="MAE")


# ── inputs ────────────────────────────────────────────────────────────────────
# ── one fold ──────────────────────────────────────────────────────────────────
def new_message_passing():
    if PRETRAINED == "none":
        return nn.BondMessagePassing(**MESSAGE_PASSING)
    state = torch.load(CHEMELEON_CHECKPOINT, weights_only=True, map_location="cpu")
    message_passing = nn.BondMessagePassing(**state["hyper_parameters"])
    message_passing.load_state_dict(state["state_dict"])
    return message_passing  # trainable: fine-tuned end to end


def configure(pretrained, accelerator):
    global PRETRAINED, ACCELERATOR, MODEL, CHECKPOINT_ROOT, FOLD_METRICS_PATH, OOF_PATH, OUT
    PRETRAINED, ACCELERATOR = pretrained, accelerator
    OUT = "chemprop" if pretrained == "none" else "chemeleon_ft"
    if pretrained == "chemeleon":
        assert CHEMELEON_CHECKPOINT.exists(), "run train_chemeleon.py once to download the checkpoint"
        MODEL = f"CheMeleon fine-tuned (Chemprop v{chemprop.__version__})"
    CHECKPOINT_ROOT = RESULTS_DIR / f"_{OUT}_checkpoints"
    FOLD_METRICS_PATH = METRICS_DIR / f"{OUT}_fold_metrics.csv"
    OOF_PATH = OOF_DIR / f"{OUT}_oof.csv"


def dataset(mols, y, row_ids, indices, extra, featurizer):
    points = [cpdata.MoleculeDatapoint(mol=Chem.Mol(mols[i]), y=np.asarray([y[i]], dtype=np.float32),
                                       x_d=extra[k], name=str(row_ids[i]))
              for k, i in enumerate(indices)]
    return cpdata.MoleculeDataset(points, featurizer)


def train_predict(data, mols, y, split, fold, outer_train_idx, test_idx, representation):
    seed = fold_seed(fold)
    train_idx, validation_idx = split_validation(data, split, outer_train_idx, seed)
    train_x, validation_x, test_x = shared_features(data, train_idx, validation_idx, test_idx)
    pl.seed_everything(seed, workers=True, verbose=False)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    row_ids = data["row_id"].to_numpy()
    train_set = dataset(mols, y, row_ids, train_idx, train_x, featurizer)
    target_scaler = train_set.normalize_targets()
    validation_set = dataset(mols, y, row_ids, validation_idx, validation_x, featurizer)
    validation_set.normalize_targets(target_scaler)
    test_set = dataset(mols, y, row_ids, test_idx, test_x, featurizer)

    message_passing = new_message_passing()
    ffn = nn.RegressionFFN(input_dim=message_passing.output_dim + train_x.shape[1],
                           hidden_dim=HEAD["hidden_dim"], n_layers=HEAD["n_layers"], dropout=HEAD["dropout"],
                           criterion=getattr(nn.metrics, HEAD["criterion"])(),
                           output_transform=nn.UnscaleTransform.from_standard_scaler(target_scaler))
    model = models.MPNN(message_passing, nn.MeanAggregation(), ffn, batch_norm=False,
                        metrics=[nn.metrics.MAE(), nn.metrics.RMSE()])
    checkpoint_dir = CHECKPOINT_ROOT / f"{representation}_{split}_{fold}".replace(" ", "_")
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, filename="best-{epoch:02d}-{val_loss:.4f}",
                                 monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(logger=False, accelerator=ACCELERATOR, devices=1, max_epochs=MAX_EPOCHS,
                         callbacks=[checkpoint, EarlyStopping(monitor="val_loss", mode="min",
                                                              patience=PATIENCE, min_delta=1e-4)],
                         enable_progress_bar=False, enable_model_summary=False, deterministic=True)
    started = perf_counter()
    trainer.fit(model,
                cpdata.build_dataloader(train_set, batch_size=BATCH_SIZE, num_workers=0, seed=seed),
                cpdata.build_dataloader(validation_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=False))
    best = models.MPNN.load_from_checkpoint(checkpoint.best_model_path, map_location="cpu")
    test_loader = cpdata.build_dataloader(test_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    prediction = torch.cat(trainer.predict(best, dataloaders=test_loader)).cpu().numpy().reshape(-1)
    best_epoch = int(Path(checkpoint.best_model_path).name.split("epoch=")[1].split("-")[0])
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    fold_row = {"model": MODEL, "representation": representation, "split": split, "fold": fold,
                "n_train": len(train_idx), "n_validation": len(validation_idx), "n_test": len(test_idx),
                "extra_features": train_x.shape[1], "best_epoch": best_epoch,
                "seconds": perf_counter() - started, **regression_metrics(y[test_idx], prediction)}
    rows = data.iloc[test_idx]
    oof = pd.DataFrame({"row_id": rows["row_id"].to_numpy(), "representation": representation,
                        "split": split, "fold": fold, "domain": rows["domain"].to_numpy(),
                        "metal": rows["metal"].to_numpy(), "doi": rows["doi"].to_numpy(),
                        "observed": y[test_idx], "prediction": prediction,
                        "absolute_error": np.abs(y[test_idx] - prediction)})
    del train_set, validation_set, test_set, model, best, trainer
    gc.collect()
    return fold_row, oof


# ── all folds ─────────────────────────────────────────────────────────────────
def cross_validate(data, representations, splits, fresh):
    y = pd.to_numeric(data["value"], errors="raise").to_numpy(np.float32)
    if fresh or not (FOLD_METRICS_PATH.exists() and OOF_PATH.exists()):
        fold_rows, oof = [], pd.DataFrame()
    else:
        fold_rows = pd.read_csv(FOLD_METRICS_PATH).to_dict("records")
        oof = pd.read_csv(OOF_PATH)
        # resume only results of this table and this Chemprop version
        assert set(oof["row_id"]) <= set(data["row_id"]) and {r["model"] for r in fold_rows} == {MODEL}, \
            f"{FOLD_METRICS_PATH.name} is from another dataset or version; rerun with --fresh"
    done = {(r["representation"], r["split"], str(r["fold"])) for r in fold_rows}
    jobs = [(rep, split, *f) for rep in representations for split in splits
            for f in evaluation_folds(data, split) if (rep, split, str(f[0])) not in done]
    print(f"Jobs: {len(jobs)} to run, {len(done)} already finished")
    start, mols = perf_counter(), {}
    for n, (rep, split, fold, train_idx, test_idx) in enumerate(jobs, 1):
        if rep not in mols:
            mols = {rep: molecules(data, rep)}
        fold_row, fold_oof = train_predict(data, mols[rep], y, split, fold, train_idx, test_idx, rep)
        fold_rows.append(fold_row)
        oof = pd.concat([oof, fold_oof], ignore_index=True)
        pd.DataFrame(fold_rows).to_csv(FOLD_METRICS_PATH, index=False)
        oof.to_csv(OOF_PATH, index=False)
        eta = (perf_counter() - start) / n * (len(jobs) - n)
        print(f"[{n:02d}/{len(jobs)}] {rep} {split} fold {fold}: MAE={fold_row['mae']:.4f}; "
              f"epoch={fold_row['best_epoch']}; ETA={eta / 60:.1f} min", flush=True)
    return pd.DataFrame(fold_rows), oof


def summarise(fold_metrics, oof):
    rows = []
    for (rep, split), group in oof.groupby(["representation", "split"], sort=False):
        f = fold_metrics[fold_metrics["representation"].eq(rep) & fold_metrics["split"].eq(split)]
        rows.append({"model": MODEL, "representation": rep, "split": split, "n_oof": len(group),
                     **regression_metrics(group["observed"], group["prediction"]),
                     **{f"fold_{m}_{a}": getattr(f[m], a)() for m in ["mae", "rmse", "r2"]
                        for a in ["mean", "std"]},
                     "seconds_total": f["seconds"].sum()})
    summary = pd.DataFrame(rows).sort_values(["split", "mae"])
    print(summary.round(4).to_string(index=False))
    return summary


def plot(summary):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (split, _) in zip(axes, SPLITS):
        s = summary[summary["split"].eq(split)].sort_values("mae", ascending=False)
        ax.barh(s["representation"], s["mae"], xerr=s["fold_mae_std"], color="#267365")
        ax.set(title=f"{split} split", xlabel="OOF MAE")
    fig.suptitle("Chemprop molecular representations", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{OUT}_representations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=["DATIVE"])
    parser.add_argument("--splits", nargs="+", choices=ALL_SPLITS, default=ALL_SPLITS)
    parser.add_argument("--fresh", action="store_true", help="discard finished folds")
    parser.add_argument("--pretrained", choices=["none", "chemeleon"], default="none")
    parser.add_argument("--accelerator", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    configure(args.pretrained, args.accelerator)
    environment(Chemprop=chemprop.__version__, Lightning=lightning.__version__,
                Torch=f"{torch.__version__} (CUDA: {torch.cuda.is_available()})")
    data = load_modeling_data()
    print("Rows:", len(data), "| shared numeric/categorical:", len(COMMON_NUMERIC), len(COMMON_CATEGORICAL))
    fold_metrics, oof = cross_validate(data, args.representations, args.splits, args.fresh)
    summary = summarise(fold_metrics, oof)
    summary.to_csv(METRICS_DIR / f"{OUT}_summary.csv", index=False)
    plot(summary)


if __name__ == "__main__":
    main()
