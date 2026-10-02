"""
MetalLipoDB modeling — train_chemeleon.py
=========================================
Frozen CheMeleon encoder (pretrained Chemprop message passing, official
checkpoint) plus a small regression head trained in every outer fold of
SMILES, DOI, LOMO and LOGO (common.evaluation_folds). The head receives the
2048-d molecular embedding and the shared descriptors of features.py, so the
metal is described exactly as for the other models. The frozen encoder needs
the standard Chemprop featurizer (atomic numbers 1-36 and 53): Ru, Pt, Ir,
Au, ... are one "other" atom type in the graph.

Hyperparameters are fixed a priori and identical for all splits; early
stopping uses VALIDATION_SHARE of the training rows grouped like the split
(common.split_validation), as for Chemprop.

Embeddings are cached in results/chemeleon_embeddings/ and reused only for the
same molecules (SHA-256 of the SMILES list) and the same RDKit version.

Outputs (results/): metrics/chemeleon_{fold_metrics,summary}.csv,
oof_predictions/chemeleon_oof.csv. The run is fast, so it always starts fresh.

Usage:
    python train_chemeleon.py [--representations DATIVE] [--splits SMILES DOI LOMO LOGO]
"""

import argparse
import copy
import hashlib
import os
import urllib.request
from pathlib import Path
from time import perf_counter

import chemprop
import numpy as np
import pandas as pd
import torch
from chemprop import featurizers, models, nn
from chemprop.data.collate import BatchMolGraph
from rdkit import rdBase
from sklearn.preprocessing import StandardScaler
from torch import nn as torch_nn
from torch.utils.data import DataLoader, TensorDataset

from common import (METAL_SPLITS, METRICS_DIR, OOF_DIR, REPRESENTATIONS, RESULTS_DIR, SPLITS, environment,
                    evaluation_folds, fold_seed, load_modeling_data, molecules, regression_metrics,
                    shared_features, split_validation)
from features import COMMON_CATEGORICAL, COMMON_NUMERIC

torch.set_num_threads(min(8, os.cpu_count() or 1))

MODEL = "CheMeleon (frozen)"
ALL_SPLITS = [s for s, _ in SPLITS] + METAL_SPLITS
CHECKPOINT = Path.home() / ".cache" / "chemeleon" / "chemeleon_mp.pt"
CHECKPOINT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CACHE_DIR = RESULTS_DIR / "chemeleon_embeddings"
# one head for every split, fixed before training (the notebook tuned it per split)
HIDDEN, DROPOUT, LOSS = [300], 0.1, "L1"
LR, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE = 1e-3, 1e-5, 256, 200, 20


# ── embeddings ────────────────────────────────────────────────────────────────
def encoder():
    if not CHECKPOINT.exists():
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading the CheMeleon checkpoint:", CHECKPOINT_URL)
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)
    state = torch.load(CHECKPOINT, weights_only=True, map_location="cpu")
    message_passing = nn.BondMessagePassing(**state["hyper_parameters"])
    message_passing.load_state_dict(state["state_dict"])
    model = models.MPNN(message_passing, nn.MeanAggregation(),
                        nn.RegressionFFN(input_dim=message_passing.output_dim)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def embed(mols, batch_size=16):
    model, featurizer, chunks = encoder(), featurizers.SimpleMoleculeMolGraphFeaturizer(), []
    started = perf_counter()
    for start in range(0, len(mols), batch_size):
        graph = BatchMolGraph([featurizer(m) for m in mols[start:start + batch_size]])
        with torch.no_grad():
            chunks.append(model.fingerprint(graph).cpu().numpy())
        done = min(start + batch_size, len(mols))
        if done % 1024 == 0 or done == len(mols):
            print(f"  encoded {done}/{len(mols)}; ETA {(perf_counter() - started) / done * (len(mols) - done):.0f} s")
    return np.vstack(chunks).astype(np.float32)


def embeddings(data, representation):
    column = "canonical_ligands" if representation == "DOT" else "canonical_complex_dative"
    key = hashlib.sha256(("\n".join(data[column]) + rdBase.rdkitVersion + representation).encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{representation.lower()}_{key}.npy"
    if path.exists():
        print(representation, "embeddings from cache:", path.name)
        return np.load(path)
    print(representation, "embeddings: encoding", len(data), "molecules")
    x = embed(molecules(data, representation))
    np.save(path, x)
    return x


# ── head ──────────────────────────────────────────────────────────────────────
class Head(torch_nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers, previous = [], input_dim
        for width in HIDDEN:
            layers += [torch_nn.Linear(previous, width), torch_nn.ReLU(), torch_nn.Dropout(DROPOUT)]
            previous = width
        self.network = torch_nn.Sequential(*layers, torch_nn.Linear(previous, 1))

    def forward(self, x):
        return self.network(x)


def fit_head(X_train, y_train, X_val, y_val, seed):
    torch.manual_seed(seed)
    model = Head(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch_nn.L1Loss() if LOSS == "L1" else torch_nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))
    X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
    best_state, best_loss, best_epoch, stale = None, np.inf, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val), y_val))
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale, best_state = val_loss, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model.eval(), best_epoch


def train_predict(data, emb, y, split, fold, outer_train_idx, test_idx, representation):
    seed = fold_seed(fold)
    train_idx, val_idx = split_validation(data, split, outer_train_idx, seed)
    tab_train, tab_val, tab_test = shared_features(data, train_idx, val_idx, test_idx)
    X_train, X_val, X_test = (np.hstack([emb[idx], tab]).astype(np.float32)
                              for idx, tab in [(train_idx, tab_train), (val_idx, tab_val), (test_idx, tab_test)])
    scaler = StandardScaler().fit(y[train_idx, None])
    started = perf_counter()
    model, best_epoch = fit_head(X_train, scaler.transform(y[train_idx, None]).astype(np.float32),
                                 X_val, scaler.transform(y[val_idx, None]).astype(np.float32), seed)
    with torch.no_grad():
        prediction = scaler.inverse_transform(model(torch.from_numpy(X_test)).numpy()).ravel()
    fold_row = {"model": MODEL, "representation": representation, "split": split, "fold": fold,
                "n_train": len(train_idx), "n_validation": len(val_idx), "n_test": len(test_idx),
                "n_features": X_train.shape[1], "best_epoch": best_epoch, "seconds": perf_counter() - started,
                **regression_metrics(y[test_idx], prediction)}
    rows = data.iloc[test_idx]
    oof = pd.DataFrame({"row_id": rows["row_id"].to_numpy(), "representation": representation,
                        "split": split, "fold": fold, "domain": rows["domain"].to_numpy(),
                        "metal": rows["metal"].to_numpy(), "doi": rows["doi"].to_numpy(),
                        "observed": y[test_idx], "prediction": prediction,
                        "absolute_error": np.abs(y[test_idx] - prediction)})
    return fold_row, oof


def summarise(fold_metrics, oof):
    rows = []
    for (rep, split), group in oof.groupby(["representation", "split"], sort=False):
        f = fold_metrics[fold_metrics["representation"].eq(rep) & fold_metrics["split"].eq(split)]
        rows.append({"model": MODEL, "representation": rep, "split": split, "n_oof": len(group),
                     **regression_metrics(group["observed"], group["prediction"]),
                     **{f"fold_{m}_{a}": getattr(f[m], a)() for m in ["mae", "rmse", "r2"]
                        for a in ["mean", "std"]},
                     "seconds_total": f["seconds"].sum()})
    summary = pd.DataFrame(rows)
    print(summary.round(4).to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=["DATIVE"])
    parser.add_argument("--splits", nargs="+", choices=ALL_SPLITS, default=ALL_SPLITS)
    args = parser.parse_args()
    environment(Chemprop=chemprop.__version__, Torch=torch.__version__)
    data = load_modeling_data()
    y = pd.to_numeric(data["value"], errors="raise").to_numpy(np.float32)
    print("Rows:", len(data), "| shared numeric/categorical:", len(COMMON_NUMERIC), len(COMMON_CATEGORICAL))
    fold_rows, oofs = [], []
    for rep in args.representations:
        emb = embeddings(data, rep)
        jobs = [(split, *f) for split in args.splits for f in evaluation_folds(data, split)]
        for n, (split, fold, train_idx, test_idx) in enumerate(jobs, 1):
            fold_row, oof = train_predict(data, emb, y, split, fold, train_idx, test_idx, rep)
            fold_rows.append(fold_row)
            oofs.append(oof)
            print(f"[{n:02d}/{len(jobs)}] {rep} {split} fold {fold}: MAE={fold_row['mae']:.4f}; "
                  f"epoch={fold_row['best_epoch']}", flush=True)
    fold_metrics, oof = pd.DataFrame(fold_rows), pd.concat(oofs, ignore_index=True)
    tag = "" if set(args.splits) == set(ALL_SPLITS) and args.representations == ["DATIVE"] else \
        "_" + "_".join(args.representations + args.splits)
    fold_metrics.to_csv(METRICS_DIR / f"chemeleon{tag}_fold_metrics.csv", index=False)
    oof.to_csv(OOF_DIR / f"chemeleon{tag}_oof.csv", index=False)
    summarise(fold_metrics, oof).to_csv(METRICS_DIR / f"chemeleon{tag}_summary.csv", index=False)


if __name__ == "__main__":
    main()
