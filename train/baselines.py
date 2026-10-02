"""
MetalLipoDB modeling — baselines.py
===================================
Trivial baselines on the frozen SMILES and DOI folds and on leave-one-metal-out
(LOMO) and leave-one-group-out (LOGO); for LOMO/LOGO the pooled metrics over
all held-out rows are the headline numbers:
  per-metal mean   mean logP of the training rows with the same metal
                   (overall training mean for metals absent from training)
  Crippen          sum of the Crippen logP of the neutral forms of the ligands
                   (RDKit Uncharger on the ligand set), linearly calibrated on
                   the training folds
  kNN              mean logP of the K most similar training ligand sets (count
                   Tanimoto on unfolded Morgan r = 2, as in analyze_domain.py);
                   repeated measurements of one ligand set are averaged first
  Ridge            ridge regression on the shared descriptors of features.py
                   (no fingerprint); fold-local imputation, scaling and one-hot
                   encoding, alpha by RidgeCV on the training folds only

Outputs (results/): metrics/baselines_{fold_metrics,summary}.csv,
oof_predictions/baselines_oof.csv

Usage:
    python baselines.py
"""

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import Crippen, SanitizeMol
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (METAL_SPLITS, METRICS_DIR, OOF_DIR, SPLITS, environment, evaluation_folds,
                    load_modeling_data, parsed, regression_metrics)
from features import COMMON_CATEGORICAL, COMMON_NUMERIC

K_NEIGHBOURS = 5

_uncharger = rdMolStandardize.Uncharger()


def neutral_crippen(smiles):
    mol = _uncharger.uncharge(parsed(smiles))
    SanitizeMol(mol)
    return Crippen.MolLogP(mol)


def per_metal_mean(data, y, train_idx, test_idx):
    means = pd.Series(y[train_idx]).groupby(data["metal"].to_numpy()[train_idx]).mean()
    return data["metal"].iloc[test_idx].map(means).fillna(y[train_idx].mean()).to_numpy()


def crippen(x):
    def predict(data, y, train_idx, test_idx):
        model = LinearRegression().fit(x[train_idx, None], y[train_idx])
        return model.predict(x[test_idx, None])
    return predict


def ligand_similarity(ligand_sets):
    """Count Tanimoto between all unique ligand sets."""
    generator = GetMorganGenerator(radius=2)
    fps = [generator.GetSparseCountFingerprint(parsed(s)) for s in ligand_sets]
    return np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps], dtype=np.float32)


def knn(codes, similarity):
    def predict(data, y, train_idx, test_idx):
        train = pd.Series(y[train_idx]).groupby(codes[train_idx]).mean()
        sim = similarity[np.ix_(codes[test_idx], train.index.to_numpy())]
        nearest = np.argpartition(-sim, K_NEIGHBOURS - 1, axis=1)[:, :K_NEIGHBOURS]
        return train.to_numpy()[nearest].mean(axis=1)
    return predict


def ridge(data, y, train_idx, test_idx):
    X = data[COMMON_NUMERIC + COMMON_CATEGORICAL].copy()
    X[COMMON_NUMERIC] = X[COMMON_NUMERIC].apply(pd.to_numeric, errors="coerce")
    X[COMMON_CATEGORICAL] = X[COMMON_CATEGORICAL].fillna("MISSING").astype(str)
    model = make_pipeline(
        ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                      StandardScaler()), COMMON_NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), COMMON_CATEGORICAL)]),
        RidgeCV(alphas=np.logspace(-2, 3, 11)))
    return model.fit(X.iloc[train_idx], y[train_idx]).predict(X.iloc[test_idx])


def main():
    environment()
    data = load_modeling_data()
    y = pd.to_numeric(data["value"], errors="raise").to_numpy(float)
    cache = {s: neutral_crippen(s) for s in data["canonical_ligands"].unique()}
    codes, ligand_sets = pd.factorize(data["canonical_ligands"])
    baselines = {"Per-metal mean": per_metal_mean,
                 "Crippen logP (neutral ligands)": crippen(data["canonical_ligands"].map(cache).to_numpy()),
                 f"kNN (k = {K_NEIGHBOURS}, count Tanimoto)": knn(codes, ligand_similarity(ligand_sets)),
                 "Ridge (descriptors)": ridge}

    fold_rows, oof_rows = [], []
    for split in [s for s, _ in SPLITS] + METAL_SPLITS:
        for fold, train_idx, test_idx in evaluation_folds(data, split):
            for name, predict in baselines.items():
                prediction = predict(data, y, train_idx, test_idx)
                fold_rows.append({"model": name, "split": split, "fold": fold,
                                  **regression_metrics(y[test_idx], prediction)})
                oof_rows.append(pd.DataFrame({"row_id": data["row_id"].iloc[test_idx].to_numpy(),
                                              "model": name, "split": split, "fold": fold,
                                              "observed": y[test_idx], "prediction": prediction}))
    folds = pd.DataFrame(fold_rows)
    summary = folds.groupby(["model", "split"], sort=False).agg(
        **{f"{m}_{s}": (m, "mean" if s == "mean" else "std")
           for m in ["mae", "rmse", "r2"] for s in ["mean", "std"]}).reset_index()
    oof = pd.concat(oof_rows, ignore_index=True)
    pooled = pd.DataFrame([{"model": m, "split": s, **{f"pooled_{k}": v for k, v in
                            regression_metrics(g["observed"], g["prediction"]).items()}}
                           for (m, s), g in oof.groupby(["model", "split"], sort=False)])
    summary = summary.merge(pooled, on=["model", "split"])
    print(summary.round(3).to_string(index=False))
    folds.to_csv(METRICS_DIR / "baselines_fold_metrics.csv", index=False)
    summary.to_csv(METRICS_DIR / "baselines_summary.csv", index=False)
    oof.to_csv(OOF_DIR / "baselines_oof.csv", index=False)


if __name__ == "__main__":
    main()
