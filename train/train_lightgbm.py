"""
MetalLipoDB modeling — train_lightgbm.py
========================================
LightGBM on the count Morgan fingerprint of the ligand set plus the shared
descriptors (features.py), 10-fold cross-validation on the frozen SMILES and DOI
splits. SHAP values are computed on each held-out fold after fitting and are not
used for any selection.

Fingerprint input (--fingerprint):
  ligands  ligand set without the metal (canonical_ligands), the default
  dative   whole complex with dative bonds (canonical_complex_dative): adds the
           environments of the metal and of the donor atoms bound to it

Outputs (results/): metrics/lightgbm{tag}_{fold_metrics,summary,shap_by_fold,
shap_features,shap_groups}.csv, oof_predictions/lightgbm{tag}_oof.csv,
figures/lightgbm{tag}_shap_{groups,top_features}.png; tag is empty for the
default and "_dative" otherwise.

Usage:
    python train_lightgbm.py [--fingerprint ligands|dative]
"""

import argparse
import warnings
from time import perf_counter

import lightgbm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lightgbm import LGBMRegressor

from common import (FIGURE_DIR, METAL_SPLITS, METRICS_DIR, OOF_DIR, SPLITS, environment,
                    evaluation_folds, load_modeling_data, regression_metrics)
from features import (COMMON_CATEGORICAL, COMMON_NUMERIC, feature_block,
                      ligand_count_fingerprint)

warnings.filterwarnings("ignore")

FP_SIZE = 2048
FINGERPRINT_COLUMN = {"ligands": "canonical_ligands", "dative": "canonical_complex_dative"}
PARAMS = dict(objective="regression", n_estimators=350, learning_rate=0.05, num_leaves=31,
              subsample=0.9, colsample_bytree=0.9, n_jobs=1, verbosity=-1)


def build_features(data, smiles_column):
    started = perf_counter()
    cache = {s: ligand_count_fingerprint(s, FP_SIZE) for s in data[smiles_column].unique()}
    morgan = [f"MorganBit_{i:04d}" for i in range(FP_SIZE)]
    fp = pd.DataFrame(np.vstack(data[smiles_column].map(cache)), columns=morgan, index=data.index)
    features = pd.concat([fp, data[COMMON_NUMERIC + COMMON_CATEGORICAL]], axis=1)
    columns = morgan + COMMON_NUMERIC + COMMON_CATEGORICAL
    print("Feature matrix:", features.shape, f"generated in {perf_counter() - started:.1f} s")
    return features, columns, morgan


def prepare_fold(features, columns, numeric, train_idx, test_idx):
    X_train = features.iloc[train_idx][columns].copy()
    X_test = features.iloc[test_idx][columns].copy()
    for column in COMMON_CATEGORICAL:
        train_values = X_train[column].astype("string").fillna("MISSING")
        test_values = X_test[column].astype("string").fillna("MISSING")
        categories = sorted(set(train_values) | {"MISSING", "OTHER"})
        X_train[column] = pd.Categorical(train_values, categories=categories)
        X_test[column] = pd.Categorical(test_values.where(test_values.isin(categories), "OTHER"),
                                        categories=categories)
    for column in numeric:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce").astype(np.float32)
        X_test[column] = pd.to_numeric(X_test[column], errors="coerce").astype(np.float32)
    return X_train, X_test


def cross_validate(data, features, columns, morgan, splits, keep_articles=False):
    y = pd.to_numeric(data["value"], errors="raise").to_numpy(float)
    fold_rows, oof_rows, shap_rows = [], [], []
    jobs = [(split, *f) for split in splits for f in evaluation_folds(data, split, keep_articles)]
    total, done, start = len(jobs), 0, perf_counter()
    for split, fold, train_idx, test_idx in jobs:
        X_train, X_test = prepare_fold(features, columns, morgan + COMMON_NUMERIC, train_idx, test_idx)
        model = LGBMRegressor(**PARAMS, random_state=42 + fold - 1 if isinstance(fold, int) else 42)
        t0 = perf_counter()
        model.fit(X_train, y[train_idx], categorical_feature=COMMON_CATEGORICAL)
        prediction = model.predict(X_test)
        fold_rows.append({"model": "LightGBM", "split": split, "fold": fold,
                          "n_train": len(train_idx), "n_test": len(test_idx),
                          "n_features": len(columns), "seconds": perf_counter() - t0,
                          **regression_metrics(y[test_idx], prediction)})
        rows = data.iloc[test_idx]
        oof_rows.append(pd.DataFrame({
            "row_id": rows["row_id"].to_numpy(), "split": split, "fold": fold,
            "domain": rows["domain"].to_numpy(), "metal": rows["metal"].to_numpy(),
            "doi": rows["doi"].to_numpy(), "observed": y[test_idx], "prediction": prediction,
            "absolute_error": np.abs(y[test_idx] - prediction)}))
        mean_abs = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(X_test))).mean(axis=0)
        shap_rows.append(pd.DataFrame({"split": split, "fold": fold, "feature": columns,
                                       "group": [feature_block(c) for c in columns],
                                       "mean_abs_shap": mean_abs}))
        done += 1
        eta = (perf_counter() - start) / done * (total - done)
        print(f"[{done:02d}/{total}] {split} fold {fold}: MAE={fold_rows[-1]['mae']:.4f}; ETA={eta:.0f}s")
    return pd.DataFrame(fold_rows), pd.concat(oof_rows, ignore_index=True), pd.concat(shap_rows, ignore_index=True)


def summarise(fold_metrics, oof, shap_folds):
    rows = []
    for split, group in oof.groupby("split", sort=False):
        f = fold_metrics[fold_metrics["split"].eq(split)]
        rows.append({"model": "LightGBM", "split": split, "n_oof": len(group),
                     **regression_metrics(group["observed"], group["prediction"]),
                     "fold_rmse_mean": f["rmse"].mean(), "fold_rmse_std": f["rmse"].std(ddof=1),
                     "fold_r2_mean": f["r2"].mean(), "fold_r2_std": f["r2"].std(ddof=1),
                     "fold_mae_mean": f["mae"].mean(), "fold_mae_std": f["mae"].std(ddof=1),
                     "seconds_total": f["seconds"].sum()})
    summary = pd.DataFrame(rows)
    print(summary.round(4).to_string(index=False))
    features = shap_folds.groupby(["split", "feature", "group"], as_index=False)["mean_abs_shap"].mean()
    groups = features.groupby(["split", "group"], as_index=False)["mean_abs_shap"].sum()
    groups["fraction"] = groups["mean_abs_shap"] / groups.groupby("split")["mean_abs_shap"].transform("sum")
    print(groups.sort_values(["split", "fraction"], ascending=[True, False]).round(4).to_string(index=False))
    return summary, features, groups


def plots(features, groups, tag):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (split, _) in zip(axes, SPLITS):
        g = groups[groups["split"].eq(split)].sort_values("fraction")
        ax.barh(g["group"], 100 * g["fraction"], color="#267365")
        ax.set(title=f"{split} split", xlabel="Share of total mean |SHAP| (%)")
    fig.suptitle("LightGBM feature-block importance", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"lightgbm{tag}_shap_groups.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, (split, _) in zip(axes, SPLITS):
        g = features[features["split"].eq(split) & ~features["feature"].str.startswith("MorganBit_")]
        g = g.nlargest(20, "mean_abs_shap").sort_values("mean_abs_shap")
        ax.barh(g["feature"], g["mean_abs_shap"], color="#C65D3B")
        ax.set(title=f"{split} split", xlabel="Mean |SHAP|")
    fig.suptitle("Top interpretable LightGBM features", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"lightgbm{tag}_shap_top_features.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprint", choices=FINGERPRINT_COLUMN, default="ligands")
    parser.add_argument("--keep-articles", action="store_true",
                        help="LOMO/LOGO without removing the articles of the held-out rows (outputs tagged _kept)")
    parser.add_argument("--splits", nargs="+", default=[s for s, _ in SPLITS] + METAL_SPLITS,
                        choices=[s for s, _ in SPLITS] + METAL_SPLITS)
    args = parser.parse_args()
    tag = "" if args.fingerprint == "ligands" else f"_{args.fingerprint}"
    if set(args.splits) != {s for s, _ in SPLITS} | set(METAL_SPLITS):
        tag += "_" + "_".join(args.splits)  # a subset never overwrites the full run
    if args.keep_articles:
        tag += "_kept"
    environment(LightGBM=lightgbm.__version__, SHAP=shap.__version__)
    data = load_modeling_data()
    print("Rows:", len(data))
    print("Shared numeric/categorical:", len(COMMON_NUMERIC), len(COMMON_CATEGORICAL))
    print("Fingerprint:", FINGERPRINT_COLUMN[args.fingerprint])
    features, columns, morgan = build_features(data, FINGERPRINT_COLUMN[args.fingerprint])
    fold_metrics, oof, shap_folds = cross_validate(data, features, columns, morgan, args.splits, args.keep_articles)
    fold_metrics.to_csv(METRICS_DIR / f"lightgbm{tag}_fold_metrics.csv", index=False)
    oof.to_csv(OOF_DIR / f"lightgbm{tag}_oof.csv", index=False)
    shap_folds.to_csv(METRICS_DIR / f"lightgbm{tag}_shap_by_fold.csv", index=False)
    summary, features_imp, groups = summarise(fold_metrics, oof, shap_folds)
    summary.to_csv(METRICS_DIR / f"lightgbm{tag}_summary.csv", index=False)
    features_imp.to_csv(METRICS_DIR / f"lightgbm{tag}_shap_features.csv", index=False)
    groups.to_csv(METRICS_DIR / f"lightgbm{tag}_shap_groups.csv", index=False)
    plots(features_imp, groups, tag)


if __name__ == "__main__":
    main()
