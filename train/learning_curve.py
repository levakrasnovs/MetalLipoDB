"""
MetalLipoDB modeling — learning_curve.py
========================================
How does the error of LightGBM depend on the amount of training data?

In every outer fold of the frozen SMILES and DOI splits, LightGBM is trained on
a random subset of the training articles (whole articles are added until the
target size is reached, so a smaller set means fewer series rather than
thinned series) and evaluated on the unchanged test fold. N_REPEATS random
subsets per size; the full training set once. Features and hyperparameters
are those of train_lightgbm.py.

Output: results/metrics/learning_curve.csv (one row per split, fold, size, repeat)

Usage:
    python learning_curve.py
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import train_lightgbm as lgb
from common import METRICS_DIR, evaluation_folds, load_modeling_data
from features import COMMON_CATEGORICAL, COMMON_NUMERIC

SIZES = [250, 500, 1000, 2000, 3000, None]  # None: the full training set
N_REPEATS, SEED = 3, 0


def article_subset(train_idx, doi, n, rng):
    counts = pd.Series(doi[train_idx]).value_counts()
    keep, total = [], 0
    for article in rng.permutation(counts.index.to_numpy()):
        keep.append(article)
        total += counts[article]
        if total >= n:
            break
    return train_idx[np.isin(doi[train_idx], keep)]


def main():
    data = load_modeling_data()
    y = pd.to_numeric(data["value"], errors="raise").to_numpy(float)
    doi = data["doi"].str.lower().fillna(data["row_id"]).to_numpy()
    features, columns, morgan = lgb.build_features(data, "canonical_ligands")
    rng, rows = np.random.default_rng(SEED), []
    for split in ["SMILES", "DOI"]:
        for fold, train_idx, test_idx in evaluation_folds(data, split):
            for n in SIZES:
                for rep in range(N_REPEATS if n else 1):
                    sub = article_subset(train_idx, doi, n, rng) if n else train_idx
                    X_train, X_test = lgb.prepare_fold(features, columns, morgan + COMMON_NUMERIC, sub, test_idx)
                    model = LGBMRegressor(**lgb.PARAMS, random_state=rep)
                    model.fit(X_train, y[sub], categorical_feature=COMMON_CATEGORICAL)
                    rows.append({"split": split, "fold": fold, "target": n or "full", "repeat": rep,
                                 "n_train": len(sub), "mae": np.abs(model.predict(X_test) - y[test_idx]).mean()})
        print(split, "done")
    result = pd.DataFrame(rows)
    result.to_csv(METRICS_DIR / "learning_curve.csv", index=False)
    print(result.groupby(["split", "target"], sort=False)[["n_train", "mae"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
