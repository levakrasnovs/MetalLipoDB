"""
MetalLipoDB modeling — stat_compare.py
======================================
Statistical comparison of the models on the frozen SMILES and DOI folds,
following Ash et al. (J. Chem. Inf. Model. 2025, 65, 9398) and their code
(github.com/polaris-hub/polaris-method-comparison, rm_tukey_hsd):

  1. repeated-measures ANOVA for each metric (subject = fold, factor = model),
     with the Bonferroni correction over the metrics (alpha / 3);
  2. the Tukey HSD test for all pairs of models with the residual mean square
     of the repeated-measures ANOVA, i.e. with the variation between folds
     removed, since all models are evaluated on the same folds (alpha = 0.05).

The interval of each model in stat_tukey_intervals.csv is a Tukey comparison
interval, mean +- q/2 * sqrt(MSE/n): two intervals do not overlap exactly when
the pair differs significantly. It is not a confidence interval of the mean. A plain Tukey HSD on the per-fold values (used before) ignores
the pairing and is much too conservative here, because the spread between folds
is larger than the differences between the models.

The per-fold metrics are recomputed from the saved out-of-fold predictions of
every model (compare_models.load), so all models are compared on the same 10
folds. Nothing is trained.

Outputs (results/metrics/): stat_anova.csv, stat_tukey_pairs.csv,
stat_tukey_intervals.csv (mean and interval half-width per model, and whether
the model differs from the best one)

Usage:
    python stat_compare.py
"""

import itertools

import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng

from common import METRICS_DIR, SPLITS, regression_metrics
from compare_models import load

ALPHA = 0.05
METRICS = {"mae": "min", "rmse": "min", "r2": "max"}  # which direction is better


def fold_metrics(oof):
    rows = []
    for (label, split, fold), g in oof[oof["split"].isin([s for s, _ in SPLITS])].groupby(["label", "split", "fold"]):
        rows.append({"model": label, "split": split, "fold": fold,
                     **{k: v for k, v in regression_metrics(g["observed"], g["prediction"]).items() if k in METRICS}})
    return pd.DataFrame(rows)


EXCLUDED = ["Ridge (descriptors)"]  # not reported in the paper


def rm_tukey(w):
    """Tukey HSD with the residual mean square of the repeated-measures ANOVA (w: folds x models)."""
    n, k = w.shape
    res = w - w.mean(axis=1).to_numpy()[:, None] - w.mean(axis=0).to_numpy()[None, :] + w.to_numpy().mean()
    df_res = (n - 1) * (k - 1)
    mse = (res.to_numpy() ** 2).sum() / df_res
    se = np.sqrt(mse / n)              # standard error of a model mean
    q = qsturng(1 - ALPHA, k, df_res)
    pairs = []
    for m1, m2 in itertools.combinations(w.columns, 2):
        diff = w[m1].mean() - w[m2].mean()
        p = float(np.atleast_1d(psturng(abs(diff) / se, k, df_res))[0])
        pairs.append({"model_1": m1, "model_2": m2, "meandiff": diff, "lower": diff - q * se,
                      "upper": diff + q * se, "p_adj": p, "reject": p < ALPHA,
                      "folds_model_1_higher": int((w[m1] > w[m2]).sum())})
    return pd.DataFrame(pairs), q * se / 2


def main():
    folds = fold_metrics(load())
    folds = folds[~folds["model"].isin(EXCLUDED)]
    anova_rows, pair_rows, interval_rows = [], [], []
    for split, _ in SPLITS:
        f = folds[folds["split"].eq(split)]
        for metric, better in METRICS.items():
            res = AnovaRM(f, depvar=metric, subject="fold", within=["model"]).fit().anova_table.iloc[0]
            anova_rows.append({"split": split, "metric": metric, "F": res["F Value"], "df_num": res["Num DF"],
                               "df_den": res["Den DF"], "p": res["Pr > F"],
                               "significant_bonferroni": res["Pr > F"] < ALPHA / len(METRICS)})
            w = f.pivot(index="fold", columns="model", values=metric)
            pairs, half = rm_tukey(w)
            pair_rows.append(pairs.assign(split=split, metric=metric))
            means = w.mean()
            best = means.idxmin() if better == "min" else means.idxmax()
            for model, mean in means.items():
                if model == best:
                    status = "best"
                else:
                    row = pairs[((pairs["model_1"] == model) & (pairs["model_2"] == best)) |
                                ((pairs["model_2"] == model) & (pairs["model_1"] == best))].iloc[0]
                    status = "worse" if row["reject"] else "equivalent"
                interval_rows.append({"split": split, "metric": metric, "model": model, "mean": mean,
                                      "halfwidth": half, "status": status})
    anova, pairs, intervals = pd.DataFrame(anova_rows), pd.concat(pair_rows), pd.DataFrame(interval_rows)
    print(anova.round(4).to_string(index=False))
    print(intervals[intervals["metric"].eq("mae")].sort_values(["split", "mean"]).round(3).to_string(index=False))
    anova.to_csv(METRICS_DIR / "stat_anova.csv", index=False)
    pairs.to_csv(METRICS_DIR / "stat_tukey_pairs.csv", index=False)
    intervals.to_csv(METRICS_DIR / "stat_tukey_intervals.csv", index=False)


if __name__ == "__main__":
    main()
