"""
MetalLipoDB modeling — time_split.py
====================================
Time-split validation of LightGBM (Sheridan, J. Chem. Inf. Model. 2013, 53,
783): the model is trained on the articles published before a given year and
predicts the articles of later years, simulating the prediction of complexes
reported after the model was built. The split is by article year, so no
article is shared between training and test.

  single    train on articles up to CUTOFF, test on the later ones (~25%)
  by year   for each test year Y >= FIRST_TEST_YEAR: train on the articles
            published before Y, test on the articles of year Y

The per-metal mean is given as the baseline, and for each test set the share
of rows whose complex (SMILES + counterion) was already measured in training.
Features and hyperparameters are those of train_lightgbm.py.

Outputs (results/): metrics/time_split_{single,by_year}.csv,
figures/time_split.png

Usage:
    python time_split.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import train_lightgbm as lgb
from common import FIGURE_DIR, METRICS_DIR, load_modeling_data, regression_metrics
from features import COMMON_CATEGORICAL, COMMON_NUMERIC

CUTOFF = 2023          # single split: train <= CUTOFF, test > CUTOFF
FIRST_TEST_YEAR = 2014


def fit_predict(features, columns, morgan, y, train_idx, test_idx):
    X_train, X_test = lgb.prepare_fold(features, columns, morgan + COMMON_NUMERIC, train_idx, test_idx)
    model = LGBMRegressor(**lgb.PARAMS, random_state=42)
    model.fit(X_train, y[train_idx], categorical_feature=COMMON_CATEGORICAL)
    return model.predict(X_test)


def per_metal(data, y, train_idx, test_idx):
    means = pd.Series(y[train_idx]).groupby(data["metal"].to_numpy()[train_idx]).mean()
    return data["metal"].iloc[test_idx].map(means).fillna(y[train_idx].mean()).to_numpy()


def evaluate(data, features, columns, morgan, y, train_idx, test_idx, label):
    key = (data["canonical_complex_dative"] + "|" + data["counterion"].fillna("")).to_numpy()
    seen = np.isin(key[test_idx], key[train_idx])
    pred = fit_predict(features, columns, morgan, y, train_idx, test_idx)
    base = per_metal(data, y, train_idx, test_idx)
    lg, bs = regression_metrics(y[test_idx], pred), regression_metrics(y[test_idx], base)
    return {"test": label, "n_train": len(train_idx), "n_test": len(test_idx),
            "test_articles": data["doi"].iloc[test_idx].nunique(),
            "complex_seen_in_training": seen.mean(),
            **{f"lgb_{k}": v for k, v in lg.items()}, **{f"base_{k}": v for k, v in bs.items()},
            "lgb_mae_unseen_complex": np.abs(y[test_idx] - pred)[~seen].mean()}


def plot(by_year, single, cv_doi):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.6,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.axhline(cv_doi, color="#999999", lw=0.9, ls="--", label=f"10-fold DOI splitting ({cv_doi:.2f})")
    ax.plot(by_year["test"], by_year["base_mae"], color="#999999", marker="^", ms=4, lw=1.1,
            label="per-metal mean")
    ax.plot(by_year["test"], by_year["lgb_mae"], color="#2E6E8E", marker="o", ms=4.5, lw=1.3,
            label="LightGBM")
    for x, m, n in zip(by_year["test"], by_year["lgb_mae"], by_year["n_test"]):
        ax.text(x, m - 0.035, str(n), ha="center", va="top", fontsize=5.8, color="#555555")
    ax.set(xlabel="test year (trained on all earlier articles)", ylabel="MAE", ylim=(0.5, 1.45))
    ax.set_xticks(by_year["test"], [str(x) for x in by_year["test"]], rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "time_split.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "time_split.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_modeling_data()
    y = pd.to_numeric(data["value"], errors="raise").to_numpy(float)
    year = data["year"].astype(int).to_numpy()
    features, columns, morgan = lgb.build_features(data, "canonical_ligands")

    single = pd.DataFrame([evaluate(data, features, columns, morgan, y, np.flatnonzero(year <= CUTOFF),
                                    np.flatnonzero(year > CUTOFF), f"{CUTOFF + 1}-{year.max()}")])
    rows = [evaluate(data, features, columns, morgan, y, np.flatnonzero(year < t), np.flatnonzero(year == t), t)
            for t in range(FIRST_TEST_YEAR, year.max() + 1) if (year == t).any()]
    by_year = pd.DataFrame(rows)
    cols = ["test", "n_train", "n_test", "test_articles", "complex_seen_in_training",
            "lgb_mae", "lgb_rmse", "lgb_r2", "base_mae", "lgb_mae_unseen_complex"]
    print("single split:\n" + single[cols].round(3).to_string(index=False))
    print("\nby test year:\n" + by_year[cols].round(3).to_string(index=False))
    single.to_csv(METRICS_DIR / "time_split_single.csv", index=False)
    by_year.to_csv(METRICS_DIR / "time_split_by_year.csv", index=False)
    cv = pd.read_csv(METRICS_DIR / "lightgbm_summary.csv").set_index("split").loc["DOI", "mae"]
    plot(by_year, single, cv)


if __name__ == "__main__":
    main()
