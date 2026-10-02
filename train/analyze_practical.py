"""
MetalLipoDB modeling — analyze_practical.py
===========================================
What the DOI-split predictions give an experimentalist. Uses saved
out-of-fold predictions only (lightgbm_oof.csv, baselines_oof.csv and the
similarities of lightgbm_domain.csv from analyze_domain.py); nothing is trained.

1. Ordering within an article: share of correctly ordered pairs of complexes
   of the same article (ties in the prediction count 0.5), for all pairs and
   for pairs differing by more than 0.5 and 1.0; median Spearman rho over
   articles with at least 4 rows.
2. Anchoring: k complexes of a series (k = 0-3) are taken as measured and the
   mean residual on them is added to the predictions of the rest. In each of
   N_REPEATS repeats one random order of the rows of every article (>= 4 rows)
   is drawn and shared by all k and all models: the anchors are its first k
   rows (nested, 1 within 2 within 3) and the evaluated rows are all rows after
   the third, so every k and model is evaluated on the same complexes. The "shift + slope"
   variant also rescales the predicted differences by the within-article slope
   of observed on predicted, estimated on the other folds.
0. Article offset (LightGBM): share of the residual variance shared by all rows
   of an article (articles with >= 3 rows), i.e. the intraclass correlation of
   the residuals. The variance between articles is estimated by the method of
   moments, subtracting the part of the variance of the article means due to the
   noise within articles (mean of s_i^2 / n_i); a random-intercept model
   (statsmodels MixedLM, REML) is fitted as a check, slopes of predicted on observed
   between article means and within articles, and the within-article
   calibration slope (observed differences on predicted ones).
3. Empirical prediction intervals (LightGBM): quantiles of the absolute
   out-of-fold residuals of the other folds, per similarity bin, 80% and 90%;
   coverage and half-width. The residuals of the other folds come from models
   whose training data included the current fold, so the intervals carry no
   formal conformal guarantee: their coverage is checked empirically.

Outputs (results/metrics/): practical_offset.csv, practical_ordering.csv, practical_anchoring.csv,
practical_intervals.csv

Usage:
    python analyze_practical.py
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from common import METRICS_DIR, OOF_DIR, load_modeling_data

warnings.filterwarnings("ignore", category=RuntimeWarning)  # spearmanr on tied predictions

MODELS = ["LightGBM", "Per-metal mean", "Crippen logP (neutral ligands)",
          "kNN (k = 5, count Tanimoto)", "Ridge (descriptors)"]
K_ANCHORS = [0, 1, 2, 3]
N_REPEATS = 30
SIM_BINS = [0, 0.5, 0.7, 0.9, 1.0001]
SIM_LABELS = ["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"]


def doi_predictions():
    doi = load_modeling_data()[["row_id", "doi"]]
    lgb = pd.read_csv(OOF_DIR / "lightgbm_oof.csv").assign(model="LightGBM")
    base = pd.read_csv(OOF_DIR / "baselines_oof.csv").merge(doi, on="row_id")
    cols = ["row_id", "model", "doi", "fold", "observed", "prediction"]
    pred = pd.concat([lgb[lgb["split"].eq("DOI")][cols], base[base["split"].eq("DOI")][cols]])
    pred["doi"] = pred["doi"].str.lower()
    pred["fold"] = pred["fold"].astype(int)
    return pred


def article_offset(pred):
    g = pred[pred["model"].eq("LightGBM")].copy()
    g = g[g.groupby("doi")["observed"].transform("size") >= 3]
    g["res"] = g["observed"] - g["prediction"]
    by = g.groupby("doi")
    n, var_i = by["res"].size(), by["res"].var()
    within = var_i.mean()
    # the variance of the article means includes the noise within articles (s_i^2 / n_i)
    between = max(by["res"].mean().var() - (var_i / n).mean(), 0.0)
    mixed = smf.mixedlm("res ~ 1", g, groups=g["doi"]).fit(reml=True)
    between_lmm, within_lmm = float(mixed.cov_re.iloc[0, 0]), float(mixed.scale)
    means = by[["observed", "prediction"]].mean()
    x = g["observed"] - by["observed"].transform("mean")
    y = g["prediction"] - by["prediction"].transform("mean")
    return pd.DataFrame([{"articles": g["doi"].nunique(), "rows": len(g),
                          "sd_offset": between ** 0.5, "sd_within": within ** 0.5,
                          "offset_share": between / (between + within),
                          "offset_share_mixedlm": between_lmm / (between_lmm + within_lmm),
                          "slope_between_articles": np.polyfit(means["observed"], means["prediction"], 1)[0],
                          "slope_within_articles": (x * y).sum() / (x * x).sum(),
                          # calibration: observed differences on predicted ones
                          "calibration_within_articles": (x * y).sum() / (y * y).sum()}])


def ordering(pred):
    rows = []
    for model in MODELS:
        g = pred[pred["model"].eq(model)]
        hits, pairs, rhos = {0: 0.0, 0.5: 0.0, 1.0: 0.0}, {0: 0, 0.5: 0, 1.0: 0}, []
        for _, a in g.groupby("doi"):
            if len(a) < 2:
                continue
            obs, prd = a["observed"].to_numpy(), a["prediction"].to_numpy()
            i, j = np.triu_indices(len(a), 1)
            d_obs, d_prd = obs[i] - obs[j], prd[i] - prd[j]
            for t in hits:
                sel = np.abs(d_obs) > t
                hits[t] += ((np.sign(d_obs) == np.sign(d_prd)) & (d_prd != 0) & sel).sum() \
                    + 0.5 * ((d_prd == 0) & sel).sum()
                pairs[t] += sel.sum()
            if len(a) >= 4 and obs.std() > 0:
                rhos.append(spearmanr(obs, prd)[0] if prd.std() > 0 else 0.0)
        rows.append({"model": model, **{f"concordance_gt{t}": hits[t] / pairs[t] for t in hits},
                     **{f"pairs_gt{t}": pairs[t] for t in pairs},
                     "median_spearman": np.nanmedian(rhos), "articles": len(rhos)})
    return pd.DataFrame(rows)


def within_slope(g):
    g = g[g.groupby("doi")["observed"].transform("size") >= 2]
    x = g["prediction"] - g.groupby("doi")["prediction"].transform("mean")
    y = g["observed"] - g.groupby("doi")["observed"].transform("mean")
    return (x * y).sum() / (x * x).sum()


def anchoring(pred):
    rng = np.random.default_rng(0)
    dois = sorted(d for d, a in pred[pred["model"].eq("LightGBM")].groupby("doi") if len(a) >= 4)
    sizes = pred[pred["model"].eq("LightGBM")].groupby("doi").size()
    # one random order per repeat and article, shared by all k and all models
    orders = [{d: rng.permutation(sizes[d]) for d in dois} for _ in range(N_REPEATS)]
    rows = []
    for model in MODELS:
        g = pred[pred["model"].eq(model)].sort_values("row_id")
        slopes = {f: within_slope(g[g["fold"].ne(f)]) for f in range(1, 11)}
        articles = {d: a for d, a in g.groupby("doi") if d in sizes.index and len(a) >= 4}
        assert sorted(articles) == dois
        for k in K_ANCHORS:
            shift, scaled = [], []
            for order in orders:
                e_shift, e_scaled = [], []
                for d, a in articles.items():
                    idx = order[d]
                    obs, prd = a["observed"].to_numpy(), a["prediction"].to_numpy()
                    ref, rest = idx[:k], idx[3:]
                    if k:
                        e_shift.append(np.abs(obs[rest] - prd[rest] - (obs[ref] - prd[ref]).mean()))
                        s = slopes[int(a["fold"].iloc[0])]
                        e_scaled.append(np.abs(obs[rest] - obs[ref].mean() - s * (prd[rest] - prd[ref].mean())))
                    else:
                        e_shift.append(np.abs(obs[rest] - prd[rest]))
                        e_scaled.append(e_shift[-1])
                shift.append(np.concatenate(e_shift).mean())
                scaled.append(np.concatenate(e_scaled).mean())
            rows.append({"model": model, "k": k, "mae_shift": np.mean(shift),
                         "mae_shift_slope": np.mean(scaled), "within_article_slope": np.mean(list(slopes.values())),
                         "rows_evaluated": sum(len(a) - 3 for a in articles.values()), "articles": len(articles)})
    return pd.DataFrame(rows)


def intervals(pred):
    sim = pd.read_csv(METRICS_DIR / "lightgbm_domain.csv")
    lgb = pred[pred["model"].eq("LightGBM")].merge(
        sim[sim["split"].eq("DOI")][["row_id", "max_sim"]], on="row_id", validate="one_to_one")
    lgb["abs_res"] = (lgb["observed"] - lgb["prediction"]).abs()
    lgb["bin"] = pd.cut(lgb["max_sim"], SIM_BINS, labels=SIM_LABELS)
    rows = []
    for level in [0.8, 0.9]:
        parts = []
        for f in range(1, 11):
            test, calib = lgb[lgb["fold"].eq(f)], lgb[lgb["fold"].ne(f)]
            q = calib.groupby("bin", observed=False)["abs_res"].quantile(level)
            half = test["bin"].map(q).astype(float)
            parts.append(test.assign(half=half, covered=test["abs_res"] <= half))
        t = pd.concat(parts)
        for b, g in [("all", t), *t.groupby("bin", observed=True)]:
            rows.append({"level": level, "similarity": b, "rows": len(g),
                         "coverage": g["covered"].mean(), "half_width": g["half"].mean()})
    return pd.DataFrame(rows)


def main():
    pred = doi_predictions()
    for name, table in [("offset", article_offset(pred)), ("ordering", ordering(pred)), ("anchoring", anchoring(pred)),
                        ("intervals", intervals(pred))]:
        print(f"\n== {name} ==\n" + table.round(3).to_string(index=False))
        table.to_csv(METRICS_DIR / f"practical_{name}.csv", index=False)


if __name__ == "__main__":
    main()
