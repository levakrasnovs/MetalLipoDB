"""
MetalLipoDB modeling — compare_models.py
========================================
Collects the saved out-of-fold predictions of all models; nothing is trained.
Every model is reported in its single configuration fixed before training
(no "best within family" choice on the test folds).

  SMILES, DOI  mean ± SD over the 10 folds, as in the tables of the paper
  LOMO, LOGO   metrics over all held-out rows (fold sizes differ 20-2000 rows)
  paired       per-fold difference from LightGBM on SMILES/DOI: mean, SD and
               the number of folds where the model is better

Missing or stale results (another version of the modeling table, e.g. TabPFN\nbefore its rerun) are skipped with a note.

Outputs (results/): metrics/model_comparison.csv, metrics/model_comparison_paired.csv,
metrics/model_comparison_rows.tex (table rows for the paper),
figures/model_comparison.png

Usage:
    python compare_models.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURE_DIR, METAL_SPLITS, METRICS_DIR, OOF_DIR, SPLITS, load_modeling_data, regression_metrics

CV_SPLITS = [s for s, _ in SPLITS]
# table label -> (OOF file, filter on its "model" column or None)
MODELS = {
    "Baseline (mean logP per metal)": ("baselines_oof.csv", "Per-metal mean"),
    "Crippen logP (neutral ligands)": ("baselines_oof.csv", "Crippen logP (neutral ligands)"),
    "kNN ($k$ = 5, count Tanimoto)": ("baselines_oof.csv", "kNN (k = 5, count Tanimoto)"),
    "Ridge (descriptors)": ("baselines_oof.csv", "Ridge (descriptors)"),
    "Chemprop (DATIVE)": ("chemprop_oof.csv", None),
    "CheMeleon (frozen, DATIVE)": ("chemeleon_oof.csv", None),
    "TabPFN": ("tabpfn_oof.csv", None),
    "LightGBM": ("lightgbm_oof.csv", None),
}
REFERENCE = "LightGBM"


def load():
    current = set(load_modeling_data()["row_id"])
    frames = []
    for label, (file, model) in MODELS.items():
        path = OOF_DIR / file
        if not path.exists():
            print(f"skipped {label}: {file} not found")
            continue
        oof = pd.read_csv(path)
        if model is not None:
            oof = oof[oof["model"].eq(model)]
        if "representation" in oof:
            oof = oof[oof["representation"].eq("DATIVE")]
        # stale results (older table, several point types per row) are not comparable
        if not set(oof["row_id"]) <= current or oof.duplicated(["row_id", "split"]).any():
            print(f"skipped {label}: {file} is from another version of the modeling table; rerun the model")
            continue
        frames.append(oof[["row_id", "split", "fold", "observed", "prediction"]].assign(label=label))
    oof = pd.concat(frames, ignore_index=True)
    oof["fold"] = oof["fold"].astype(str)
    return oof


def summary(oof):
    rows = []
    for (label, split), g in oof.groupby(["label", "split"], sort=False):
        folds = pd.DataFrame([regression_metrics(f["observed"], f["prediction"])
                              for _, f in g.groupby("fold")])
        rows.append({"model": label, "split": split, "n": len(g), "n_folds": len(folds),
                     **{f"pooled_{k}": v for k, v in regression_metrics(g["observed"], g["prediction"]).items()},
                     **{f"fold_{m}_{a}": getattr(folds[m], a)() for m in ["mae", "rmse", "r2"]
                        for a in ["mean", "std"]}})
    return pd.DataFrame(rows)


def paired(oof):
    mae = (oof[oof["split"].isin(CV_SPLITS)]
           .groupby(["label", "split", "fold"])
           .apply(lambda f: np.abs(f["observed"] - f["prediction"]).mean(), include_groups=False)
           .rename("mae").reset_index())
    ref = mae[mae["label"].eq(REFERENCE)].set_index(["split", "fold"])["mae"]
    rows = []
    for (label, split), g in mae[~mae["label"].eq(REFERENCE)].groupby(["label", "split"], sort=False):
        d = g.set_index(["split", "fold"])["mae"] - ref.loc[g.set_index(["split", "fold"]).index]
        rows.append({"model": label, "split": split, "delta_mae_mean": d.mean(), "delta_mae_std": d.std(ddof=1),
                     "better_folds": int((d < 0).sum()), "n_folds": len(d)})
    return pd.DataFrame(rows)


def latex_rows(table):
    f3, f2 = (lambda x: f"{x:.3f}"), (lambda x: f"{x:.2f}".replace("-", "$-$"))
    lines = ["% SMILES / DOI (mean $\\pm$ SD over folds)"]
    for label in table["model"].unique():
        t = table[table["model"].eq(label)].set_index("split")
        if set(CV_SPLITS) <= set(t.index):
            lines.append(f"    {label} & " + " & ".join(
                f"{f3(t.loc[s, 'fold_mae_mean'])} $\\pm$ {f3(t.loc[s, 'fold_mae_std'])} & "
                f"{f3(t.loc[s, 'fold_rmse_mean'])} $\\pm$ {f3(t.loc[s, 'fold_rmse_std'])} & "
                f"{f2(t.loc[s, 'fold_r2_mean'])} $\\pm$ {f2(t.loc[s, 'fold_r2_std'])}" for s in CV_SPLITS) + " \\\\")
    lines.append("% LOMO / LOGO (over all held-out rows)")
    for label in table["model"].unique():
        t = table[table["model"].eq(label)].set_index("split")
        if set(METAL_SPLITS) <= set(t.index):
            lines.append(f"    {label} & " + " & ".join(
                f"{f3(t.loc[s, 'pooled_mae'])} & {f3(t.loc[s, 'pooled_rmse'])} & {f2(t.loc[s, 'pooled_r2'])}"
                for s in METAL_SPLITS) + " \\\\")
    return "\n".join(lines) + "\n"


def plot(table):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)
    order = [m for m in MODELS if m in set(table["model"])]
    for ax, split in zip(axes, CV_SPLITS + METAL_SPLITS):
        t = table[table["split"].eq(split)].set_index("model").reindex(order[::-1])
        cv = split in CV_SPLITS
        ax.barh([m.replace("$", "") for m in t.index], t["fold_mae_mean" if cv else "pooled_mae"],
                xerr=t["fold_mae_std"] if cv else None,
                color=["#1F6E5A" if m == REFERENCE else "#9AA5A0" for m in t.index])
        ax.set(title=split, xlabel="MAE" + (" (mean ± SD over folds)" if cv else " (all held-out rows)"))
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    oof = load()
    table, diffs = summary(oof), paired(oof)
    print(table[["model", "split", "n", "fold_mae_mean", "fold_mae_std", "pooled_mae", "pooled_r2"]]
          .round(3).to_string(index=False))
    print("\nPaired difference from LightGBM (per-fold MAE):")
    print(diffs.round(3).to_string(index=False))
    table.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    diffs.to_csv(METRICS_DIR / "model_comparison_paired.csv", index=False)
    (METRICS_DIR / "model_comparison_rows.tex").write_text(latex_rows(table))
    plot(table)


if __name__ == "__main__":
    main()
