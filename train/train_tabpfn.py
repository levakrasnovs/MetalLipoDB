"""
MetalLipoDB modeling — train_tabpfn.py
======================================
TabPFN 3.5 regressor (model_path "v3.5_default") through the Prior Labs API (tabpfn_client) on the count
Morgan fingerprint of the ligand set (512 bits: TabPFN handles hundreds, not
thousands, of features) plus the shared descriptors of features.py, on
SMILES, DOI, LOMO and LOGO (common.evaluation_folds). No hyperparameters are
tuned; the reported point prediction is the median of the predictive
distribution, chosen before training. The 10% and 90% quantiles are stored
with it when the API returns them.

The API token is read from the TABPFN_TOKEN environment variable and is never
written anywhere. Finished (split, fold) jobs are kept and skipped on restart,
so an interrupted run costs no extra API calls; --fresh discards them.

Outputs (results/): metrics/tabpfn_{fold_metrics,summary}.csv,
oof_predictions/tabpfn_oof.csv

Usage:
    TABPFN_TOKEN=... python train_tabpfn.py [--splits SMILES DOI LOMO LOGO] [--fresh]
"""

import argparse
import contextlib
import io
import os
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from common import (METAL_SPLITS, METRICS_DIR, OOF_DIR, SPLITS, environment, evaluation_folds,
                    load_modeling_data, regression_metrics)
from features import COMMON_CATEGORICAL, COMMON_NUMERIC, ligand_count_fingerprint

MODEL = "TabPFN 3.5 (API)"
MODEL_PATH = "v3.5_default"
ALL_SPLITS = [s for s, _ in SPLITS] + METAL_SPLITS
FP_SIZE = 512
FOLD_METRICS_PATH = METRICS_DIR / "tabpfn_fold_metrics.csv"
OOF_PATH = OOF_DIR / "tabpfn_oof.csv"
N_ATTEMPTS = 3


def features(data):
    cache = {s: ligand_count_fingerprint(s, FP_SIZE) for s in data["canonical_ligands"].unique()}
    morgan = np.vstack(data["canonical_ligands"].map(cache)).astype(np.float32)
    numeric = data[COMMON_NUMERIC].apply(pd.to_numeric, errors="coerce")
    categorical = data[COMMON_CATEGORICAL].fillna("MISSING").astype(str)
    return morgan, numeric, categorical


def fold_matrices(morgan, numeric, categorical, train_idx, test_idx):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(numeric.iloc[train_idx])
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32) \
        .fit(categorical.iloc[train_idx])
    return [np.hstack([morgan[idx], imputer.transform(numeric.iloc[idx]),
                       onehot.transform(categorical.iloc[idx])]).astype(np.float32)
            for idx in (train_idx, test_idx)]


def predict(X_train, y_train, X_test):
    from tabpfn_client import TabPFNRegressor
    for attempt in range(1, N_ATTEMPTS + 1):
        try:
            model = TabPFNRegressor(
                model_path=MODEL_PATH,
                random_state=0,
                thinking_mode=False,
            )
            model.fit(X_train, y_train)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                output = model.predict(X_test, output_type="main")
            result = {"prediction": np.asarray(output["median"], dtype=float).reshape(-1)}
            quantiles = output.get("quantiles") if isinstance(output, dict) else None
            if quantiles is not None and len(quantiles) == 9:  # default levels 0.1, 0.2, ..., 0.9
                result["q10"] = np.asarray(quantiles[0], dtype=float).reshape(-1)
                result["q90"] = np.asarray(quantiles[-1], dtype=float).reshape(-1)
            return result
        except Exception as error:
            if attempt == N_ATTEMPTS:
                raise
            print(f"API retry {attempt}/{N_ATTEMPTS - 1} after {type(error).__name__}; "
                  f"waiting {10 * attempt} s", flush=True)
            time.sleep(10 * attempt)


def summarise(fold_metrics, oof):
    rows = []
    for split, group in oof.groupby("split", sort=False):
        f = fold_metrics[fold_metrics["split"].eq(split)]
        rows.append({"model": MODEL, "split": split, "n_oof": len(group),
                     **regression_metrics(group["observed"], group["prediction"]),
                     **{f"fold_{m}_{a}": getattr(f[m], a)() for m in ["mae", "rmse", "r2"]
                        for a in ["mean", "std"]},
                     "seconds_total": f["seconds"].sum()})
    summary = pd.DataFrame(rows)
    print(summary.round(4).to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=ALL_SPLITS, default=ALL_SPLITS)
    parser.add_argument("--fresh", action="store_true", help="discard finished folds")
    args = parser.parse_args()
    if not os.environ.get("TABPFN_TOKEN"):
        raise SystemExit("Set the TABPFN_TOKEN environment variable (Prior Labs API token).")
    import tabpfn_client
    environment(tabpfn_client=getattr(tabpfn_client, "__version__", "unknown"))
    print("API usage before:", tabpfn_client.get_api_usage())

    data = load_modeling_data()
    y = pd.to_numeric(data["value"], errors="raise").to_numpy(np.float32)
    morgan, numeric, categorical = features(data)
    print("Rows:", len(data), "| features:", FP_SIZE, "+", len(COMMON_NUMERIC), "+ one-hot of",
          len(COMMON_CATEGORICAL))

    if args.fresh or not (FOLD_METRICS_PATH.exists() and OOF_PATH.exists()):
        fold_rows, oof = [], pd.DataFrame()
    else:
        fold_rows, oof = pd.read_csv(FOLD_METRICS_PATH).to_dict("records"), pd.read_csv(OOF_PATH)
        assert set(oof["row_id"]) <= set(data["row_id"]) and {r["model"] for r in fold_rows} == {MODEL}, \
            f"{FOLD_METRICS_PATH.name} is from another dataset or model version; rerun with --fresh"
    done = {(r["split"], str(r["fold"])) for r in fold_rows}
    jobs = [(split, *f) for split in args.splits for f in evaluation_folds(data, split)
            if (split, str(f[0])) not in done]
    print(f"Jobs: {len(jobs)} to run, {len(done)} already finished")

    for n, (split, fold, train_idx, test_idx) in enumerate(jobs, 1):
        X_train, X_test = fold_matrices(morgan, numeric, categorical, train_idx, test_idx)
        started = time.perf_counter()
        result = predict(X_train, y[train_idx], X_test)
        fold_rows.append({"model": MODEL, "split": split, "fold": fold, "n_train": len(train_idx),
                          "n_test": len(test_idx), "n_features": X_train.shape[1],
                          "seconds": time.perf_counter() - started,
                          **regression_metrics(y[test_idx], result["prediction"])})
        rows = data.iloc[test_idx]
        oof = pd.concat([oof, pd.DataFrame({
            "row_id": rows["row_id"].to_numpy(), "split": split, "fold": fold,
            "domain": rows["domain"].to_numpy(), "metal": rows["metal"].to_numpy(),
            "doi": rows["doi"].to_numpy(), "observed": y[test_idx], **result,
            "absolute_error": np.abs(y[test_idx] - result["prediction"])})], ignore_index=True)
        pd.DataFrame(fold_rows).to_csv(FOLD_METRICS_PATH, index=False)
        oof.to_csv(OOF_PATH, index=False)
        print(f"[{n:02d}/{len(jobs)}] {split} fold {fold}: MAE={fold_rows[-1]['mae']:.4f}", flush=True)

    summarise(pd.DataFrame(fold_rows), oof).to_csv(METRICS_DIR / "tabpfn_summary.csv", index=False)
    print("API usage after:", tabpfn_client.get_api_usage())


if __name__ == "__main__":
    main()
