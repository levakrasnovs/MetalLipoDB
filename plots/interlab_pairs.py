"""
Inter-laboratory differences in MetalLipoDB as an error scale for models.

For every complex--counterion pair measured in more than one article, the
article means are compared pairwise. Reported:
  - mean |Δ| between two articles, first averaged per complex so that each
    complex has equal weight (cisplatin alone has 45 articles), with a
    bootstrap 95% CI over complexes: the expected MAE of an ideal model that
    predicts a measurement from an unseen study;
  - the same for pairs in the same and in different medium classes;
  - the paired test: complexes having both kinds of pairs, difference of the
    per-complex mean |Δ| (different minus same medium).

Usage:
    python interlab_pairs.py
"""

import itertools

from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 5000


def article_pairs(data):
    data = data[data["value_relation"].isin(["=", "~"])].copy()
    data["ci"] = data["counterion"].fillna("none")
    articles = data.groupby(["smiles_complex", "ci", "doi"]).agg(
        v=("value", "mean"), med=("medium_class", "first")).reset_index()
    rows = []
    for key, group in articles.groupby(["smiles_complex", "ci"]):
        for (_, p), (_, q) in itertools.combinations(group.iterrows(), 2):
            rows.append((key, abs(p.v - q.v), p.med == q.med and pd.notna(p.med)))
    return pd.DataFrame(rows, columns=["key", "d", "same_medium"])


def bootstrap_mean(values, rng):
    boot = [rng.choice(values, len(values)).mean() for _ in range(N_BOOT)]
    return values.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main():
    pairs = article_pairs(pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "MetalLipoDB.csv",
                                    low_memory=False))
    rng = np.random.default_rng(0)
    for label, sel in [("all pairs", pairs["d"].notna()), ("same medium class", pairs["same_medium"]),
                       ("different medium class", ~pairs["same_medium"])]:
        per_complex = pairs[sel].groupby("key")["d"].mean().to_numpy()
        mean, lo, hi = bootstrap_mean(per_complex, rng)
        print(f"{label}: {sel.sum()} pairs, {len(per_complex)} complexes, "
              f"mean |Δ| {mean:.2f} (95% CI {lo:.2f}-{hi:.2f})")
    both = pairs.groupby(["key", "same_medium"])["d"].mean().unstack().dropna()
    diff = (both[False] - both[True]).to_numpy()
    mean, lo, hi = bootstrap_mean(diff, rng)
    print(f"paired: {len(both)} complexes with both kinds of pairs; same medium {both[True].mean():.2f}, "
          f"different {both[False].mean():.2f}; difference {mean:.2f} (95% CI {lo:.2f} to {hi:.2f}); "
          f"larger for different medium in {(diff > 0).sum()}/{len(diff)}")


if __name__ == "__main__":
    main()
