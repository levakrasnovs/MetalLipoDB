"""
MetalLipoDB — plot_ml_metal.py
==============================
Models on metals (LOMO) and groups of metals (LOGO) absent from the training
data, in the style of plot_model_stats.py. A point is the MAE over all
held-out measurements (as in the LOMO/LOGO table); the interval is a 95%
bootstrap over the held-out metals or groups, which are resampled whole since
the measurements of one metal are not independent. Unlike the SMILES and DOI
panels there is no Tukey test: the 18 metals and 8 groups differ in size by
two orders of magnitude. The grey band is the noise level of the data.

Usage:
    python plot_ml_metal.py [--out_dir figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_ml_practical import C_GREY, RC, interlab_reference
from plot_model_stats import LABELS, ORDER

ROOT = Path(__file__).resolve().parent.parent  # repository root
OOF = ROOT / 'train' / 'results' / 'oof_predictions'
OUT_DIR = str(ROOT / 'figures')
SOURCES = {  # table name -> (file, filter on the model column of the baselines file)
    'Baseline (mean logP per metal)': ('baselines_oof.csv', 'Per-metal mean'),
    'Crippen logP (neutral ligands)': ('baselines_oof.csv', 'Crippen logP (neutral ligands)'),
    'kNN ($k$ = 5, count Tanimoto)': ('baselines_oof.csv', 'kNN (k = 5, count Tanimoto)'),
    'Chemprop (DATIVE)': ('chemprop_oof.csv', None),
    'CheMeleon (frozen, DATIVE)': ('chemeleon_oof.csv', None),
    'LightGBM': ('lightgbm_oof.csv', None),
    'TabPFN': ('tabpfn_oof.csv', None),
}
N_BOOT = 2000


def load(n_rows):
    """Held-out predictions of every model that is up to date (same rows as LightGBM)."""
    out = {}
    for name, (file, model) in SOURCES.items():
        path = OOF / file
        if not path.exists():
            continue
        o = pd.read_csv(path)
        if model:
            o = o[o['model'].eq(model)]
        o = o[o['split'].isin(['LOMO', 'LOGO'])]
        if len(o) != n_rows:   # trained on another version of the data: left out of the draft
            print(f'skipped {name}: {len(o)} held-out rows, expected {n_rows}')
            continue
        out[name] = o.assign(ae=(o['observed'] - o['prediction']).abs())
    return out


def pooled_ci(o, rng):
    by_fold = o.groupby('fold')['ae'].agg(['sum', 'size'])
    s, n = by_fold['sum'].to_numpy(), by_fold['size'].to_numpy()
    idx = rng.integers(0, len(s), (N_BOOT, len(s)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    return s.sum() / n.sum(), *np.percentile(boot, [2.5, 97.5])


def figure(out_dir):
    plt.rcParams.update(RC)
    ref_rows = pd.read_csv(OOF / 'lightgbm_oof.csv')
    models = load(int(ref_rows['split'].isin(['LOMO', 'LOGO']).sum()))
    ref = interlab_reference()
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.4), sharey=True)
    names = [m for m in ORDER if m in models]
    for ax, split in zip(axs, ['LOMO', 'LOGO']):
        rng = np.random.default_rng(0)
        stats = {m: pooled_ci(models[m][models[m]['split'].eq(split)], rng) for m in names}
        best = min(stats, key=lambda m: stats[m][0])
        ax.axvspan(ref[1], ref[2], color=C_GREY, alpha=0.22, lw=0)   # noise level, as in plot_ml_practical.py
        ax.axvline(ref[0], color=C_GREY, lw=0.9, ls='--')
        for k, m in enumerate(names):
            mae, lo, hi = stats[m]
            ax.errorbar(mae, k, xerr=[[mae - lo], [hi - mae]], fmt='o', ms=4.5, capsize=2.5, elinewidth=1.2,
                        color='#2E6E8E' if m == best else '#777777')
            print(f'{split} {LABELS[m]:15s} MAE {mae:.3f} ({lo:.3f}-{hi:.3f})')
        ax.set_yticks(range(len(names)), [LABELS[m] for m in names])
        ax.set_xlabel('MAE')
        ax.invert_xaxis()   # better models to the right, as in plot_model_stats.py
        n_folds = models[names[0]].query('split == @split')['fold'].nunique()
        ax.set_title(f'{split} ({n_folds} held-out {"metals" if split == "LOMO" else "groups"})',
                     fontsize=9, loc='left')
        ax.grid(axis='x', lw=0.3, color='#dddddd')
    for k, ax in enumerate(axs):
        ax.text(-0.02 if k else -0.32, 1.04, 'ab'[k] + ')', transform=ax.transAxes, fontsize=10, ha='right')
    plt.tight_layout()
    out = Path(out_dir) / 'figure_ml_metal'
    fig.savefig(out.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    print(f'saved: {out}.png / .pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=OUT_DIR)
    figure(ap.parse_args().out_dir)


if __name__ == '__main__':
    main()
