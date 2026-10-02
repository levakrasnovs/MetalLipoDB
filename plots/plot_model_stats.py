"""
MetalLipoDB — plot_model_stats.py
=================================
Statistical comparison of the models (train/stat_compare.py): mean of each
metric over the 10 folds with the comparison interval of the repeated-measures
Tukey HSD test (variation between folds removed, non-overlapping intervals mean
a significant difference),
in the style of Ash et al. (2025) and Burns et al. (2026). Blue: best model;
grey: not significantly different from it; red: significantly worse
(alpha = 0.05). Panels: MAE in SMILES and DOI splitting; the axis is reversed
so that better models are to the right.

Usage:
    python plot_model_stats.py [--out_dir figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from plot_ml_practical import RC

ROOT = Path(__file__).resolve().parent.parent  # repository root
METRICS = ROOT / 'train' / 'results' / 'metrics'
OUT_DIR = str(ROOT / 'figures')
COLORS = {'best': '#2E6E8E', 'equivalent': '#999999', 'worse': '#C1442F'}
LABELS = {'Baseline (mean logP per metal)': 'mean per metal',
          'Crippen logP (neutral ligands)': 'Crippen log P',
          'kNN ($k$ = 5, count Tanimoto)': 'kNN',
          'Chemprop (DATIVE)': 'Chemprop',
          'CheMeleon (frozen, DATIVE)': 'CheMeleon',
          'LightGBM': 'LightGBM',
          'TabPFN': 'TabPFN 3.5'}
ORDER = list(LABELS)  # baselines at the bottom, best models on top


def panel(ax, t, metric, split):
    t = t[t['split'].eq(split) & t['metric'].eq(metric)].set_index('model').reindex(ORDER).dropna()
    y = range(len(t))
    best = t[t['status'].eq('best')].iloc[0]
    ax.axvspan(best['mean'] - best['halfwidth'], best['mean'] + best['halfwidth'], color=COLORS['best'],
               alpha=0.08, lw=0)
    for k, (model, r) in enumerate(t.iterrows()):
        c = COLORS[r['status']]
        ax.errorbar(r['mean'], k, xerr=r['halfwidth'], fmt='o', color=c, ms=4.5, capsize=2.5, elinewidth=1.2)
    ax.set_yticks(list(y), [LABELS[m] for m in t.index])
    ax.set_xlabel({'mae': 'MAE', 'r2': '$R^2$'}[metric])
    if metric == 'mae':
        ax.invert_xaxis()  # better models to the right, as for R²
    ax.set_title(f'{split} splitting', fontsize=9, loc='left')
    ax.grid(axis='x', lw=0.3, color='#dddddd')


def figure(out_dir):
    plt.rcParams.update(RC)
    t = pd.read_csv(METRICS / 'stat_tukey_intervals.csv')
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.4), sharey=True)
    for j, split in enumerate(['SMILES', 'DOI']):
        panel(axs[j], t, 'mae', split)
    for k, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.02, 1.04, 'abcd'[k] + ')', transform=ax.transAxes, fontsize=10, ha='right', va='bottom')
    plt.tight_layout()  # the colour code is explained in the figure caption
    out = Path(out_dir) / 'figure_model_stats'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved: {out}.png / .pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=OUT_DIR)
    figure(p.parse_args().out_dir)
