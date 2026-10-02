"""
MetalLipoDB — plot_time_learning.py
===================================
How LightGBM behaves on new data and how much data it needs.

  a) time-split validation: each year predicted by the model trained on all
     earlier articles (train/time_split.py), with the per-metal baseline and the
     MAE of 10-fold DOI splitting
  b) learning curve: MAE against the number of training measurements (subsets
     of whole articles, train/learning_curve.py) for DOI and SMILES splitting,
     mean ± SD over folds; grey band: noise level of the data, mean |Δ|/sqrt(2)
     (interlab_pairs.py)

Usage:
    python plot_time_learning.py [--out_dir figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_ml_practical import C_ACC, C_GREY, C_MAIN, RC, interlab_reference

ROOT = Path(__file__).resolve().parent.parent  # repository root
METRICS = ROOT / 'train' / 'results' / 'metrics'
OUT_DIR = str(ROOT / 'figures')


def panel_time(ax):
    t = pd.read_csv(METRICS / 'time_split_by_year.csv')
    cv = pd.read_csv(METRICS / 'lightgbm_summary.csv').set_index('split').loc['DOI', 'mae']
    ax.axhline(cv, color=C_GREY, lw=0.9, ls='--', label=f'10-fold DOI splitting ({cv:.2f})')
    ax.plot(t['test'], t['base_mae'], color=C_GREY, marker='^', ms=4, lw=1.1, label='per-metal mean')
    ax.plot(t['test'], t['lgb_mae'], color=C_MAIN, marker='o', ms=4.5, lw=1.3, label='LightGBM')
    for x, m, n in zip(t['test'], t['lgb_mae'], t['n_test']):
        ax.text(x, m - 0.035, str(n), ha='center', va='top', fontsize=5.5, color='#555555')
    ax.set_xticks(t['test'], [str(x) for x in t['test']], rotation=45)
    ax.set_xlabel('test year')
    ax.set_ylabel('MAE')
    ax.set_ylim(0.5, 1.45)
    ax.legend(frameon=False, loc='upper right')


def panel_learning(ax, ref):
    lc = pd.read_csv(METRICS / 'learning_curve.csv')
    mean, lo, hi = ref
    ax.axhspan(lo, hi, color=C_GREY, alpha=0.22, lw=0)
    ax.axhline(mean, color=C_GREY, lw=0.9, ls='--', label=f'noise level ({mean:.2f})')
    for split, color, marker in [('DOI', C_MAIN, 'o'), ('SMILES', C_ACC, 's')]:
        per_fold = lc[lc['split'].eq(split)].groupby(['target', 'fold'], sort=False) \
            .agg(n=('n_train', 'mean'), mae=('mae', 'mean')).reset_index()
        g = per_fold.groupby('target', sort=False).agg(n=('n', 'mean'), mae=('mae', 'mean'), sd=('mae', 'std'))
        ax.errorbar(g['n'], g['mae'], yerr=g['sd'], color=color, marker=marker, ms=4.5, lw=1.3,
                    capsize=2, elinewidth=0.8, label=f'{split} splitting')
        print(f'b) {split}:', dict(zip(g['n'].round(0).astype(int), g['mae'].round(3))))
    ax.set_xscale('log')
    ax.set_xticks([250, 500, 1000, 2000, 4600], ['250', '500', '1000', '2000', '4600'])
    ax.minorticks_off()
    ax.set_xlabel('training measurements')
    ax.set_ylabel('MAE')
    ax.set_ylim(0.3, 1.05)
    ax.legend(frameon=False, loc='lower left', fontsize=7)


def figure(out_dir):
    plt.rcParams.update(RC)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
    panel_time(axs[0])
    panel_learning(axs[1], interlab_reference())
    for k, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.02, 1.04, 'ab'[k] + ')', transform=ax.transAxes, fontsize=10, ha='right', va='bottom')
    plt.tight_layout()
    out = Path(out_dir) / 'figure_time_learning'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved: {out}.png / .pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=OUT_DIR)
    figure(p.parse_args().out_dir)
