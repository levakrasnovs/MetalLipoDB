"""
MetalLipoDB — plot_ml_practical.py
==================================
Why the error on unseen articles is what it is, and how to reduce it
(LightGBM, DOI splitting; saved out-of-fold predictions only).

  a) MAE against the similarity of the test complex to its training fold
     (count Tanimoto to the nearest training set of ligands)
  b) article offset (mean residual of an article) against the mean observed
     log P of the article: the model draws new series towards the mean
  c) MAE after anchoring to k measured complexes of the series, for LightGBM,
     the calibrated Crippen log P and the mean of the measured complexes

The grey band in a) and c) is the noise level of the data: the mean |Δ|/sqrt(2) of
between two articles reporting the same complex, with its bootstrap 95% CI
(interlab_pairs.py).

Needs train/results from train_lightgbm.py, baselines.py, analyze_domain.py and
analyze_practical.py.

Usage:
    python plot_ml_practical.py [--out_dir figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from interlab_pairs import article_pairs, bootstrap_mean

ROOT = Path(__file__).resolve().parent.parent  # repository root
RESULTS = ROOT / 'train' / 'results'
OUT_DIR = str(ROOT / 'figures')

RC = {
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 0.6, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
}
C_MAIN = '#2E6E8E'
C_ACC = '#C1442F'
C_GREY = '#999999'

SIM_BINS = [0, 0.3, 0.5, 0.7, 0.9, 1.0001]
SIM_LABELS = ['<0.3', '0.3–0.5', '0.5–0.7', '0.7–0.9', '≥0.9']
ANCHOR_MODELS = [('LightGBM', 'LightGBM', C_MAIN, 'o'),
                 ('Crippen logP (neutral ligands)', 'Crippen log P', C_ACC, 's'),
                 ('Per-metal mean', 'mean of measured', C_GREY, '^')]


def interlab_reference():
    """Noise level of the data: the expected MAE of an ideal model (the true value)
    against a single measurement. Two published values each carry their own error,
    so the mean |Δ| between two articles is divided by sqrt(2)."""
    pairs = article_pairs(pd.read_csv(ROOT / 'data' / 'MetalLipoDB.csv', low_memory=False))
    per_complex = pairs.groupby('key')['d'].mean().to_numpy()
    return tuple(v / np.sqrt(2) for v in bootstrap_mean(per_complex, np.random.default_rng(0)))


def doi_predictions():
    oof = pd.read_csv(RESULTS / 'oof_predictions' / 'lightgbm_oof.csv')
    return oof[oof['split'].eq('DOI')].copy()


def band(ax, ref, label=True):
    mean, lo, hi = ref
    ax.axhspan(lo, hi, color=C_GREY, alpha=0.22, lw=0)
    ax.axhline(mean, color=C_GREY, lw=0.9, ls='--',
               label=f'noise level ({mean:.2f})' if label else None)


def panel_similarity(ax, oof, ref):
    sim = pd.read_csv(RESULTS / 'metrics' / 'lightgbm_domain.csv')
    x = oof.merge(sim[sim['split'].eq('DOI')][['row_id', 'max_sim']], on='row_id', validate='one_to_one')
    x['bin'] = pd.cut(x['max_sim'], SIM_BINS, labels=SIM_LABELS, right=False)  # as in analyze_domain.py
    g = x.groupby('bin', observed=True)['absolute_error'].agg(['mean', 'size'])
    band(ax, ref)
    ax.plot(range(len(g)), g['mean'], color=C_MAIN, marker='o', ms=4.5, lw=1.3)
    for i, (m, n) in enumerate(zip(g['mean'], g['size'])):
        ax.text(i + 0.12, m + 0.03, str(n), ha='left', va='bottom', fontsize=6, color='#555555')
    ax.set_xticks(range(len(g)), g.index, fontsize=7)
    ax.set_xlabel('similarity to the training data')
    ax.set_ylabel('MAE')
    ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False, loc='lower left')
    print('a) MAE by similarity:', g['mean'].round(3).to_dict())


def panel_offset(ax, oof):
    x = oof.assign(res=oof['observed'] - oof['prediction'], doi=oof['doi'].str.lower())
    x = x[x.groupby('doi')['res'].transform('size') >= 3]
    a = x.groupby('doi').agg(obs=('observed', 'mean'), off=('res', 'mean'), n=('res', 'size'))
    ax.axhline(0, color=C_GREY, lw=0.6)
    ax.scatter(a['obs'], a['off'], s=np.clip(a['n'], 3, 30) * 0.8, color=C_MAIN, alpha=0.45, lw=0)
    k, b = np.polyfit(a['obs'], a['off'], 1)
    xx = np.array([a['obs'].quantile(0.01), a['obs'].quantile(0.99)])
    ax.plot(xx, b + k * xx, color=C_ACC, lw=1.3)
    ax.text(0.97, 0.03, f'{len(a)} articles', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5)
    ax.set_xlabel('mean log P of the article')
    ax.set_ylabel('mean error of the article\n(observed − predicted)')
    ax.set_ylim(-2.5, 2.5)
    print(f'b) {len(a)} articles, slope of offset on mean log P {k:.2f}')


def panel_anchoring(ax, ref):
    t = pd.read_csv(RESULTS / 'metrics' / 'practical_anchoring.csv')
    band(ax, ref, label=False)
    for model, label, color, marker in ANCHOR_MODELS:
        s = t[t['model'].eq(model)].sort_values('k')
        ax.plot(s['k'], s['mae_shift'], color=color, marker=marker, ms=4.5, lw=1.3, label=label)
        print(f'c) {label}:', dict(zip(s['k'], s['mae_shift'].round(3))))
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel('measured complexes, $k$')
    ax.set_ylabel('MAE')
    ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False, loc='upper right')


def figure(out_dir):
    plt.rcParams.update(RC)
    ref = interlab_reference()
    oof = doi_predictions()
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
    panel_similarity(axs[0], oof, ref)
    panel_offset(axs[1], oof)
    panel_anchoring(axs[2], ref)
    for k, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.02, 1.06, 'abc'[k] + ')', transform=ax.transAxes, fontsize=10, ha='right', va='bottom')
    plt.tight_layout()
    out = Path(out_dir) / 'figure_ml_practical'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved: {out}.png / .pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=OUT_DIR)
    figure(p.parse_args().out_dir)
