"""
MetalLipoDB — plot_parity.py
============================
Predicted versus measured logP (out-of-fold predictions) for the trained
models in SMILES and DOI splitting, as recommended by Ash et al. (2025) for
the Supporting Information. Hexbin density, y = x line, MAE and R² of the
pooled out-of-fold predictions.

Usage:
    python plot_parity.py [--out_dir figures]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # repository root
sys.path.insert(0, str(ROOT / 'train'))
from compare_models import load  # noqa: E402
from common import regression_metrics  # noqa: E402
from plot_ml_practical import RC  # noqa: E402

OUT_DIR = str(ROOT / 'figures')
MODELS = [('TabPFN', 'TabPFN 3.5'), ('LightGBM', 'LightGBM'),
          ('Chemprop (DATIVE)', 'Chemprop'), ('CheMeleon (frozen, DATIVE)', 'CheMeleon')]
LIM = (-4, 5.5)


def figure(out_dir):
    plt.rcParams.update(RC)
    oof = load()
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.0), sharex=True, sharey=True)
    for i, split in enumerate(['SMILES', 'DOI']):
        for j, (label, title) in enumerate(MODELS):
            ax = axs[i, j]
            g = oof[oof['label'].eq(label) & oof['split'].eq(split)]
            ax.hexbin(g['observed'], g['prediction'], gridsize=40, extent=(*LIM, *LIM), cmap='Blues',
                      mincnt=1, bins='log', linewidths=0)
            ax.plot(LIM, LIM, color='#C1442F', lw=0.8, ls='--')
            m = regression_metrics(g['observed'], g['prediction'])
            ax.text(0.04, 0.96, f"MAE {m['mae']:.2f}\n$R^2$ {m['r2']:.2f}", transform=ax.transAxes,
                    va='top', fontsize=7)
            ax.set(xlim=LIM, ylim=LIM, aspect='equal')
            if i == 0:
                ax.set_title(title, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f'{split} splitting\npredicted logP')
            if i == 1:
                ax.set_xlabel('measured logP')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = Path(out_dir) / 'figure_parity'
    fig.savefig(out.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out}.png / .pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=OUT_DIR)
    figure(p.parse_args().out_dir)
