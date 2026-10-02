"""
MetalLipoDB — plot_si_years.py
Supporting figure: number of articles and of entries per publication year.
Output: figures/figure_years.png / .pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from plot_figures_metallipodb import C_MAIN, RCPARAMS_STANDARD, load_lipo, save

ROOT = Path(__file__).resolve().parent.parent  # repository root


def main():
    plt.rcParams.update(RCPARAMS_STANDARD)
    d = load_lipo()
    years = range(int(d['year'].min()), int(d['year'].max()) + 1)
    art = d.drop_duplicates('doi')['year'].value_counts().reindex(years, fill_value=0)
    ent = d['year'].value_counts().reindex(years, fill_value=0)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6))
    for ax, s, lab in zip(axs, [art, ent], ['Number of articles', 'Number of entries']):
        ax.bar(s.index, s.values, color=C_MAIN, width=0.8)
        ax.set(xlabel='Year of publication', ylabel=lab)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for k, ax in enumerate(axs):
        ax.text(-0.02, 1.05, 'ab'[k] + ')', transform=ax.transAxes, fontsize=10, ha='right')
    plt.tight_layout()
    save(fig, ROOT / 'figures', 'figure_years')
    print(f'{art.sum()} articles, {ent.sum()} entries, {years.start}-{years.stop - 1}; '
          f'since 2015: {art[art.index >= 2015].sum() / art.sum():.0%} of articles')


if __name__ == '__main__':
    main()
