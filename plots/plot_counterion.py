"""
MetalLipoDB — plot_counterion.py
================================
Same complex, same article, same aqueous medium, different counterion: how far
does log P move? One row per such group, every counterion shown as a marker.
Supporting figure for the record-identity convention (complex + counterion).

Usage:
    python plot_counterion.py [--data MetalLipoDB.csv] [--out_dir figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA = ROOT / 'data' / 'MetalLipoDB.csv'
OUT_DIR = ROOT / 'figures'

RC = {
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 0.6, 'xtick.labelsize': 8, 'ytick.labelsize': 7,
    'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
}

ION_NAMES = {
    '[Cl-]': 'Cl⁻', '[Br-]': 'Br⁻', '[I-]': 'I⁻',
    'F[P-](F)(F)(F)(F)F': 'PF₆⁻', 'F[B-](F)(F)F': 'BF₄⁻',
    '[F][Sb-]([F])([F])([F])([F])[F]': 'SbF₆⁻',
    'O=S(=O)([O-])C(F)(F)F': 'OTf⁻', 'O=[N+]([O-])[O-]': 'NO₃⁻',
    'O=C([O-])C(F)(F)F': 'TFA⁻', '[O-][Cl+3]([O-])([O-])[O-]': 'ClO₄⁻',
    '[K+]': 'K⁺', '[Cs+]': 'Cs⁺', '[Na+]': 'Na⁺',
}
# one style per ion, in a fixed order so the same ion looks the same everywhere
ION_STYLE = {
    'Cl⁻': ('o', '#2E6E8E'), 'PF₆⁻': ('s', '#C1442F'), 'OTf⁻': ('D', '#4A9B6F'),
    'NO₃⁻': ('^', '#D08C34'), 'ClO₄⁻': ('v', '#7B5EA7'), 'BF₄⁻': ('P', '#8C564B'),
    'SbF₆⁻': ('X', '#E377C2'), 'Br⁻': ('<', '#17BECF'), 'I⁻': ('>', '#BCBD22'),
    'TFA⁻': ('h', '#1F77B4'), 'Na⁺': ('p', '#9467BD'), 'K⁺': ('*', '#FF7F0E'),
    'Cs⁺': ('d', '#2CA02C'), 'organic anion': ('o', '#999999'),
    'organic cation': ('s', '#999999'),
}


def ion_label(smiles):
    """Name of the counterion without stoichiometry: 2 PF6- and PF6- are the
    same ion for this comparison. Organic ions are pooled."""
    names = set()
    for part in smiles.split('.'):
        if part in ION_NAMES:
            names.add(ION_NAMES[part])
        else:
            names.add('organic cation' if '+]' in part and '-]' not in part else 'organic anion')
    return ' + '.join(sorted(names))


def groups(path=DATA):
    d = pd.read_csv(path)
    d = d[d['value_relation'].isin(['=', '~']) & d['counterion'].notna()].copy()
    g = (d.groupby(['doi', 'smiles_complex', 'medium_class', 'counterion'], dropna=False)
          ['value'].mean().reset_index())
    keep = g.groupby(['doi', 'smiles_complex', 'medium_class'], dropna=False)['counterion'] \
            .transform('nunique') > 1
    g = g[keep].copy()
    g['ion'] = g['counterion'].map(ion_label)
    g['key'] = g['doi'] + '|' + g['smiles_complex'] + '|' + g['medium_class'].fillna('')
    return g


def figure(g, out_dir):
    plt.rcParams.update(RC)
    rng = g.groupby('key')['value'].agg(lambda v: v.max() - v.min())
    order = rng.sort_values().index
    # rows of one article share a label; the article is named once per block
    ions = [i for i in ION_STYLE if i in set(g['ion'])]
    style = ION_STYLE

    fig, ax = plt.subplots(figsize=(7.2, 0.24 * len(order) + 1.4))
    for y, key in enumerate(order):
        r = g[g['key'] == key]
        ax.plot([r['value'].min(), r['value'].max()], [y, y], color='#cccccc', lw=1.4, zorder=1)
        for _, row in r.iterrows():
            m, c = style[row['ion']]
            ax.scatter(row['value'], y, marker=m, s=30, color=c, zorder=3, lw=0)
        ax.text(1.01, y, f'{rng[key]:.2f}', transform=ax.get_yaxis_transform(),
                va='center', fontsize=7, color='#555555')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([k.split('|')[0] for k in order])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel('logP')
    ax.text(1.01, len(order) - 0.2, 'range', transform=ax.get_yaxis_transform(),
            fontsize=7, color='#555555')
    for ion in ions:
        m, c = style[ion]
        ax.scatter([], [], marker=m, s=30, color=c, lw=0, label=ion)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.45, -0.06 - 1.2 / len(order)),
              ncol=min(len(ions), 7))
    ax.grid(axis='x', lw=0.3, color='#dddddd')
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    out = Path(out_dir) / 'figure_counterion_si'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)

    boot = np.median(np.random.default_rng(0).choice(rng.values, (5000, len(rng))), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f'groups {len(rng)} from {g["doi"].nunique()} articles | median range '
          f'{rng.median():.2f} (95% CI {lo:.2f}-{hi:.2f})')
    print(f'saved: {out}.png / .pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(DATA))
    p.add_argument('--out_dir', default=str(OUT_DIR))
    a = p.parse_args()
    figure(groups(a.data), a.out_dir)
