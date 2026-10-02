"""
MetalLipoDB — plot_conditions.py
================================
SI figure: effect of the aqueous phase on log P within articles (the laboratory
and the protocol are shared). Pairs of conditions are taken from
analyze_conditions.py (they differ in one factor only).

  a) chloride (about 0-4 mM vs 100-154 mM, same buffer and pH): shift per
     complex, by charge and by the presence of coordinated chloride
  b) log P against pH in the articles that varied pH (buffer shown by marker)
  c) log P against chloride concentration in the articles that varied it

Usage:
    python plot_conditions.py [--out_dir figures]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_conditions import conditions, pairs
from plot_counterion import RC

ROOT = Path(__file__).resolve().parent.parent  # repository root
PH_ARTICLES = ['10.1016/j.jinorgbio.2019.110922', '10.1016/j.jinorgbio.2025.113041',
               '10.1021/acs.inorgchem.3c03716', '10.1039/c2dt12136k']
CL_SERIES = {'10.1016/j.jinorgbio.2022.112058': 'Phosphate', '10.1016/j.jinorgbio.2017.12.017': 'Phosphate'}
ART_COLOURS = ['#2E6E8E', '#C1442F', '#4A9B6F', '#D08C34']
BUF_MARKERS = {'Acetate': 's', 'Phosphate': 'o', 'Tris': '^', 'HEPES': 'D', 'MES': 'v'}
GROUPS = [('cationic', True, 'cat.\nM–Cl'), ('neutral', True, 'neut.\nM–Cl'),
          ('cationic', False, 'cat.\nno M–Cl'), ('neutral', False, 'neut.\nno M–Cl'),
          ('anionic', True, 'an.\nM–Cl')]


def short(doi):
    return doi.split('/')[-1]


def panel_chloride(ax, p):
    s = p[p['contrast'].eq('salt')]
    rng = np.random.default_rng(0)
    labels = []
    for i, (cls, mcl, label) in enumerate(GROUPS):
        g = s[s['class'].eq(cls) & s['mcl'].eq(mcl)]['shift']
        labels.append(f'{label}\n({len(g)})')
        if g.empty:
            continue
        ax.scatter(i + rng.normal(0, 0.07, len(g)), g, s=14, color='#2E6E8E' if mcl else '#999999',
                   alpha=0.75, lw=0)
        if len(g) >= 3:
            ax.plot([i - 0.3, i + 0.3], [g.median()] * 2, color='black', lw=1.4)
    ax.axhline(0, color='#999999', lw=0.8, ls='--')
    ax.set_xticks(range(len(GROUPS)), labels, fontsize=6.5)
    ax.set_ylabel('Δlog$P$ (high − low Cl$^-$)')


def panel_ph(ax, c):
    ab = pd.read_csv(ROOT / 'data' / 'MetalLipoDB.csv', low_memory=False)
    for k, doi in enumerate(PH_ARTICLES):
        x = c[c['doi'].eq(doi) & c['ph'].notna()]
        for smi, g in x.groupby('smiles_complex'):
            g = g.sort_values('ph')
            ax.plot(g['ph'], g['v'], color=ART_COLOURS[k], lw=0.9, alpha=0.8)
            for buf, h in g.groupby('buf'):
                ax.scatter(h['ph'], h['v'], color=ART_COLOURS[k], s=14, marker=BUF_MARKERS.get(buf, 'o'), lw=0)
        ax.plot([], [], color=ART_COLOURS[k], label=short(doi))
    for buf, m in BUF_MARKERS.items():
        if buf in set(c[c['doi'].isin(PH_ARTICLES)]['buf']):
            ax.scatter([], [], color='#555555', marker=m, s=14, label=buf.lower())
    ax.set(xlabel='pH', ylabel='log$P$')
    ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)


def panel_cl(ax, c):
    """Grouped bars: one group per complex, one bar per chloride concentration."""
    raw = pd.read_csv(ROOT / 'data' / 'MetalLipoDB.csv', low_memory=False)
    raw['doi'] = raw['doi'].str.lower()
    raw = raw[raw['doi'].isin(CL_SERIES) & raw['value_relation'].isin(['=', '~'])
              & raw['buffer_class'].eq('Phosphate')].copy()
    raw['cl'] = pd.to_numeric(raw['halide_conc_mM'], errors='coerce').fillna(0)
    tab = raw.groupby(['doi', 'abbreviation_in_the_article', 'cl'])['value'].mean().unstack('cl')
    levels = list(tab.columns)
    cmap = plt.get_cmap('Blues')
    colours = {cl: cmap(0.3 + 0.7 * k / max(len(levels) - 1, 1)) for k, cl in enumerate(levels)}
    art = {doi: 'AB'[k] for k, doi in enumerate(sorted(tab.index.get_level_values(0).unique()))}
    seen = set()
    for i, (key, row) in enumerate(tab.iterrows()):
        vals = row.dropna()
        w = 0.8 / len(vals)
        for k, (cl, v) in enumerate(vals.items()):
            ax.bar(i - 0.4 + w * (k + 0.5), v, width=w, color=colours[cl], edgecolor='white', lw=0.3,
                   label=None if cl in seen else f'{cl:g} mM')
            seen.add(cl)
    ax.axhline(0, color='#555555', lw=0.6)
    ax.set_xticks(range(len(tab)), [f'{a} ({art[doi]})' for doi, a in tab.index], fontsize=7)
    print('panel c articles:', {v: k for k, v in art.items()})
    ax.set_ylabel('log$P$')
    h, l = ax.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda k: float(l[k].split()[0]))
    ax.legend([h[k] for k in order], [l[k] for k in order], title='[Cl$^-$]', frameon=False, fontsize=6,
              title_fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=str(ROOT / 'figures'))
    out_dir = Path(ap.parse_args().out_dir)
    c = conditions()
    p = pairs(c)
    s = p[p['contrast'].eq('salt')]
    for cls, mcl, label in GROUPS:
        g = s[s['class'].eq(cls) & s['mcl'].eq(mcl)]
        if len(g):
            print(f"{label.replace(chr(10), ' ')}: {len(g)} complexes, {g['doi'].nunique()} articles, "
                  f"median {g['shift'].median():+.2f}, positive {(g['shift'] > 0).sum()}/{len(g)}")
    plt.rcParams.update(RC)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.6), gridspec_kw={'width_ratios': [1.2, 1, 1]})
    panel_chloride(axs[0], p)
    panel_ph(axs[1], c)
    panel_cl(axs[2], c)
    for k, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.02, 1.04, 'abc'[k] + ')', transform=ax.transAxes, fontsize=10, ha='right')
    plt.tight_layout()
    out = out_dir / 'figure_conditions'
    fig.savefig(out.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    print(f'saved: {out}.png / .pdf')


if __name__ == '__main__':
    main()
