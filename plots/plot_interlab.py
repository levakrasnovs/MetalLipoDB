"""
MetalLipoDB — plot_interlab.py
==============================
Experiment E1: how far apart do independent papers land on the same compound?

E1a  figure_interlab_spread          spread across replicated compounds
E1b  figure_cisplatin_decomposition  cisplatin (45 papers) split by conditions

The unit of replication is the pair (smiles_complex, counterion) measured in more
than one paper; see the record-identity convention. Censored rows are excluded —
a bound is not a measurement.

Usage:
    python plot_interlab.py [--data MetalLipoDB.csv] [--out_dir figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA = str(ROOT / 'data' / 'MetalLipoDB.csv')
OUT_DIR = str(ROOT / 'figures')

RC = {
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 0.6, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
}
C_MAIN = '#2E6E8E'
C_ACC = '#C1442F'
C_GREY = '#999999'

# minimum replicates for a standard deviation to be worth plotting
MIN_N_FOR_SD = 3

ANCHORS = [
    ('cisplatin',        '[NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]', 'none'),
    ('carboplatin',      '[NH3]->[Pt+2]1(<-[NH3])<-[O-]C(=O)C2(C(=O)[O-]->1)CCC2', 'none'),
    ('oxaliplatin',      'O=C1[O-]->[Pt+2]2(<-[NH2][C@@H]3CCCC[C@H]3[NH2]->2)<-[O-]C1=O', 'none'),
    ('satraplatin',      'CC(=O)[O-]->[Pt+4](<-[NH3])(<-[Cl-])(<-[Cl-])(<-[NH2]C1CCCCC1)<-[O-]C(C)=O', 'none'),
]


def load(path=DATA):
    d = pd.read_csv(path)
    d = d.loc[:, ~d.columns.str.startswith('Unnamed')]
    d['v'] = pd.to_numeric(d['value'], errors='coerce')
    # a censored row carries a bound, not a value: it cannot enter a spread
    d = d[d['value_relation'].isin(['=', '~'])].dropna(subset=['v'])
    d['cion'] = d['counterion'].fillna('none')
    return d


def replicate_groups(d):
    """(compound, counterion) pairs reported by more than one paper."""
    g = d.groupby(['smiles_complex', 'cion'])
    stat = g.agg(n=('v', 'size'), papers=('doi', 'nunique'),
                 sd=('v', 'std'), rng=('v', lambda s: s.max() - s.min()),
                 med=('v', 'median'))
    return stat[stat['papers'] > 1]


def save(fig, out_dir, name):
    out = Path(out_dir) / name
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved: {out}.png / .pdf')


# ── E1a ───────────────────────────────────────────────────────────────────────
def figure_interlab_spread(data=DATA, out_dir=OUT_DIR):
    plt.rcParams.update(RC)
    d = load(data)
    stat = replicate_groups(d)
    sd = stat.loc[stat['n'] >= MIN_N_FOR_SD, 'sd'].dropna()

    print(f'--- E1a\nreplicated groups: {len(stat)} covering {int(stat["n"].sum())} records')
    print(f'groups with n >= {MIN_N_FOR_SD}: {len(sd)} | median SD {sd.median():.2f} '
          f'| median range {stat["rng"].median():.2f}')

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                               gridspec_kw={'width_ratios': [1, 1.15]})

    # mean |Δ| between two articles, per complex (each complex weighted equally),
    # for all pairs and separately for pairs in the same / different medium class
    from interlab_pairs import article_pairs, bootstrap_mean
    pairs = article_pairs(pd.read_csv(data, low_memory=False))
    rng_b = np.random.default_rng(0)
    # distribution of the per-complex mean |Δ| (each complex weighted equally)
    per = pairs.groupby('key')['d'].mean().to_numpy()
    mean, lo, hi = bootstrap_mean(per, rng_b)
    print(f'all pairs: {len(per)} complexes, mean |Δ| {mean:.2f} ({lo:.2f}-{hi:.2f}), median {np.median(per):.2f}')
    a.axvspan(lo, hi, color=C_ACC, alpha=0.15, lw=0)
    a.hist(per, bins=np.arange(0, per.max() + 0.2, 0.2), color=C_MAIN, edgecolor='white', lw=0.4)
    a.axvline(mean, color=C_ACC, lw=1.4, ls='--')
    a.text(mean + 0.08, a.get_ylim()[1] * 0.92, f'mean {mean:.2f}\n(95% CI {lo:.2f}–{hi:.2f})',
           color=C_ACC, fontsize=7.5, va='top')
    a.set_xlabel('mean |Δlog$P$| between two articles')
    a.set_ylabel(f'number of complexes (of {len(per)})')
    a.text(-0.02, 1.03, 'a)', transform=a.transAxes, fontsize=12, ha='right', va='bottom')

    # anchors: every individual measurement, so the reader sees the raw disagreement
    rows, labels = [], []
    for name, smi, ci in ANCHORS:
        v = d[(d['smiles_complex'] == smi) & (d['cion'] == ci)]['v'].values
        rows.append(v)
        labels.append(f'{name}\n({len(v)} papers)')
    rng = np.random.default_rng(0)
    for i, v in enumerate(rows):
        b.scatter(v, i + rng.normal(0, 0.06, len(v)), s=14, color=C_MAIN,
                  alpha=0.65, lw=0)
        b.plot([np.median(v)] * 2, [i - 0.28, i + 0.28], color=C_ACC, lw=1.6)
    b.set_yticks(range(len(rows)))
    b.set_yticklabels(labels)
    b.set_ylim(-0.6, len(rows) - 0.4)
    b.set_xlabel('log$P$ (log units)')
    b.text(-0.02, 1.03, 'b)', transform=b.transAxes, fontsize=12, ha='right', va='bottom')
    b.grid(axis='x', lw=0.3, color='#dddddd')
    b.set_axisbelow(True)

    for ax in (a, b):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, out_dir, 'figure_interlab_spread')


# ── E1b ───────────────────────────────────────────────────────────────────────
def figure_cisplatin_decomposition(data=DATA, out_dir=OUT_DIR):
    plt.rcParams.update(RC)
    d = load(data)
    smi = ANCHORS[0][1]
    c = d[(d['smiles_complex'] == smi) & (d['cion'] == 'none')].copy()
    c['detection'] = c['detection'].fillna('not reported')
    c['medium_class'] = c['medium_class'].fillna('not reported')

    print(f'--- E1b\ncisplatin: {len(c)} records from {c["doi"].nunique()} papers, '
          f'SD {c["v"].std():.2f}, range {c["v"].min():.2f}..{c["v"].max():.2f}')

    # sequential (type I) sums of squares: how much of the spread do the reported
    # conditions account for at all
    model = ols('v ~ C(medium_class) + C(detection)', data=c).fit()
    aov = sm.stats.anova_lm(model, typ=1)
    ss_tot = aov['sum_sq'].sum()
    print((aov['sum_sq'] / ss_tot * 100).round(1).to_string())
    resid_sd = np.sqrt(aov.loc['Residual', 'sum_sq'] / model.df_resid)
    explained = 100 * (1 - aov.loc['Residual', 'sum_sq'] / ss_tot)
    print(f'explained by reported conditions: {explained:.0f}% | residual SD {resid_sd:.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2),
                            gridspec_kw={'width_ratios': [1, 1.25]})
    rng = np.random.default_rng(0)

    for ax, col, title in ((axes[0], 'medium_class', 'A. By aqueous medium'),
                           (axes[1], 'detection', 'B. By detection method')):
        order = c[col].value_counts().index[::-1]
        for i, k in enumerate(order):
            v = c.loc[c[col] == k, 'v'].values
            ax.scatter(v, i + rng.normal(0, 0.07, len(v)), s=16, color=C_MAIN,
                       alpha=0.65, lw=0)
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=C_ACC, lw=1.6)
        ax.axvline(c['v'].median(), color=C_GREY, lw=0.8, ls=':', zorder=0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f'{k} ({(c[col] == k).sum()})' for k in order])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel('log$P$ (log units)')
        ax.set_title(title, fontsize=10, loc='left')
        ax.grid(axis='x', lw=0.3, color='#dddddd')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Cisplatin, {len(c)} independent reports: '
                 f'conditions explain {explained:.0f}% of the variance, '
                 f'residual SD {resid_sd:.2f}',
                 fontsize=9.5, y=1.03)
    plt.tight_layout()
    save(fig, out_dir, 'figure_cisplatin_decomposition')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=DATA)
    p.add_argument('--out_dir', default=OUT_DIR)
    a = p.parse_args()
    figure_interlab_spread(a.data, a.out_dir)
    figure_cisplatin_decomposition(a.data, a.out_dir)
