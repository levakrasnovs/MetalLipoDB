"""
MetalLipoDB — plot_figures_metallipodb.py
=========================================
Generates the publication figures from the MetalLipoDB lipophilicity table.

Usage:
    python plot_figures_metallipodb.py                       # all figures
    python plot_figures_metallipodb.py --figures 1 3         # selected figures
    python plot_figures_metallipodb.py --data ... --out_dir figures/

Figures:
    1  figure_detection_counts     detection methods
    2  figure_logp_vs_sangster     MetalLipoDB vs SangsterLogP
    3  figure_logp_by_metal        log P distributions for Ru, Pt, Ir, Au, Cu
    4  figure_metal_counts         records per metal
    5  figure_charge_counts        records per complex charge
    6  figure_medium_counts        records per aqueous-medium class

Output: <out_dir>/<name>.pdf and <name>.png (600 dpi)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Shared config
# ══════════════════════════════════════════════════════════════════════════════

# Paths are resolved next to this script so the figures always come from the
# current cleaned table rather than a stale export.
ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA_LIPO = str(ROOT / 'data' / 'MetalLipoDB.csv')
DATA_SANGSTER = str(ROOT / 'data' / 'SangsterLogP.csv')
OUT_DIR = str(ROOT / 'figures')

RCPARAMS_STANDARD = {
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.linewidth':    0.6,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'legend.fontsize':   8,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
}

# the per-metal panel figure uses smaller y tick labels
RCPARAMS_BYMETAL = {**RCPARAMS_STANDARD, 'ytick.labelsize': 7}

C_MAIN = '#2E6E8E'   # primary
C_GREY = '#999999'   # aggregated / not-reported categories
C_SANG = '#C1442F'   # SangsterLogP

METAL_COLORS = {'Ru': '#2E6E8E', 'Pt': '#C1442F', 'Ir': '#4A9B6F',
                'Au': '#D08C34', 'Cu': '#7B5EA7'}


def load_lipo(path=DATA_LIPO):
    d = pd.read_csv(path)
    return d.loc[:, ~d.columns.str.startswith('Unnamed')]


def save(fig, out_dir, name):
    out = Path(out_dir) / name
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — detection methods
# ══════════════════════════════════════════════════════════════════════════════

def point_values(d):
    """Rows whose value is a measurement, not a detection-limit bound.

    The table keeps censored records with their bound in `value` and the
    operator in `value_relation`, so a distribution plot must exclude them: ">3.7"
    is not the number 3.7.
    """
    return d['value_relation'].isin(['=', '~'])


def figure_detection_counts(data=DATA_LIPO, out_dir=OUT_DIR):
    plt.rcParams.update(RCPARAMS_STANDARD)
    d = load_lipo(data)
    s = d['detection'].astype(str).str.strip()

    # normalisation: unicode dashes + synonymous technique names
    s = s.str.replace('\u2013', '-', regex=False).str.replace('\u2212', '-', regex=False)
    SYN = {'ICP-AES': 'ICP-OES', 'ICP spectroscopy': 'ICP-OES'}
    s = s.replace(SYN)
    s = s.mask(d['detection'].isna(), 'not reported')
    s = s.replace({'nan': 'not reported'})

    vc = s.value_counts()
    THR = 100
    keep = vc[(vc >= THR) & (vc.index != 'not reported')]
    small = vc[(vc < THR) & (vc.index != 'not reported')]
    nr = int(vc.get('not reported', 0))

    order = list(keep.items()) + [('not reported', nr), ('other', int(small.sum()))]
    order.sort(key=lambda t: -t[1])
    labels = [k for k, _ in order]
    counts = [v for _, v in order]
    cols = [C_GREY if k in ('not reported', 'other') else C_MAIN for k in labels]

    fig, ax = plt.subplots(figsize=(4.0, 2.1))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, counts, color=cols, height=0.72, edgecolor='none', zorder=3)
    for yy, c in zip(y, counts):
        ax.text(c + 45, yy, f'{c:,}', va='center', ha='left', fontsize=7, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Number of records')
    ax.set_xlim(0, max(counts) * 1.16)
    ax.tick_params(axis='y', length=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.text(0.98, 0.02, f'{len(d):,} records\n{vc.size} categories',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, color='#555555', linespacing=1.4)
    fig.tight_layout()
    save(fig, out_dir, 'figure_detection_counts')

    print('total', len(d), '| sum shown', sum(counts))
    print('other =', ', '.join(f'{k} {v}' for k, v in small.items()))
    print(dict(order))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — MetalLipoDB vs SangsterLogP
# ══════════════════════════════════════════════════════════════════════════════

def figure_logp_vs_sangster(data=DATA_LIPO, sangster=DATA_SANGSTER, out_dir=OUT_DIR):
    plt.rcParams.update(RCPARAMS_STANDARD)
    C_M, C_S = C_MAIN, C_SANG

    d = pd.read_csv(data)
    v = d.loc[point_values(d), 'value'].dropna()
    s = pd.read_csv(sangster)
    lp = pd.to_numeric(s['logP'].astype(str).str.replace(',', '.'),
                       errors='coerce').dropna()

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bins = np.arange(-5, 12.01, 0.25)
    ax.hist(lp, bins=bins, density=True, color=C_S, alpha=0.45, edgecolor='none',
            label=f'SangsterLogP, organic ($n$ = {len(lp):,})')
    ax.hist(v, bins=bins, density=True, color=C_M, alpha=0.65, edgecolor='none',
            label=f'MetalLipoDB (this work) ($n$ = {len(v):,})')

    mM, mS = v.median(), lp.median()
    ax.set_xlim(-5, 9)
    ax.set_ylim(0, 0.52)
    _yTOP = 0.435
    for x, c in [(mM, C_M), (mS, C_S)]:
        ax.vlines(x, 0, _yTOP, color=c, lw=1.0, ls=(0, (4, 2)), zorder=6)

    y = 0.435
    ax.annotate('', xy=(mM, y), xytext=(mS, y),
                arrowprops=dict(arrowstyle='<->', lw=0.7, color='#444444',
                                shrinkA=0, shrinkB=0))
    ax.text((mM + mS) / 2, y + 0.028, f'\u0394 median = {mS - mM:.2f}',
            ha='center', va='bottom', fontsize=7.5)
    ax.text(mM - 0.20, y, f'{mM:.2f}', color=C_M, ha='right', va='center', fontsize=7.5)
    ax.text(mS + 0.20, y, f'{mS:.2f}', color=C_S, ha='left', va='center', fontsize=7.5)

    ax.set_xlabel('log $P$')
    ax.set_ylabel('Density')
    ax.set_xticks(np.arange(-4, 9.1, 2))
    ax.legend(frameon=False, fontsize=7.5, loc='lower left', bbox_to_anchor=(0.0, 1.01),
              ncol=1, handlelength=1.1, handletextpad=0.6, borderpad=0.0,
              labelspacing=0.35)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, out_dir, 'figure_logp_vs_sangster')

    print('medians %.2f %.2f  delta %.2f' % (mM, mS, mS - mM))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — log P distributions per metal
# ══════════════════════════════════════════════════════════════════════════════

def figure_logp_by_metal(data=DATA_LIPO, out_dir=OUT_DIR):
    plt.rcParams.update(RCPARAMS_BYMETAL)
    d = load_lipo(data)
    m = d['metal'].astype(str).str.strip()

    SEL = ['Ir', 'Au', 'Ru', 'Cu', 'Pt']      # ordered by median, high -> low
    COL = METAL_COLORS
    bins = np.arange(-5, 5.01, 0.4)

    fig, axes = plt.subplots(len(SEL), 1, figsize=(4.6, 4.6), sharex=True)
    for ax, k in zip(axes, SEL):
        x = d.loc[m.eq(k) & point_values(d), 'value'].dropna()
        ax.hist(x, bins=bins, density=True, color=COL[k],
                edgecolor='white', linewidth=0.3)
        med = x.median()
        ax.axvline(med, color='#333333', lw=0.9, ls=(0, (3, 2)), zorder=5)
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 0.52)
        ax.set_yticks([0, 0.25, 0.5])
        ax.text(0.015, 0.86, k, transform=ax.transAxes, fontsize=10,
                fontweight='bold', color=COL[k], va='top', ha='left')
        ax.text(0.985, 0.86, f'$n$ = {len(x):,}   median = {med:.2f}',
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                color='#444444')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel('log $P$')
    fig.supylabel('Density', fontsize=9, x=0.005)
    fig.subplots_adjust(hspace=0.28)
    save(fig, out_dir, 'figure_logp_by_metal')

    print('ok')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — records per metal
# ══════════════════════════════════════════════════════════════════════════════

def figure_metal_counts(data=DATA_LIPO, out_dir=OUT_DIR):
    plt.rcParams.update(RCPARAMS_STANDARD)
    d = load_lipo(data)
    m = d['metal'].astype(str).str.strip()
    mixed = m.str.contains('-')

    vc = m[~mixed].value_counts()
    keep = vc[vc >= 50]
    small = vc[vc < 50]
    labels = list(keep.index) + ['other metals', 'heterometallic']
    counts = list(keep.values) + [int(small.sum()), int(mixed.sum())]
    cols = [C_MAIN] * len(keep) + [C_GREY, C_GREY]
    print('collapsed into "other metals": %d elements, %d records ->'
          % (len(small), small.sum()),
          ', '.join(f'{k} {v}' for k, v in small.items()))

    fig, ax = plt.subplots(figsize=(4.0, 3.1))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, counts, color=cols, height=0.72, edgecolor='none')
    for yy, c in zip(y, counts):
        ax.text(c + 18, yy, f'{c:,}', va='center', ha='left', fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Number of records')
    ax.set_xlim(0, max(counts) * 1.16)
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.text(0.98, 0.02, f'{len(d):,} records\n{len(vc)} elements',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, color='#555555', linespacing=1.4)
    fig.tight_layout()
    save(fig, out_dir, 'figure_metal_counts')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — complex charge
# ══════════════════════════════════════════════════════════════════════════════

def figure_charge_counts(data=DATA_LIPO, out_dir=OUT_DIR):
    plt.rcParams.update(RCPARAMS_STANDARD)
    d = load_lipo(data)
    q = pd.to_numeric(d['complex_charge'], errors='coerce')

    # charges beyond -2..+2 are rare; collapse the tails
    order, counts, cols = [], [], []
    lo = int((q <= -2).sum())
    if lo:
        order.append('\u2264 \u22122'); counts.append(lo); cols.append(C_MAIN)
    for z in (-1, 0, 1, 2):
        order.append(f'{z:+d}'.replace('+0', '0').replace('-', '\u2212'))
        counts.append(int((q == z).sum()))
        cols.append(C_MAIN)
    hi = int((q >= 3).sum())
    if hi:
        order.append('\u2265 +3'); counts.append(hi); cols.append(C_MAIN)
    nr = int(q.isna().sum())

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    x = np.arange(len(order))
    ax.bar(x, counts, color=cols, width=0.72, edgecolor='none')
    for xx, c in zip(x, counts):
        ax.text(xx, c + max(counts) * 0.02, f'{c:,}', ha='center', va='bottom',
                fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=7.5)
    ax.set_xlabel('Charge of the complex')
    ax.set_ylabel('Number of records')
    ax.set_ylim(0, max(counts) * 1.16)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, out_dir, 'figure_charge_counts')

    print('cationic %d | neutral %d | anionic %d | not reported %d'
          % (int((q > 0).sum()), int((q == 0).sum()), int((q < 0).sum()), nr))
    print(dict(zip(order, counts)), '| charge not reported (excluded):', nr)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — aqueous medium class
# ══════════════════════════════════════════════════════════════════════════════

def figure_medium_counts(data=DATA_LIPO, out_dir=OUT_DIR):
    plt.rcParams.update(RCPARAMS_STANDARD)
    d = load_lipo(data)
    mc = d['medium_class'].astype(str).str.strip()
    mc = mc.mask(d['medium_class'].isna(), 'Unknown').replace({'nan': 'Unknown'})
    mc = mc.mask(mc.eq('medium_class'), 'Unknown')   # stray repeated header cell

    vc = mc.value_counts()
    known = vc[vc.index != 'Unknown']
    labels = list(known.index) + ['not reported']
    counts = list(known.values) + [int(vc.get('Unknown', 0))]
    cols = [C_MAIN] * len(known) + [C_GREY]

    fig, ax = plt.subplots(figsize=(4.0, 2.1))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, counts, color=cols, height=0.72, edgecolor='none')
    for yy, c in zip(y, counts):
        ax.text(c + 30, yy, f'{c:,}', va='center', ha='left', fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Number of records')
    ax.set_xlim(0, max(counts) * 1.16)
    ax.tick_params(axis='y', length=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.text(0.98, 0.02, f'{len(d):,} records', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.5, color='#555555')
    fig.tight_layout()
    save(fig, out_dir, 'figure_medium_counts')

    print(dict(zip(labels, counts)), '| sum', sum(counts))


# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 (SI) — mononuclear vs multinuclear
# ══════════════════════════════════════════════════════════════════════════════

def figure_nuclearity(data=DATA_LIPO, out_dir=OUT_DIR):
    plt.rcParams.update(RCPARAMS_STANDARD)
    d = load_lipo(data)
    # the column marks multinuclear rows with YES and leaves the rest empty
    multi = d['multinuclear'].fillna('').astype(str).str.upper().eq('YES')

    counts = [int((~multi).sum()), int(multi.sum())]
    labels = ['mononuclear', 'multinuclear']
    total = len(d)
    print(f'mononuclear {counts[0]:,} | multinuclear {counts[1]:,} '
          f'({counts[1] / total:.1%}) | {d.loc[multi, "doi"].nunique()} papers')

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    x = np.arange(len(labels))
    ax.bar(x, counts, color=[C_MAIN, C_SANG], width=0.6, edgecolor='none')
    for xx, c in zip(x, counts):
        ax.text(xx, c + total * 0.015, f'{c:,}\n({c / total:.0%})',
                ha='center', va='bottom', fontsize=7.5, linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of records')
    ax.set_ylim(0, max(counts) * 1.22)
    ax.tick_params(axis='x', length=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, out_dir, 'figure_nuclearity')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — composition of the database (main-text overview, panels A-E)
# ══════════════════════════════════════════════════════════════════════════════

def _cat_metal(d):
    m = d['metal'].astype(str).str.strip()
    mixed = m.str.contains('-')
    vc = m[~mixed].value_counts()
    keep, small = vc[vc >= 50], vc[vc < 50]
    labels = list(keep.index) + ['other metals', 'heterometallic']
    counts = list(keep.values) + [int(small.sum()), int(mixed.sum())]
    return labels, counts, [C_MAIN] * len(keep) + [C_GREY, C_GREY]


def _cat_detection(d):
    s = d['detection'].astype(str).str.strip()
    s = s.str.replace('\u2013', '-', regex=False).str.replace('\u2212', '-', regex=False)
    s = s.replace({'ICP-AES': 'ICP-OES', 'ICP spectroscopy': 'ICP-OES'})
    s = s.mask(d['detection'].isna(), 'not reported').replace({'nan': 'not reported'})
    vc = s.value_counts()
    keep = vc[(vc >= 100) & (vc.index != 'not reported')]
    small = vc[(vc < 100) & (vc.index != 'not reported')]
    order = list(keep.items()) + [('not reported', int(vc.get('not reported', 0))),
                                  ('other', int(small.sum()))]
    order.sort(key=lambda kv: -kv[1])
    labels = [k for k, _ in order]
    return labels, [v for _, v in order], [
        C_GREY if k in ('not reported', 'other') else C_MAIN for k in labels]


def _cat_medium(d):
    mc = d['medium_class'].astype(str).str.strip()
    mc = mc.mask(d['medium_class'].isna(), 'Unknown').replace({'nan': 'Unknown'})
    mc = mc.mask(mc.eq('medium_class'), 'Unknown')
    vc = mc.value_counts()
    known = vc[vc.index != 'Unknown']
    return (list(known.index) + ['not reported'],
            list(known.values) + [int(vc.get('Unknown', 0))],
            [C_MAIN] * len(known) + [C_GREY])


def _barh(ax, labels, counts, cols, xlabel='Number of records'):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, counts, color=cols, height=0.72, edgecolor='none')
    span = max(counts)
    for yy, c in zip(y, counts):
        ax.text(c + span * 0.02, yy, f'{c:,}', va='center', ha='left', fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_xlim(0, span * 1.2)
    ax.tick_params(axis='y', length=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)


def figure_composition(data=DATA_LIPO, out_dir=OUT_DIR):
    plt.rcParams.update(RCPARAMS_STANDARD)
    d = load_lipo(data)
    total = len(d)

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], hspace=0.55, wspace=0.55)
    ax_metal = fig.add_subplot(gs[:, 0])
    ax_det = fig.add_subplot(gs[0, 1])
    ax_med = fig.add_subplot(gs[0, 2])
    ax_chg = fig.add_subplot(gs[1, 1])
    ax_nuc = fig.add_subplot(gs[1, 2])

    _barh(ax_metal, *_cat_metal(d))
    _barh(ax_det, *_cat_detection(d))
    _barh(ax_med, *_cat_medium(d))

    # charge: vertical, the axis is ordered and reads left to right
    q = pd.to_numeric(d['complex_charge'], errors='coerce')
    order, counts = [], []
    lo = int((q <= -2).sum())
    if lo:
        order.append('\u2264 \u22122'); counts.append(lo)
    for z in (-1, 0, 1, 2):
        order.append(f'{z:+d}'.replace('+0', '0').replace('-', '\u2212'))
        counts.append(int((q == z).sum()))
    hi = int((q >= 3).sum())
    if hi:
        order.append('\u2265 +3'); counts.append(hi)
    x = np.arange(len(order))
    ax_chg.bar(x, counts, color=C_MAIN, width=0.72, edgecolor='none')
    for xx, c in zip(x, counts):
        ax_chg.text(xx, c + max(counts) * 0.02, f'{c:,}', ha='center', va='bottom',
                    fontsize=6.5)
    ax_chg.set_xticks(x)
    ax_chg.set_xticklabels(order, fontsize=6.5)
    ax_chg.set_xlabel('Charge of the complex', fontsize=8)
    ax_chg.set_ylabel('Number of records', fontsize=8)
    ax_chg.set_ylim(0, max(counts) * 1.18)
    for sp in ('top', 'right'):
        ax_chg.spines[sp].set_visible(False)

    multi = d['multinuclear'].fillna('').astype(str).str.upper().eq('YES')
    nuc = [int((~multi).sum()), int(multi.sum())]
    x = np.arange(2)
    ax_nuc.bar(x, nuc, color=[C_MAIN, C_SANG], width=0.6, edgecolor='none')
    for xx, c in zip(x, nuc):
        ax_nuc.text(xx, c + total * 0.015, f'{c:,}\n({c / total:.0%})', ha='center',
                    va='bottom', fontsize=6.5, linespacing=1.3)
    ax_nuc.set_xticks(x)
    ax_nuc.set_xticklabels(['mononuclear', 'multinuclear'], fontsize=7)
    ax_nuc.set_ylabel('Number of records', fontsize=8)
    ax_nuc.set_ylim(0, max(nuc) * 1.28)
    ax_nuc.tick_params(axis='x', length=0)
    for sp in ('top', 'right'):
        ax_nuc.spines[sp].set_visible(False)

    for ax, letter, title in ((ax_metal, 'A', 'Metal'),
                              (ax_det, 'B', 'Detection'),
                              (ax_med, 'C', 'Aqueous medium'),
                              (ax_chg, 'D', 'Complex charge'),
                              (ax_nuc, 'E', 'Nuclearity')):
        ax.set_title(f'{letter}. {title}', fontsize=9, loc='left', pad=4)

    fig.text(0.995, 0.005, f'{total:,} records \u00b7 {d["doi"].nunique():,} papers',
             ha='right', va='bottom', fontsize=7.5, color='#555555')
    save(fig, out_dir, 'figure_composition')
    print(f'composition panels drawn on {total:,} records')


FIGURES = {
    1: figure_detection_counts,
    2: figure_logp_vs_sangster,
    3: figure_logp_by_metal,
    4: figure_metal_counts,
    5: figure_charge_counts,
    6: figure_medium_counts,
    7: figure_nuclearity,
    8: figure_composition,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--figures', nargs='*', type=int, default=sorted(FIGURES),
                   help='figure numbers to generate (default: all)')
    p.add_argument('--data', default=DATA_LIPO)
    p.add_argument('--sangster', default=DATA_SANGSTER)
    p.add_argument('--out_dir', default=OUT_DIR)
    a = p.parse_args()

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for n in a.figures:
        fn = FIGURES[n]
        print(f'--- figure {n}: {fn.__name__}')
        if fn is figure_logp_vs_sangster:
            fn(a.data, a.sangster, a.out_dir)
        else:
            fn(a.data, a.out_dir)


if __name__ == '__main__':
    main()
