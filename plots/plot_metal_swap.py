"""
MetalLipoDB — plot_metal_swap.py
================================
Same set of ligands, same article, same aqueous medium, different metal: how far
does log P move? Supporting figure (companion of plot_counterion.py): one point
per pair, lighter against heavier metal (figure_metal_swap_parity_si). figure()
draws the older one-row-per-group version and is not used in the SI.

Usage:
    python plot_metal_swap.py [--data MetalLipoDB.csv] [--out_dir figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA = ROOT / 'data' / 'MetalLipoDB.csv'
OUT_DIR = ROOT / 'figures'

RC = {
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 0.6, 'xtick.labelsize': 8, 'ytick.labelsize': 7,
    'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
}
PERIODIC_GROUP = {m: g for g, row in enumerate(
    [['Mn', 'Tc', 'Re'], ['Fe', 'Ru', 'Os'], ['Co', 'Rh', 'Ir'],
     ['Ni', 'Pd', 'Pt'], ['Cu', 'Ag', 'Au'], ['Zn', 'Cd', 'Hg']], start=7) for m in row}
# one colour per periodic group, marker by period, so group mates share a colour
GROUP_COLOR = {7: '#7B5EA7', 8: '#2E6E8E', 9: '#4A9B6F', 10: '#C1442F', 11: '#D08C34', 12: '#8C564B'}
PERIOD_MARKER = {3: 'o', 4: 's', 5: 'D'}
PERIOD = {**{m: 3 for m in 'Mn Fe Co Ni Cu Zn'.split()},
          **{m: 4 for m in 'Tc Ru Rh Pd Ag Cd'.split()},
          **{m: 5 for m in 'Re Os Ir Pt Au Hg'.split()}}


def _canon_set(smiles):
    out = []
    for f in str(smiles).split('.'):
        m = Chem.MolFromSmiles(f)
        out.append(Chem.MolToSmiles(m) if m else f)
    return '.'.join(sorted(out))


def groups(path=DATA):
    d = pd.read_csv(path)
    d = d[d['value_relation'].isin(['=', '~']) & d['smiles_ligands'].notna()
          & d['metal'].isin(PERIODIC_GROUP)].copy()
    d['ligands'] = d['smiles_ligands'].map(_canon_set)
    g = (d.groupby(['doi', 'ligands', 'medium_class', 'metal'], dropna=False)
          .agg(value=('value', 'mean'), charge=('complex_charge', 'first')).reset_index())
    keep = g.groupby(['doi', 'ligands', 'medium_class'], dropna=False)['metal'].transform('nunique') > 1
    g = g[keep].copy()
    g['key'] = g['doi'] + '|' + g['ligands'] + '|' + g['medium_class'].fillna('')
    return g


def figure(g, out_dir):
    plt.rcParams.update(RC)
    rng = g.groupby('key')['value'].agg(lambda v: v.max() - v.min())
    same = g.groupby('key')['metal'].agg(lambda m: m.map(PERIODIC_GROUP).nunique() == 1)
    order = rng.sort_values().index
    metals = sorted(g['metal'].unique(), key=lambda m: (PERIODIC_GROUP[m], PERIOD[m]))

    fig, ax = plt.subplots(figsize=(7.2, 0.2 * len(order) + 1.4))
    for y, key in enumerate(order):
        r = g[g['key'] == key]
        ax.plot([r['value'].min(), r['value'].max()], [y, y], lw=1.4, zorder=1,
                color='#cccccc' if same[key] else '#f0c9c0')
        for _, row in r.iterrows():
            ax.scatter(row['value'], y, s=26, lw=0, zorder=3,
                       marker=PERIOD_MARKER[PERIOD[row['metal']]],
                       color=GROUP_COLOR[PERIODIC_GROUP[row['metal']]])
        ax.text(1.01, y, f'{rng[key]:.2f}', transform=ax.get_yaxis_transform(),
                va='center', fontsize=6.5, color='#555555')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([k.split('|')[0] for k in order])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel('logP')
    ax.text(1.01, len(order) - 0.2, 'range', transform=ax.get_yaxis_transform(),
            fontsize=7, color='#555555')
    for m in metals:
        ax.scatter([], [], s=26, lw=0, marker=PERIOD_MARKER[PERIOD[m]],
                   color=GROUP_COLOR[PERIODIC_GROUP[m]], label=m)
    ax.plot([], [], color='#cccccc', lw=1.4, label='same group')
    ax.plot([], [], color='#f0c9c0', lw=1.4, label='different groups')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.45, -0.06 - 1.2 / len(order)),
              ncol=8)
    ax.grid(axis='x', lw=0.3, color='#dddddd')
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    out = Path(out_dir) / 'figure_metal_swap_si'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)

    for label, sel in (('same group', same), ('different groups', ~same)):
        v = rng[sel[rng.index]]
        boot = np.median(np.random.default_rng(0).choice(v.values, (5000, len(v))), axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f'{label}: {len(v)} groups, median range {v.median():.2f} (95% CI {lo:.2f}-{hi:.2f})')
    print(f'{len(rng)} groups from {g["doi"].nunique()} articles')
    print(f'saved: {out}.png / .pdf')




# ── compact variant: one point per pair, lighter against heavier metal ────────
MAIN_PAIRS = {('Rh', 'Ir'): '#4A9B6F', ('Ru', 'Os'): '#2E6E8E',
              ('Pd', 'Pt'): '#C1442F', ('Tc', 'Re'): '#7B5EA7'}


def pairs(g):
    import itertools
    rows = []
    for _, x in g.groupby('key'):
        for (_, a), (_, b) in itertools.combinations(x.iterrows(), 2):
            ka = (PERIODIC_GROUP[a['metal']], PERIOD[a['metal']])
            kb = (PERIODIC_GROUP[b['metal']], PERIOD[b['metal']])
            if ka > kb:
                a, b = b, a
            rows.append((a['metal'], b['metal'], a['value'], b['value'],
                         PERIODIC_GROUP[a['metal']] == PERIODIC_GROUP[b['metal']], a['doi']))
    return pd.DataFrame(rows, columns=['m1', 'm2', 'x', 'y', 'same', 'doi'])


def _stats(p):
    d = p['y'] - p['x']
    r = np.corrcoef(p['x'], p['y'])[0, 1]
    return len(p), p['doi'].nunique(), d.abs().median(), d.median(), r


def figure_parity(g, out_dir):
    plt.rcParams.update(RC)
    p = pairs(g)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([-4, 4], [-4, 4], ls=':', color='#999999', lw=1)
    for (m1, m2), c in MAIN_PAIRS.items():
        x = p[(p['m1'] == m1) & (p['m2'] == m2)]
        n, _, mad, med, _ = _stats(x)
        ax.scatter(x['x'], x['y'], s=18, color=c, lw=0, alpha=0.85,
                   label=f'{m1} → {m2} ({n}): |Δ| {mad:.2f}, Δ {med:+.2f}')
    main = p.apply(lambda r: (r['m1'], r['m2']) in MAIN_PAIRS, axis=1)
    x = p[p['same'] & ~main]
    ax.scatter(x['x'], x['y'], s=18, facecolor='none', edgecolor='#555555', lw=0.7,
               label=f'other, same group ({len(x)})')
    x = p[~p['same']]
    ax.scatter(x['x'], x['y'], s=20, marker='x', color='#999999', lw=0.9,
               label=f'different groups ({len(x)})')

    lines = []
    for label, sel in (('same group', p['same']), ('different groups', ~p['same'])):
        n, na, mad, med, r = _stats(p[sel])
        lines.append(f'{label}: {n} pairs, {na} articles\n'
                     f'  median |Δ| {mad:.2f}, r = {r:.2f}')
        print(f'{label}: {n} pairs, {na} articles, median |Δ| {mad:.2f}, median Δ {med:+.2f}, r {r:.2f}')
    ax.text(0.97, 0.03, '\n'.join(lines), transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6.8, linespacing=1.35)
    ax.set_xlim(-3.8, 3.6)
    ax.set_ylim(-3.8, 3.6)
    ax.set_aspect('equal')
    ax.set_xlabel('logP, lighter metal')
    ax.set_ylabel('logP, heavier metal')
    ax.legend(frameon=False, fontsize=6.5, loc='upper left', handletextpad=0.3)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = Path(out_dir) / 'figure_metal_swap_parity_si'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved: {out}.png / .pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(DATA))
    p.add_argument('--out_dir', default=str(OUT_DIR))
    a = p.parse_args()
    figure_parity(groups(a.data), a.out_dir)
