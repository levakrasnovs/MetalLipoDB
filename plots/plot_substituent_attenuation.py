"""
MetalLipoDB — plot_substituent_attenuation.py
=============================================
Does a change of ligand shift log P of a metal complex as much as the same
change shifts log P of an organic molecule?

Pairs are matched molecular pairs (one cut, variable part <= 6 heavy atoms)
taken within one article, so the laboratory and the protocol are shared and
inter-laboratory noise cancels. The organic control is built the same way on
SangsterLogP, grouped by its Source column.

  a) measured ΔlogP against ΔCrippen logP (Wildman & Crippen 1999), binned,
     for complexes (Crippen of the ligands) and organic molecules
  b) median shift for H -> X in complexes and in organic molecules, both from
     the same pair search; Hansch π (J. Med. Chem. 1973, 16, 1207, Table I)
     is printed as a check of the organic medians

Usage:
    python plot_substituent_attenuation.py [--out_dir figures] [--rebuild]

Pair tables are cached in <out_dir>/pairs_metallipo.csv and
pairs_sangster.csv; --rebuild recomputes them (the organic set takes minutes).
"""

import argparse
import collections
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMMPA
from scipy.stats import theilslopes

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA_LIPO = ROOT / 'data' / 'MetalLipoDB.csv'
DATA_SANGSTER = ROOT / 'data' / 'SangsterLogP.csv'
OUT_DIR = ROOT / 'figures'

MAX_VARIABLE_ATOMS = 6
SEED = 0

RC = {
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 0.6, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
}
C_MAIN = '#2E6E8E'   # metal complexes
C_ORG = '#C1442F'    # organic

# π(benzene) from Hansch et al. 1973, Table I (checked against paper/literature/hansch1973.pdf), used
# only as an internal check printed to the console, not shown in the figure or the paper.
# n-butyl is listed there without a π value.
HANSCH_PI = [
    ('*CCCC', '$n$-C$_4$H$_9$', float('nan')),  # no π in Table I
    ('*c1ccccc1', 'C$_6$H$_5$', 1.96),
    ('*CC', 'C$_2$H$_5$', 1.02),
    ('*C(F)(F)F', 'CF$_3$', 0.88),
    ('*Br', 'Br', 0.86),
    ('*Cl', 'Cl', 0.71),
    ('*C', 'CH$_3$', 0.56),
    ('*F', 'F', 0.14),
    ('*OC', 'OCH$_3$', -0.02),
    ('*O', 'OH', -0.67),
    ('*CO', 'CH$_2$OH', -1.03),
]
H_KEY = '[*][H]'


# ── pair search ───────────────────────────────────────────────────────────────
def _canon(smi):
    m = Chem.MolFromSmiles(re.sub(r'\[\*:1\]', '[*]', smi))
    return Chem.MolToSmiles(m) if m else smi


def find_pairs(mols):
    """mols: {canonical smiles: (mol, value)} of one article.

    Returns (smiles_a, smiles_b, variable_a, variable_b). A hydrogen as the
    variable part is found by capping the constant part and looking the
    parent up among the article's molecules.
    """
    index = collections.defaultdict(dict)
    for cs, (m, _) in mols.items():
        for _, chains in rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False,
                                            maxCutBonds=200):
            parts = chains.split('.')
            if len(parts) != 2:
                continue
            frags = [Chem.MolFromSmiles(p) for p in parts]
            if None in frags:
                continue
            (small, ns), (big, _) = sorted(
                zip(parts, [f.GetNumHeavyAtoms() for f in frags]), key=lambda t: t[1])
            if ns - 1 > MAX_VARIABLE_ATOMS:
                continue
            key = _canon(big)
            index[key].setdefault(_canon(small), cs)
            parent = Chem.MolFromSmiles(re.sub(r'\[\*:1\]', '[H]', big))
            if parent is not None:
                ps = Chem.MolToSmiles(Chem.RemoveHs(parent))
                if ps in mols and ps != cs:
                    index[key].setdefault(H_KEY, ps)
    # the same two molecules are reached through several cuts; keep one pair,
    # the cut with the smallest variable part
    def size(v):
        return 0 if v == H_KEY else Chem.MolFromSmiles(v).GetNumHeavyAtoms()
    pairs = {}
    for variants in index.values():
        items = sorted(variants.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (va, sa), (vb, sb) = items[i], items[j]
                if sa == sb:
                    continue
                if sb < sa:
                    sa, sb, va, vb = sb, sa, vb, va
                cost = size(va) + size(vb)
                if (sa, sb) not in pairs or cost < pairs[(sa, sb)][0]:
                    pairs[(sa, sb)] = (cost, va, vb)
    return [(sa, sb, va, vb) for (sa, sb), (_, va, vb) in pairs.items()]


def pairs_metallipo(path=DATA_LIPO, keys=('medium_class',), subset=None):
    """keys: conditions a pair must share besides article and counterion,
    subset: optional row filter (e.g. only rows with reported pH)."""
    d = pd.read_csv(path)
    d = d[d['value_relation'].isin(['=', '~']) & (d['multinuclear'] != 'YES')].copy()
    if subset is not None:
        d = d[subset(d)]
    d['cion'] = d['counterion'].fillna('none')
    ligands = (d.dropna(subset=['smiles_ligands'])
                .drop_duplicates('smiles_complex')
                .set_index('smiles_complex'))
    rows = []
    # a pair must share article, counterion and aqueous medium
    for (doi, *_), g in d.groupby(['doi', 'cion', *keys], dropna=False):
        mols = {}
        for smi, v in g.groupby('smiles_complex')['value'].mean().items():
            m = Chem.MolFromSmiles(smi)
            if m is not None and smi in ligands.index:
                mols[Chem.MolToSmiles(m)] = (m, v, smi)
        if len(mols) < 2:
            continue
        for a, b, va, vb in find_pairs({k: v[:2] for k, v in mols.items()}):
            la = Chem.MolFromSmiles(ligands.at[mols[a][2], 'smiles_ligands'])
            lb = Chem.MolFromSmiles(ligands.at[mols[b][2], 'smiles_ligands'])
            if la is None or lb is None:
                continue
            rows.append((doi, va, vb, mols[b][1] - mols[a][1],
                         Crippen.MolLogP(lb) - Crippen.MolLogP(la),
                         ligands.at[mols[a][2], 'complex_charge']))
    return pd.DataFrame(rows, columns=['doi', 'var_a', 'var_b', 'd_meas', 'd_crippen', 'charge'])


def pairs_sangster(path=DATA_SANGSTER):
    s = pd.read_csv(path)
    s['v'] = pd.to_numeric(s['logP'].astype(str).str.replace(',', '.'), errors='coerce')
    s = s.dropna(subset=['v', 'SMILES', 'Source'])
    rows = []
    for src, g in s.groupby('Source'):
        mols = {}
        for smi, v in zip(g['SMILES'], g['v']):
            m = Chem.MolFromSmiles(smi)
            if m is None or m.GetNumHeavyAtoms() > 60:
                continue
            mols.setdefault(Chem.MolToSmiles(m), (m, []))[1].append(v)
        if len(mols) < 2:
            continue
        mols = {k: (m, np.mean(v)) for k, (m, v) in mols.items()}
        for a, b, va, vb in find_pairs(mols):
            rows.append((src, va, vb, mols[b][1] - mols[a][1],
                         Crippen.MolLogP(mols[b][0]) - Crippen.MolLogP(mols[a][0])))
    return pd.DataFrame(rows, columns=['doi', 'var_a', 'var_b', 'd_meas', 'd_crippen'])


# ── statistics ────────────────────────────────────────────────────────────────
def slope_by_paper(p, n_boot=300, cap=3000):
    """Theil–Sen slope with a bootstrap over articles: pairs within one article
    are not independent. Large sets are subsampled for speed."""
    rng = np.random.default_rng(SEED)
    sub = p.sample(min(len(p), 8000), random_state=SEED)
    slope = theilslopes(sub['d_meas'], sub['d_crippen'])[0]
    groups = {k: v for k, v in p.groupby('doi')}
    keys = list(groups)
    boot = []
    for _ in range(n_boot):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys))])
        s = s.sample(min(len(s), cap), random_state=int(rng.integers(1 << 30)))
        boot.append(theilslopes(s['d_meas'], s['d_crippen'])[0])
    return slope, *np.percentile(boot, [2.5, 97.5])


def binned(p, edges=(0, 0.5, 1, 1.5, 2, 3)):
    """Orient every pair so that Crippen predicts an increase, then take the
    median measured change per bin with a bootstrap interval."""
    rng = np.random.default_rng(SEED)
    sign = np.sign(p['d_crippen'])
    x, y = p['d_crippen'] * sign, p['d_meas'] * sign
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x > lo) & (x <= hi)
        v = y[m].values
        boot = np.median(rng.choice(v, (1000, len(v))), axis=1)
        rows.append((x[m].median(), np.median(v), *np.percentile(boot, [2.5, 97.5]), len(v)))
    return pd.DataFrame(rows, columns=['x', 'y', 'lo', 'hi', 'n'])


def substituent_shifts(p):
    """Median log P change for H -> X, positive when X raises it. The π column
    is kept only to check that the organic medians reproduce Hansch."""
    rng = np.random.default_rng(SEED)
    rows = []
    for smi, label, pi in HANSCH_PI:
        a = p[(p['var_a'] == smi) & (p['var_b'] == H_KEY)]['d_meas'].values
        b = p[(p['var_a'] == H_KEY) & (p['var_b'] == smi)]['d_meas'].values
        v = np.concatenate([-a, b])
        boot = np.median(rng.choice(v, (2000, len(v))), axis=1)
        rows.append((label, pi, np.median(v), *np.percentile(boot, [2.5, 97.5]), len(v)))
    return pd.DataFrame(rows, columns=['label', 'pi', 'med', 'lo', 'hi', 'n'])


# ── figure ────────────────────────────────────────────────────────────────────
def figure(pm, po, out_dir):
    plt.rcParams.update(RC)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 4.0),
                               gridspec_kw={'width_ratios': [1, 1.15]})

    tm, to = binned(pm), binned(po)
    sm, so = slope_by_paper(pm), slope_by_paper(po)
    a.plot([0, 2.6], [0, 2.6], ls=':', color='#999999', lw=1)
    a.errorbar(to['x'], to['y'], yerr=[to['y'] - to['lo'], to['hi'] - to['y']],
               fmt='s-', ms=5, color=C_ORG, capsize=2, lw=1.2,
               label=f'organic, SangsterLogP ({len(po):,} pairs)')
    a.errorbar(tm['x'], tm['y'], yerr=[tm['y'] - tm['lo'], tm['hi'] - tm['y']],
               fmt='o-', ms=5.5, color=C_MAIN, capsize=2, lw=1.2,
               label=f'metal complexes, MetalLipoDB ({len(pm):,} pairs)')
    # the Theil–Sen slopes (over all pairs, not the binned medians drawn here)
    # are given in the text and the caption
    a.set_xlim(0, 2.6)
    a.set_ylim(-0.1, 2.6)
    a.set_aspect('equal')
    a.set_xlabel('predicted ΔlogP (Crippen)')
    a.set_ylabel('measured ΔlogP')
    a.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16), fontsize=7)
    a.text(-0.02, 1.03, 'a)', transform=a.transAxes, fontsize=12, ha='right', va='bottom')

    t = substituent_shifts(pm)
    t_org = substituent_shifts(po)
    y = np.arange(len(t))[::-1]
    for yy, r, o in zip(y, t.itertuples(), t_org.itertuples()):
        b.plot([r.med, o.med], [yy, yy], color='#cccccc', lw=1.6, zorder=1)
        b.errorbar(o.med, yy, xerr=[[o.med - o.lo], [o.hi - o.med]], fmt='s', ms=5,
                   color=C_ORG, capsize=2, zorder=3)
        b.errorbar(r.med, yy, xerr=[[r.med - r.lo], [r.hi - r.med]], fmt='o', ms=5.5,
                   color=C_MAIN, capsize=2, zorder=3)
        b.text(2.73, yy, f'{o.n} / {r.n}', va='center', ha='right', fontsize=7, color='#666666')
    b.set_yticks(y)
    b.set_yticklabels(t['label'])
    b.axvline(0, color='#999999', lw=0.6)
    b.set_xlim(-1.3, 2.75)
    b.set_xlabel('ΔlogP for H → X')
    b.errorbar([], [], xerr=[[0], [0]], fmt='s', ms=5, color=C_ORG,
               label='organic, SangsterLogP')
    b.errorbar([], [], xerr=[[0], [0]], fmt='o', ms=5.5, color=C_MAIN,
               label='metal complexes, MetalLipoDB')
    b.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.45, -0.16), ncol=2, fontsize=7)
    b.text(2.73, len(t) - 0.35, 'pairs: organic / complexes', ha='right', fontsize=7, color='#666666')
    b.tick_params(axis='y', length=0)
    b.text(-0.02, 1.03, 'b)', transform=b.transAxes, fontsize=12, ha='right', va='bottom')

    for ax in (a, b):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = Path(out_dir) / 'figure_substituent_attenuation'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)

    print(f'complexes: {len(pm)} pairs, {pm["doi"].nunique()} articles, '
          f'slope {sm[0]:.2f} [{sm[1]:.2f}-{sm[2]:.2f}]')
    print(f'organic:   {len(po)} pairs, {po["doi"].nunique()} sources, '
          f'slope {so[0]:.2f} [{so[1]:.2f}-{so[2]:.2f}]')
    for q, name in ((pm['charge'] == 0, 'neutral'), (pm['charge'] > 0, 'cationic')):
        s = slope_by_paper(pm[q])
        print(f'  {name}: {q.sum()} pairs, slope {s[0]:.2f} [{s[1]:.2f}-{s[2]:.2f}]')
    print(t.assign(org=t_org['med'], org_n=t_org['n']).round(2).to_string(index=False))
    print(f'saved: {out}.png / .pdf')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=str(OUT_DIR))
    p.add_argument('--rebuild', action='store_true')
    args = p.parse_args()
    out = Path(args.out_dir)
    cache_m, cache_o = out / 'pairs_metallipo.csv', out / 'pairs_sangster.csv'
    if args.rebuild or not cache_m.exists():
        pairs_metallipo().to_csv(cache_m, index=False)
    if args.rebuild or not cache_o.exists():
        pairs_sangster().to_csv(cache_o, index=False)
    pm = pd.read_csv(cache_m)
    po = pd.read_csv(cache_o)
    # identical ligand sets give ΔCrippen = 0 and carry no information here
    pm = pm[pm['d_crippen'].abs() > 1e-6]
    po = po[po['d_crippen'].abs() > 1e-6]
    figure(pm, po, out)
    # robustness: conditions matched more strictly (missing values are not taken as a match)
    for name, keys, subset in [
            ('same reported aqueous phase and pH', ('aqueous_phase', 'pH'),
             lambda d: d['aqueous_phase'].notna() & d['pH'].notna()),
            ('pure water', ('aqueous_phase',), lambda d: d['medium_class'].eq('water'))]:
        q = pairs_metallipo(keys=keys, subset=subset)
        q = q[q['d_crippen'].abs() > 1e-6]
        s = slope_by_paper(q)
        print(f'  {name}: {len(q)} pairs, {q["doi"].nunique()} articles, slope {s[0]:.2f} [{s[1]:.2f}-{s[2]:.2f}]')


if __name__ == '__main__':
    main()
