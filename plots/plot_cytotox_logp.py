"""
MetalLipoDB — plot_cytotox_logp.py
==================================
Is a more lipophilic complex of a series also more cytotoxic?

Dark IC50 values come from two sources: MetalCytoToxDB, matched to MetalLipoDB
by metal, the sorted canonical ligand SMILES and the counterion, and
MetalLipoDB_LogP_IC50.csv (IC50 values extracted from the MetalLipoDB articles
themselves; its rows carry the MetalLipoDB SMILES, counterion and DOI and are
matched on them). The two sources share no articles. A series is one article, one
cell line and one exposure time with at least MIN_SERIES complexes whose log P
and dark IC50 were both reported in that article; values are centred on the
series mean, so differences between laboratories, cell lines and protocols
drop out. Censored or approximate values (>, <, >=, ~, ...) and rows without a
reported exposure time are excluded.

  a) centred pIC50 against centred log P within series of cancer cell lines
     (NON_CANCER lines are kept aside in load.non_cancer), coloured by metal;
     Theil–Sen slope, 95% CI from a bootstrap over articles
  b-f) the five cell lines with the most matched complexes at 48 h: median
     pIC50 of each complex across articles against its log P, Theil–Sen line

Usage:
    python plot_cytotox_logp.py [--cyto MetalCytoToxDB.csv]
                                [--extra MetalLipoDB_LogP_IC50_harmonised.csv] [--out_dir figures]

The second source is not public yet: without it the figure is built from
MetalCytoToxDB alone, so the counts are smaller than in the paper.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr, theilslopes

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA_LIPO = ROOT / 'data' / 'MetalLipoDB.csv'
DATA_CYTO = ROOT / 'data' / 'MetalCytoToxDB.csv'   # MetalCytoToxDB from Zenodo, see README
DATA_EXTRA = ROOT / 'data' / 'MetalLipoDB_LogP_IC50_harmonised.csv'  # cell lines named as in MetalCytoToxDB
OUT_DIR = ROOT / 'figures'

MIN_SERIES = 4
N_CELL_LINES = 5
# panels b-f: the five lines with most matched complexes in MetalCytoToxDB alone;
# (MDA-MB-231 is sixth, 205 complexes against 206 for A549cisR, and is not shown)
PANEL_LINES = ['HeLa', 'A549', 'MCF-7', 'HepG2', 'A549cisR']
PANEL_TIME = 48
# non-cancerous cell lines: their pIC50 reads as toxicity, not antitumour activity
NON_CANCER = ['HEK293', 'HEK293T', 'MRC-5', 'MRC5pd30', 'LO2', 'HL-7702', 'MCF-10A', 'MCF-12A', 'HaCaT', 'BEAS-2B',
              'RPE-1', 'hTERT-RPE1', 'ARPE-19', 'NIH-3T3', 'BALB/3T3', 'BALB/3T3 clone A31', 'L-929', 'CHO', 'CHO-K1',
              'HFF-1', 'HFF-2', 'WI-38', 'Vero', 'HUVEC', 'HUV-EC-C', 'PNT1A', 'fibroblasts',
              'human embryonic lung fibroblasts', 'primary human dermal fibroblasts', 'CCD-19Lu', 'CCD-1072Sk', 'HLF',
              'HFL-1', 'V79-4', 'IOSE-80', 'GES-1', 'HK-2', '16HBE', 'NHDF', 'NHDF-Ad', 'LX-2', 'H9c2', 'RWPE-1',
              'HSF', 'hFB', 'KMST-6', 'hSMCs', 'AC16', 'BJ', 'IMR-90', 'NCM460', 'CCD-18Co', 'HPDE', 'SC-1']
SEED = 0

RC = {
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 0.6, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
}
METAL_COLORS = {'Ru': '#2E6E8E', 'Ir': '#4A9B6F', 'Au': '#D08C34',
                'Re': '#7B5EA7', 'Rh': '#E377C2', 'Os': '#17BECF',
                'Pt': '#B23A3A', 'Cu': '#8C564B', 'Fe': '#7F7F7F', 'Ag': '#BCBD22'}
SERIES = ['doi', 'cell_line', 'time']
LEGEND_METALS = ['Ru', 'Pt', 'Ir', 'Au', 'Cu', 'Os', 'Rh']  # the rest shown as 'other'


def _canon_set(smiles):
    """Dot-separated SMILES as a sorted set of canonical fragments."""
    if pd.isna(smiles) or str(smiles).strip() == '':
        return 'none'
    out = []
    for f in str(smiles).split('.'):
        m = Chem.MolFromSmiles(f)
        out.append(Chem.MolToSmiles(m) if m else f)
    return '.'.join(sorted(out))


def extra_tox(path, lipo):
    """Dark IC50 values reported in the MetalLipoDB articles themselves."""
    e = pd.read_csv(path, low_memory=False, encoding='utf-8-sig')
    e.columns = [c.lstrip('\ufeff') for c in e.columns]
    ic = e['IC50,μM Dark'].astype(str).str.strip()
    exact = ic.str.match(r'^\d+(\.\d+)?(\s*(±|\().*)?$')  # value, optionally ± or (+a/−b)
    e = e[exact & e['Cell_line'].notna()].copy()
    e['ic50'] = ic[exact].str.extract(r'^(\d+(?:\.\d+)?)')[0].astype(float)
    e['time'] = pd.to_numeric(e['Time(h)'], errors='coerce')
    e = e[(e['ic50'] > 0) & e['time'].notna()]
    e['doi'] = e['DOI'].str.lower()
    e['ci'] = e['Counterion'].fillna('')
    keys = lipo[['smiles_complex', 'ci', 'doi', 'key']].drop_duplicates(['smiles_complex', 'ci', 'doi'])
    e = e.merge(keys, left_on=['SMILES_complex', 'ci', 'doi'], right_on=['smiles_complex', 'ci', 'doi'])
    e['pic50'] = 6 - np.log10(e['ic50'])
    return e.rename(columns={'Cell_line': 'cell_line'})[SERIES + ['key', 'pic50']]


def load(lipo=DATA_LIPO, cyto=DATA_CYTO, extra=DATA_EXTRA):
    d = pd.read_csv(lipo)
    d = d[d['value_relation'].isin(['=', '~']) & d['smiles_ligands'].notna()].copy()
    d['key'] = d['metal'] + '|' + d['smiles_ligands'].map(_canon_set) + '|' + \
        d['counterion'].map(_canon_set)
    d['doi'] = d['doi'].str.lower()
    d['ci'] = d['counterion'].fillna('')

    c = pd.read_csv(cyto, low_memory=False)
    c = c.dropna(subset=['SMILES_Ligands'])
    censored = c['IC50_Dark(M*10^-6)'].astype(str).str.contains('>|<')
    c = c[~censored & (c['IC50_Dark_value'] > 0)].copy()
    c['key'] = c['Metal'] + '|' + c['SMILES_Ligands'].map(_canon_set) + '|' + \
        c['Counterion'].map(_canon_set)
    c['doi'] = c['DOI'].astype(str).str.lower()
    c['pic50'] = 6 - np.log10(c['IC50_Dark_value'])
    c = c.rename(columns={'Cell_line': 'cell_line', 'Time(h)': 'time'})
    c['time'] = pd.to_numeric(c['time'], errors='coerce')
    if extra and not Path(extra).exists():
        print(f'{extra} not found: MetalCytoToxDB only')
        extra = None
    if extra:
        e = extra_tox(extra, d)
        assert not set(e['doi']) & set(c['doi']), 'the two IC50 sources overlap'
        print(f'IC50 rows: MetalCytoToxDB {len(c)}, extra {len(e)}')
        c = pd.concat([c[SERIES + ['key', 'pic50']], e], ignore_index=True)

    logp = d.groupby(['doi', 'key'])['value'].median().rename('logp').reset_index()
    tox = c.groupby(SERIES + ['key'])['pic50'].median().reset_index()
    # across articles, for the per-cell-line panels
    load.cell = (c[c['time'] == PANEL_TIME].groupby(['cell_line', 'key'])['pic50'].median()
                 .reset_index().merge(d.groupby('key')['value'].median().rename('logp'),
                                      left_on='key', right_index=True))
    m = tox.merge(logp, on=['doi', 'key'])
    m = m[m.groupby(SERIES)['key'].transform('size') >= MIN_SERIES].copy()
    g = m.groupby(SERIES)
    m['logp_c'] = m['logp'] - g['logp'].transform('mean')
    m['pic50_c'] = m['pic50'] - g['pic50'].transform('mean')
    m['metal'] = m['key'].str.split('|').str[0]
    non_cancer = m['cell_line'].isin(NON_CANCER)
    load.non_cancer = m[non_cancer].copy()
    return m[~non_cancer].copy()


def slope_ci(x, n_boot=300):
    """Theil–Sen slope; the interval resamples articles, since series of one
    article share compounds and are not independent."""
    rng = np.random.default_rng(SEED)
    slope = theilslopes(x['pic50_c'], x['logp_c'])[0]
    groups = {k: v for k, v in x.groupby('doi')}
    keys = list(groups)
    boot = []
    for _ in range(n_boot):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys))])
        boot.append(theilslopes(s['pic50_c'], s['logp_c'])[0])
    return slope, *np.percentile(boot, [2.5, 97.5])


def series_rho(x):
    return pd.Series([spearmanr(s['logp'], s['pic50'])[0] for _, s in x.groupby(SERIES)]).dropna()


def figure(m, out_dir):
    plt.rcParams.update(RC)
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.2))
    axs = axs.ravel()

    a = axs[0]
    group = m['metal'].where(m['metal'].isin(LEGEND_METALS), 'other')
    for metal in LEGEND_METALS + ['other']:
        x = m[group == metal]
        a.scatter(x['logp_c'], x['pic50_c'], s=4, alpha=0.4, lw=0,
                  color=METAL_COLORS.get(metal, '#999999'), label=metal)
    sl, lo, hi = slope_ci(m)
    ic = np.median(m['pic50_c'] - sl * m['logp_c'])
    xx = np.array([-2.5, 2.5])
    a.plot(xx, ic + sl * xx, color='black', lw=1.3)
    rs = series_rho(m)
    a.set_title('within series', fontsize=8.5, loc='left')
    a.text(0.03, 0.97, f'{rs.size} series, {m["doi"].nunique()} articles\n'
           f'slope {sl:.2f} ({lo:.2f}–{hi:.2f})', transform=a.transAxes, va='top', fontsize=6.5)
    a.set_xlim(-3, 3)
    a.set_ylim(-2.2, 2.2)
    a.set_xlabel('logP − series mean')
    a.set_ylabel('pIC$_{50}$ − series mean')
    a.legend(frameon=False, loc='lower right', markerscale=2.5, ncol=2, fontsize=6,
             handletextpad=0.1, columnspacing=0.5)
    print(f'within series: {len(m)} points, {rs.size} series, {m["doi"].nunique()} articles, '
          f'{m["key"].nunique()} complexes | slope {sl:.2f} ({lo:.2f}-{hi:.2f}) | '
          f'positive in {(rs > 0).sum()}/{rs.size} series')

    cell = load.cell
    counts = cell.groupby('cell_line')['key'].nunique().sort_values(ascending=False)
    print('top lines by matched complexes:', counts.head(N_CELL_LINES + 1).to_dict())
    lines = PANEL_LINES
    for ax, line in zip(axs[1:], lines):
        x = cell[cell['cell_line'] == line]
        x = x.assign(metal=x['key'].str.split('|').str[0])
        for metal in x['metal'].value_counts().index:
            y = x[x['metal'] == metal]
            ax.scatter(y['logp'], y['pic50'], s=5, alpha=0.5, lw=0,
                       color=METAL_COLORS.get(metal, '#999999') if metal in LEGEND_METALS else '#999999')
        s_, i_, _, _ = theilslopes(x['pic50'], x['logp'])
        xx = np.array([x['logp'].quantile(0.01), x['logp'].quantile(0.99)])
        ax.plot(xx, i_ + s_ * xx, color='black', lw=1.3)
        rho = spearmanr(x['logp'], x['pic50'])[0]
        ax.set_title(f'{line}, {PANEL_TIME} h', fontsize=8.5, loc='left')
        ax.text(0.03, 0.97, f'{len(x)} complexes\nslope {s_:.2f}, ρ = {rho:.2f}',
                transform=ax.transAxes, va='top', fontsize=6.5)
        ax.set_xlim(-3.5, 4.5)
        ax.set_xlabel('logP')
        ax.set_ylabel('pIC$_{50}$')
        print(f'{line}: {len(x)} complexes, slope {s_:.2f}, rho {rho:.2f}')

    for k, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.02, 1.1, 'abcdef'[k] + ')', transform=ax.transAxes, fontsize=10,
                ha='right', va='bottom')
    plt.tight_layout()
    out = Path(out_dir) / 'figure_cytotox_logp'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved: {out}.png / .pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lipo', default=str(DATA_LIPO))
    p.add_argument('--cyto', default=str(DATA_CYTO))
    p.add_argument('--extra', default=str(DATA_EXTRA), help="'' to use MetalCytoToxDB only")
    p.add_argument('--out_dir', default=str(OUT_DIR))
    a = p.parse_args()
    figure(load(a.lipo, a.cyto, a.extra), a.out_dir)
