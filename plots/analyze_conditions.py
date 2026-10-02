"""
MetalLipoDB — analyze_conditions.py
===================================
Effect of the aqueous phase on log P within articles. For every complex (same
counterion) measured in one article under several conditions, pairs of
conditions that differ in exactly one factor are compared:

  salt     no salt vs 100-154 mM chloride, same buffer (or none) and pH
  buffer   water vs buffer without salt
  pH       same buffer and chloride, different pH (|ΔpH| >= 0.5)

Conditions are described by buffer_class, halide concentration (rounded) and
pH. A missing pH is taken as 'unbuffered' for water/saline and as the nominal
7.4 for PBS-type media, and pairs whose pH is unknown on either side are left
out of the pH contrast. One value per complex and contrast (median over pairs).

Usage:
    python analyze_conditions.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # repository root


def conditions(path=ROOT / 'data' / 'MetalLipoDB.csv'):
    d = pd.read_csv(path, low_memory=False)
    d = d[d['value_relation'].isin(['=', '~'])].copy()
    d['ci'] = d['counterion'].fillna('none')
    d['cl'] = pd.to_numeric(d['halide_conc_mM'], errors='coerce').fillna(0).round(-1)
    d['buf'] = d['buffer_class'].fillna('absent')
    ph = pd.to_numeric(d['pH'], errors='coerce')
    pbs = d['aqueous_phase'].astype(str).str.contains('PBS', case=False)
    d['ph'] = ph.where(ph.notna(), np.where(pbs, 7.4, np.nan))
    d['charge'] = pd.to_numeric(d['complex_charge'], errors='coerce')
    d['mcl'] = d['smiles_complex'].str.contains(r'<-\[Cl-\]|\[Cl-\]->', regex=True)
    return d.groupby(['doi', 'smiles_complex', 'ci', 'buf', 'cl', 'ph'], dropna=False).agg(
        v=('value', 'mean'), charge=('charge', 'first'), mcl=('mcl', 'first'),
        medium=('aqueous_phase', 'first')).reset_index()


def contrast(a, b):
    """Name of the single factor in which conditions a and b differ, else None."""
    same_buf, same_cl = a['buf'] == b['buf'], a['cl'] == b['cl']
    ph_known = pd.notna(a['ph']) and pd.notna(b['ph'])
    same_ph = (pd.isna(a['ph']) and pd.isna(b['ph'])) or (ph_known and abs(a['ph'] - b['ph']) < 0.5)
    if same_buf and same_ph and {a['cl'], b['cl']} <= {0} | set(range(100, 160, 10)) and a['cl'] != b['cl'] \
            and min(a['cl'], b['cl']) == 0:
        return 'salt'
    if same_cl and a['cl'] == 0 and {a['buf'], b['buf']} & {'absent'} and a['buf'] != b['buf']:
        return 'buffer'
    if same_buf and same_cl and ph_known and abs(a['ph'] - b['ph']) >= 0.5:
        return 'pH'
    return None


def pairs(c):
    rows = []
    for (doi, smi, ci), g in c.groupby(['doi', 'smiles_complex', 'ci']):
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            kind = contrast(a, b)
            if kind is None:
                continue
            # orient: salt -> (with salt) - (without); buffer -> buffer - water; pH -> high - low
            if kind == 'salt':
                lo, hi = (a, b) if a['cl'] < b['cl'] else (b, a)
            elif kind == 'buffer':
                lo, hi = (a, b) if a['buf'] == 'absent' else (b, a)
            else:
                lo, hi = (a, b) if a['ph'] < b['ph'] else (b, a)
            rows.append({'doi': doi, 'smiles': smi, 'contrast': kind, 'shift': hi['v'] - lo['v'],
                         'charge': a['charge'], 'mcl': a['mcl'], 'from': lo['medium'], 'to': hi['medium'],
                         'dph': (hi['ph'] - lo['ph']) if kind == 'pH' else np.nan})
    p = pd.DataFrame(rows)
    p['class'] = np.select([p['charge'] > 0, p['charge'] < 0], ['cationic', 'anionic'], 'neutral')
    return p.groupby(['doi', 'smiles', 'contrast', 'class', 'mcl'], as_index=False).agg(
        shift=('shift', 'median'), dph=('dph', 'median'), frm=('from', 'first'), to=('to', 'first'))


def main():
    p = pairs(conditions())
    p.to_csv(ROOT / 'figures' / 'condition_contrasts.csv', index=False)
    pd.set_option('display.width', 200)
    s = p.groupby(['contrast', 'class', 'mcl']).agg(complexes=('shift', 'size'), articles=('doi', 'nunique'),
                                                    median=('shift', 'median'),
                                                    positive=('shift', lambda x: int((x > 0).sum())))
    print(s.round(2).to_string())


if __name__ == '__main__':
    main()
