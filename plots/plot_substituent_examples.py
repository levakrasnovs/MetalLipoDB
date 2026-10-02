"""
MetalLipoDB — plot_substituent_examples.py
==========================================
Supporting figure for Figure 5b: one example pair of complexes for each
substituent X (H -> X within one article, counterion and aqueous medium), as
found by the pair search of plot_substituent_attenuation.py.

For each X the three pairs whose measured shifts are closest to the median
shift of the group are taken, each from a different article, so the examples
are representative rather than extreme. One row per substituent; the figure
is split into two parts to fit the page. The atoms of X are
highlighted (atoms of the X complex outside the maximum common substructure of
the pair). Structures are drawn with metal2d.

Output: figures/figure_substituent_examples_{1,2}.png / .pdf and
figures/substituent_examples.csv (the pairs shown, with DOIs)

Usage:
    python plot_substituent_examples.py [--out_dir figures]
"""

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import metal2d
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

from plot_substituent_attenuation import DATA_LIPO, H_KEY, HANSCH_PI, RC, find_pairs

RDLogger.DisableLog('rdApp.*')
ROOT = Path(__file__).resolve().parent.parent  # repository root
OUT_DIR = ROOT / 'figures'
HIGHLIGHT = (0.76, 0.27, 0.18, 0.35)
K_EXAMPLES = 3          # pairs per substituent, from different articles
ROWS_PER_PAGE = 4


def pairs_with_structures(path=DATA_LIPO):
    """Same search as pairs_metallipo(), keeping the two complexes of each pair."""
    d = pd.read_csv(path)
    d = d[d['value_relation'].isin(['=', '~']) & (d['multinuclear'] != 'YES')].copy()
    d['cion'] = d['counterion'].fillna('none')
    ligands = d.dropna(subset=['smiles_ligands']).drop_duplicates('smiles_complex').set_index('smiles_complex')
    rows = []
    for (doi, cion, _), g in d.groupby(['doi', 'cion', 'medium_class'], dropna=False):
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
            dc = Crippen.MolLogP(lb) - Crippen.MolLogP(la)
            if abs(dc) < 1e-6:
                continue
            rows.append((doi, cion, va, vb, mols[a][2], mols[b][2], mols[a][1], mols[b][1]))
    return pd.DataFrame(rows, columns=['doi', 'counterion', 'var_a', 'var_b', 'smiles_a', 'smiles_b',
                                       'logp_a', 'logp_b'])


def choose_examples(p):
    rows = []
    for smi, label, _ in HANSCH_PI:
        fwd = p[(p['var_a'] == H_KEY) & (p['var_b'] == smi)]
        rev = p[(p['var_a'] == smi) & (p['var_b'] == H_KEY)]
        g = pd.concat([
            fwd.rename(columns={'smiles_a': 'smiles_h', 'smiles_b': 'smiles_x', 'logp_a': 'logp_h', 'logp_b': 'logp_x'}),
            rev.rename(columns={'smiles_b': 'smiles_h', 'smiles_a': 'smiles_x', 'logp_b': 'logp_h', 'logp_a': 'logp_x'})])
        g = g.assign(shift=g['logp_x'] - g['logp_h'])
        med = g['shift'].median()
        g = g.assign(dist=(g['shift'] - med).abs(),
                     size=g['smiles_x'].map(lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms()))
        picked = []
        for _, cand in g.sort_values('dist').iterrows():   # closest to the median, one per article
            if cand['doi'] in {c['doi'] for c in picked}:
                continue
            picked.append(cand)
            if len(picked) == K_EXAMPLES:
                break
        for rank, best in enumerate(picked, 1):
            rows.append({'substituent': label, 'rank': rank, 'n_pairs': len(g), 'median_shift': med, **best[
                ['doi', 'counterion', 'smiles_h', 'smiles_x', 'logp_h', 'logp_x', 'shift']].to_dict()})
    return pd.DataFrame(rows)


def substituent_atoms(smiles_h, smiles_x):
    """For H -> X the H complex is a substructure of the X complex; the atoms of
    the X complex outside that match are X. The maximum common substructure is
    only a fallback."""
    mh, mx = Chem.MolFromSmiles(smiles_h), Chem.MolFromSmiles(smiles_x)
    core = mx.GetSubstructMatch(mh)
    if not core:
        res = rdFMCS.FindMCS([mh, mx], timeout=5, ringMatchesRingOnly=True,
                             bondCompare=rdFMCS.BondCompare.CompareOrderExact)
        core = mx.GetSubstructMatch(Chem.MolFromSmarts(res.smartsString)) if res.smartsString else ()
    out = [i for i in range(mx.GetNumAtoms()) if i not in core]
    return out if 0 < len(out) <= 8 else []  # never highlight most of the molecule


def render(smiles, highlight=(), size=(520, 380)):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    p = metal2d.prepare_for_drawing(metal2d.depict(mol))
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = d.drawOptions()
    metal2d.style_options(opts)
    opts.padding = 0.08
    p = rdMolDraw2D.PrepareMolForDrawing(p, kekulize=False, wedgeBonds=False)
    hl = [i for i in highlight if i < p.GetNumAtoms()]
    d.DrawMolecule(p, highlightAtoms=hl, highlightAtomColors={i: HIGHLIGHT for i in hl},
                   highlightBonds=[b.GetIdx() for b in p.GetBonds()
                                   if b.GetBeginAtomIdx() in hl and b.GetEndAtomIdx() in hl],
                   highlightBondColors={b.GetIdx(): HIGHLIGHT for b in p.GetBonds()
                                        if b.GetBeginAtomIdx() in hl and b.GetEndAtomIdx() in hl})
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def figure(ex, out_dir):
    plt.rcParams.update(RC)
    subs = list(dict.fromkeys(ex['substituent']))
    pages = [subs[i:i + ROWS_PER_PAGE] for i in range(0, len(subs), ROWS_PER_PAGE)]
    for part, page in enumerate(pages, 1):
        fig = plt.figure(figsize=(7.4, 1.95 * len(page)))
        outer = fig.add_gridspec(len(page), K_EXAMPLES, wspace=0.08, hspace=0.12)
        for row, sub in enumerate(page):
            for col, r in enumerate(ex[ex['substituent'].eq(sub)].itertuples()):
                cell = outer[row, col].subgridspec(2, 2, height_ratios=[1, 0.22], wspace=0.02, hspace=0.0)
                ah, ax_ = fig.add_subplot(cell[0, 0]), fig.add_subplot(cell[0, 1])
                at = fig.add_subplot(cell[1, :])
                ah.imshow(render(r.smiles_h, size=(420, 380)))
                ax_.imshow(render(r.smiles_x, substituent_atoms(r.smiles_h, r.smiles_x), size=(420, 380)))
                for a in (ah, ax_, at):
                    a.axis('off')
                ah.set_title(f'H, logP {r.logp_h:.2f}', fontsize=6.5, pad=1)
                ax_.set_title(f'{r.substituent}, logP {r.logp_x:.2f}', fontsize=6.5, pad=1)
                at.text(0.5, 1.0, f'ΔlogP {r.shift:+.2f} (median {r.median_shift:+.2f}, n = {r.n_pairs})'
                        f'\n{r.doi}', fontsize=6, va='top', ha='center', transform=at.transAxes)
        out = Path(out_dir) / f'figure_substituent_examples_{part}'
        fig.savefig(out.with_suffix('.png'), dpi=400, bbox_inches='tight')
        fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f'saved: {out}.png / .pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=str(OUT_DIR))
    out = Path(ap.parse_args().out_dir)
    ex = choose_examples(pairs_with_structures())
    ex.to_csv(out / 'substituent_examples.csv', index=False)
    print(ex[['substituent', 'rank', 'n_pairs', 'median_shift', 'shift', 'doi']].round(2).to_string(index=False))
    figure(ex, out)


if __name__ == '__main__':
    main()
