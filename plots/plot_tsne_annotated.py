"""
MetalLipoDB — plot_tsne_annotated.py
====================================
Figure 3 with examples: the t-SNE map of metallipo_tsne_pipeline.py (same
embedding, same colours) with callouts showing a typical complex of the
densest same-metal region for several metals. The complex is chosen
automatically: the point with the most same-metal neighbours in the map,
then the smallest complex among its nearest same-metal neighbours (for
readability). Structures are drawn with metal2d.
Output: figures/figure_tsne_annotated.png / .pdf
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import metal2d
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.spatial import cKDTree

RDLogger.DisableLog('rdApp.*')
ROOT = Path(__file__).resolve().parent.parent  # repository root
TOP_N = 9
METAL_ORDER = ['Ru', 'Pt', 'Ir', 'Au', 'Cu', 'Tc', 'Rh', 'Re', 'Os']   # as in metallipo_tsne_pipeline.py
# (key, metal, SMILES or None for the automatic pick); the box shows the metal only
# boxes moved up (-1) or down (+1) against the default top-to-bottom order
SIDE = {'Pt': -1, 'Tc': -1, 'Au': -1, 'Ru': 1, 'Ir': 1, 'Ru (dip)': 1}   # left / right column
BOX_ORDER = {'Tc': -1}
RU_DIP = 'c1ccc(-c2cc[n]3->[Ru+2]45(<-[n]6cccnc6-c6nccc[n]->46)(<-[n]4ccc(-c6ccccc6)c6ccc2c3c64)<-[n]2ccc(-c3ccccc3)c3ccc4c(-c6ccccc6)cc[n]->5c4c32)cc1'   # [Ru(dip)2(bpm)]2+
CALLOUTS = [('Ru', 'Ru', None), ('Pt', 'Pt', None), ('Ir', 'Ir', None), ('Au', 'Au', None),
            ('Ru (dip)', 'Ru', RU_DIP), ('Tc', 'Tc', None)]
RADIUS = 6.0     # neighbourhood in t-SNE units for the density
N_CAND = 25      # nearest same-metal points considered for the example
NO_STEREO = {'Pt', 'Tc'}
# hand-picked examples from the same dense region, where the automatic pick is less typical
PREFERRED = {
    'Au': 'C(#[C-]->[Au+]<-[P]12CN3CN(CN(C3)C1)C2)c1ccccc1',   # [Au(C#CPh)(PTA)]
    'Pt': 'CC(=O)[O-]->[Pt+4](<-[NH3])(<-[Cl-])(<-[Cl-])(<-[NH2]C1CCCCC1)<-[O-]C(C)=O',   # satraplatin
    'Tc': 'COc1cc(CNC(=O)[c]23->[Tc+]456(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[cH]([cH]->4[cH-]->52)[cH]->63)ccc1O',   # CpTc(CO)3
}   # wedges clutter these drawings: take an example without stereocentres


def render(smiles, size=(360, 300)):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    p = metal2d.prepare_for_drawing(metal2d.depict(mol))
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    metal2d.style_options(o)
    o.padding = 0.05
    o.fixedBondLength = 22
    o.minFontSize = 13
    d.DrawMolecule(rdMolDraw2D.PrepareMolForDrawing(p, kekulize=False, wedgeBonds=False))
    d.FinishDrawing()
    return np.asarray(Image.open(io.BytesIO(d.GetDrawingText())))


def pick(e, metal, smiles=None):
    smiles = smiles or PREFERRED.get(metal)
    if smiles:
        # identical fingerprints form a grid in the map: take its central point
        x = e[e['smiles_complex'].eq(smiles)]
        c = x[['tsne_x', 'tsne_y']].median().to_numpy()
        return x.iloc[np.hypot(*(x[['tsne_x', 'tsne_y']].to_numpy() - c).T).argmin()]
    xy = e[['tsne_x', 'tsne_y']].to_numpy()
    same = (e['metal'] == metal).to_numpy()
    tree = cKDTree(xy[same])
    counts = np.array([len(tree.query_ball_point(p, RADIUS)) for p in xy[same]])
    centre = xy[same][counts.argmax()]
    sub = e[same].assign(dist=np.hypot(*(xy[same] - centre).T))
    cand = sub.nsmallest(N_CAND, 'dist').drop_duplicates('smiles_complex')
    cand = cand.assign(size=cand['smiles_complex'].map(
        lambda s: Chem.MolFromSmiles(s, sanitize=False).GetNumAtoms()))
    cand = cand[cand['size'] >= 8] if (cand['size'] >= 8).any() else cand
    if metal in NO_STEREO:
        flat = cand[~cand['smiles_complex'].str.contains('@')]
        cand = flat if len(flat) else cand
    return cand.sort_values(['size', 'dist']).iloc[len(cand) // 3]   # small but not the smallest


def main():
    e = pd.read_csv(ROOT / 'figures' / 'tsne' / 'embedding_and_clusters.csv')
    top = e['metal'].value_counts().index[:TOP_N]
    top = sorted(top, key=lambda m: METAL_ORDER.index(m) if m in METAL_ORDER else len(METAL_ORDER))
    tab = plt.get_cmap('tab10').colors
    palette = [c for i, c in enumerate(tab) if i != 7]
    colour = {m: palette[i % len(palette)] for i, m in enumerate(top)}

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    oth = ~e['metal'].isin(top)
    ax.scatter(e.loc[oth, 'tsne_x'], e.loc[oth, 'tsne_y'], s=3, c='#d8d8d8', lw=0, label='other')
    for m in top:
        s = e['metal'] == m
        ax.scatter(e.loc[s, 'tsne_x'], e.loc[s, 'tsne_y'], s=4, color=colour[m], lw=0, alpha=0.8,
                   label=f'{m} ({s.sum()})')
    ax.legend(fontsize=7, markerscale=2.5, loc='lower center', ncol=5, frameon=False,
              bbox_to_anchor=(0.5, -0.13))
    x0, x1 = e['tsne_x'].min(), e['tsne_x'].max()
    y0, y1 = e['tsne_y'].min(), e['tsne_y'].max()
    w, h = x1 - x0, y1 - y0
    ax.set_xlim(x0 - 0.42 * w, x1 + 0.42 * w)
    ax.set_ylim(y0 - 0.05 * h, y1 + 0.05 * h)
    picks = [((lab, m), pick(e, m, smi)) for lab, m, smi in CALLOUTS]
    rows = []
    for side in (-1, 1):
        group = [p for p in picks if SIDE[p[0][0]] == side]
        group = sorted(group, key=lambda p: -p[1]['tsne_y'])   # top to bottom: lines do not cross
        group = sorted(group, key=lambda p: BOX_ORDER.get(p[0][0], 0))   # manual tweaks, stable
        ys = np.linspace(y1 - 0.12 * h, y0 + 0.12 * h, len(group))
        for ((lab, m), r), yb in zip(group, ys):
            xb = x0 - 0.22 * w if side < 0 else x1 + 0.22 * w
            img = OffsetImage(render(r['smiles_complex']), zoom=0.26)
            ab = AnnotationBbox(img, (r['tsne_x'], r['tsne_y']), xybox=(xb, yb), xycoords='data',
                                boxcoords='data', frameon=True, pad=0.1,
                                bboxprops=dict(edgecolor=colour[m], lw=1.4, boxstyle='round,pad=0.1'),
                                arrowprops=dict(arrowstyle='-', color=colour[m], lw=0.9,
                                                relpos=(1, 0.5) if side < 0 else (0, 0.5)))   # from the inner edge
            ax.add_artist(ab)
            ax.text(xb, yb + 0.155 * h, m, ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=colour[m])
            ax.plot(r['tsne_x'], r['tsne_y'], 'o', ms=5, mfc='none', mec='black', mew=0.8)
            rows.append((lab, r['smiles_complex'], r['abbreviation_in_the_article'], r['doi']))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    out = ROOT / 'figures' / 'figure_tsne_annotated'
    fig.savefig(out.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    pd.DataFrame(rows, columns=['metal', 'smiles', 'abbreviation', 'doi']).to_csv(
        ROOT / 'figures' / 'tsne_annotated_examples.csv', index=False)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
