"""
MetalLipoDB — classify_metal_from_ligands.py
============================================
How much does the ligand representation alone know about the metal?

A classifier sees only the Morgan count fingerprint of the ligands (the same
representation as in t-SNE and in the ligand block of the models) and predicts
the metal. If it succeeds well above chance, a ligand-only regression model
knows the metal implicitly, and only leave-one-metal-out tests transfer to an
unseen metal.

One row per complex and article. Classes are the metals with >= 100 records
(as in the LOMO split); heterometallic and rarer metals are left out.
With --by group the label is the periodic group (7-11) and the lighter
congeners (Mn, Os, Co, Ni, Pd, Ag) are included.
Splits: grouped 5-fold CV by article, and by connected components of articles
and ligand sets (the same complex never on both sides).

Usage:
    python classify_metal_from_ligands.py [--data MetalLipoDB.csv] [--out_dir figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA = ROOT / 'data' / 'MetalLipoDB.csv'
OUT_DIR = ROOT / 'figures'
METALS = ['Ru', 'Pt', 'Ir', 'Au', 'Cu', 'Fe', 'Tc', 'Rh', 'Re']
# periodic groups 7-11; lighter congeners join here, since the class is pooled
GROUPS = {'Mn': 'group 7', 'Tc': 'group 7', 'Re': 'group 7',
          'Fe': 'group 8', 'Ru': 'group 8', 'Os': 'group 8',
          'Co': 'group 9', 'Rh': 'group 9', 'Ir': 'group 9',
          'Ni': 'group 10', 'Pd': 'group 10', 'Pt': 'group 10',
          'Cu': 'group 11', 'Ag': 'group 11', 'Au': 'group 11'}
N_FOLDS = 5
SEED = 0


def load(path=DATA, by='metal'):
    d = pd.read_csv(path)
    allowed = GROUPS if by == 'group' else METALS
    d = d[d['metal'].isin(allowed) & d['smiles_ligands'].notna()].copy()
    d['label'] = d['metal'].map(GROUPS) if by == 'group' else d['metal']
    d = d.drop_duplicates(['doi', 'smiles_ligands', 'metal']).reset_index(drop=True)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, keep = [], []
    for i, smi in enumerate(d['smiles_ligands']):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            fps.append(gen.GetCountFingerprintAsNumPy(m))
            keep.append(i)
    d = d.iloc[keep].reset_index(drop=True)
    return d, np.log1p(np.vstack(fps).astype(np.float32))


def component_groups(d):
    """Articles and ligand sets linked into components: a complex measured in
    two articles pulls both articles into one group."""
    a = pd.factorize(d['doi'])[0]
    l = pd.factorize(d['smiles_ligands'])[0] + a.max() + 1
    n = l.max() + 1
    g = coo_matrix((np.ones(len(d)), (a, l)), shape=(n, n))
    return connected_components(g, directed=False)[1][a]


def evaluate(X, y, groups, model):
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(N_FOLDS).split(X, y, groups):
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def main(data, out_dir, by='metal'):
    d, X = load(data, by)
    y = d['label'].values
    counts = pd.Series(y).value_counts()
    majority = counts.max() / counts.sum()
    print(f'{len(d)} complex–article rows, {d["doi"].nunique()} articles')
    print('classes:', counts.to_dict())
    print(f'chance: balanced accuracy {1 / counts.size:.2f}, majority class {majority:.2f}')

    splits = {'by article': d['doi'].values, 'by article + ligand set': component_groups(d)}
    models = {
        'kNN (k = 5, cosine)': lambda: KNeighborsClassifier(5, metric='cosine'),
        'LightGBM': lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                           class_weight='balanced', random_state=SEED,
                                           verbose=-1),
    }
    results, keep_pred = [], None
    for sname, groups in splits.items():
        for mname, make in models.items():
            p = evaluate(X, y, groups, make())
            results.append((sname, mname, (p == y).mean(),
                            balanced_accuracy_score(y, p), f1_score(y, p, average='macro')))
            if sname == 'by article + ligand set' and mname == 'LightGBM':
                keep_pred = p
    r = pd.DataFrame(results, columns=['split', 'model', 'accuracy', 'balanced accuracy', 'macro F1'])
    print(r.round(3).to_string(index=False))

    order = counts.index.tolist()
    if by == 'group':
        order = sorted(order, key=lambda g: int(g.split()[1]))
    short = [o.split()[1] if by == 'group' else o for o in order]
    cm = confusion_matrix(y, keep_pred, labels=order).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(order)):
        for j in range(len(order)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f'{100 * cm[i, j]:.0f}', ha='center', va='center', fontsize=6.5,
                        color='white' if cm[i, j] > 0.5 else 'black')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(short)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{k} ({counts[m]})' for k, m in zip(short, order)])
    ax.set_xlabel(f'predicted {by}')
    ax.set_ylabel(f'true {by}')
    fig.colorbar(im, ax=ax, fraction=0.046, label='fraction of true class')
    plt.tight_layout()
    out = Path(out_dir) / ('figure_metal_from_ligands_si' if by == 'metal'
                           else 'figure_group_from_ligands_si')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=600)
    plt.close(fig)
    recall = pd.Series(np.diag(cm), index=order)
    print(f'recall per {by} (LightGBM, by article + ligand set):', recall.round(2).to_dict())
    print(f'saved: {out}.png / .pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(DATA))
    p.add_argument('--out_dir', default=str(OUT_DIR))
    p.add_argument('--by', choices=['metal', 'group'], default='metal')
    a = p.parse_args()
    main(a.data, a.out_dir, a.by)
