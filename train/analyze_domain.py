"""
MetalLipoDB modeling — analyze_domain.py
========================================
How close is each test row to its training fold, and does the error follow?

For every out-of-fold prediction of a model, two quantities are computed against
the training part of the same fold and split:

  max_sim      count Tanimoto, sum(min) / sum(max), of the ligand set to its
               nearest training neighbour; unfolded Morgan r = 2 counts on
               canonical_ligands, so there are no bit collisions and repeated
               fragments count (bpy2Cl2 vs bpy3 is 0.69, not 1)
  doi_in_train whether the training part holds rows from the same article

Row errors are then summarised by similarity bins and by doi_in_train, per split.
A low error that exists only where the article is in training is interpolation
within a series; the drop of error with similarity describes the applicability
domain. Only saved predictions are used: nothing is trained.

Usage:
    python analyze_domain.py [--model lightgbm]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'MetalLipoDB_modeling.csv'
MANIFEST = HERE / 'data' / 'fold_manifest.csv'
OOF_DIR = HERE / 'results' / 'oof_predictions'
OUT_METRICS = HERE / 'results' / 'metrics'
OUT_FIG = HERE / 'results' / 'figures'
SIM_BINS = [0, 0.3, 0.5, 0.7, 0.9, 1.0001]
SIM_LABELS = ['<0.3', '0.3–0.5', '0.5–0.7', '0.7–0.9', '≥0.9']


def count_fingerprints(smiles):
    gen = GetMorganGenerator(radius=2)
    cache = {s: gen.GetSparseCountFingerprint(Chem.MolFromSmiles(s)) for s in pd.unique(smiles)}
    return [cache[s] for s in smiles]


def nearest_similarity(fps_test, fps_train):
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, fps_train)) for f in fps_test])


def main(model):
    data = pd.read_csv(DATA, low_memory=False).merge(
        pd.read_csv(MANIFEST)[['row_id', 'SMILES_fold', 'DOI_fold']], on='row_id', validate='one_to_one')
    fps = count_fingerprints(data['canonical_ligands'].tolist())
    doi = data['doi'].str.lower().fillna('').to_numpy()
    oof = pd.read_csv(OOF_DIR / f'{model}_oof.csv')
    index = pd.Series(np.arange(len(data)), index=data['row_id'])

    rows = []
    for split in [s for s in ('SMILES', 'DOI') if s in set(oof['split'])]:  # grouped CV splits only
        folds = data[f'{split}_fold'].to_numpy()
        for fold in np.unique(folds):
            te, tr = np.where(folds == fold)[0], np.where(folds != fold)[0]
            sim = nearest_similarity([fps[i] for i in te], [fps[i] for i in tr])
            train_dois = set(doi[tr])
            rows.append(pd.DataFrame({'row_id': data['row_id'].iloc[te].to_numpy(), 'split': split,
                                      'max_sim': sim,
                                      'doi_in_train': [d in train_dois and d != '' for d in doi[te]]}))
    dom = pd.concat(rows).merge(oof[['row_id', 'split', 'absolute_error']], on=['row_id', 'split'])
    dom['sim_bin'] = pd.cut(dom['max_sim'], SIM_BINS, labels=SIM_LABELS, right=False)

    print(f'model: {model}')
    for split, g in dom.groupby('split'):
        print(f'\n== {split}: {len(g)} rows | median nearest similarity {g["max_sim"].median():.2f} '
              f'| article in training for {g["doi_in_train"].mean():.0%} of rows')
        t = g.groupby('sim_bin', observed=False)['absolute_error'].agg(['size', 'mean']).round(3)
        t.columns = ['rows', 'MAE']
        print(t.to_string())
        t2 = g.groupby('doi_in_train')['absolute_error'].agg(['size', 'mean']).round(3)
        t2.columns = ['rows', 'MAE']
        print(t2.rename(index={True: 'article in training', False: 'article not in training'}).to_string())
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    dom.to_csv(OUT_METRICS / f'{model}_domain.csv', index=False)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = {'SMILES': '#2E6E8E', 'DOI': '#C1442F'}
    for split, g in dom.groupby('split'):
        a.hist(g['max_sim'], bins=np.linspace(0, 1, 41), alpha=0.55, color=colors.get(split),
               label=f'{split} split', density=True)
        m = g.groupby('sim_bin', observed=False)['absolute_error'].mean()
        n = g.groupby('sim_bin', observed=False).size()
        keep = n >= 20
        b.plot(np.arange(len(m))[keep.values], m[keep].values, 'o-', color=colors.get(split),
               label=f'{split} split')
    a.set_xlabel('nearest-neighbour count Tanimoto to training fold')
    a.set_ylabel('density')
    a.legend(frameon=False)
    b.set_xticks(range(len(SIM_LABELS)))
    b.set_xticklabels(SIM_LABELS)
    b.set_xlabel('nearest-neighbour similarity')
    b.set_ylabel('MAE (bins with ≥ 20 rows)')
    # mean |logP difference| between two articles for the same complex and
    # counterion, each complex weighted equally (94 complexes): what a predictor
    # that knew another laboratory's value exactly would reach on the DOI split
    b.axhspan(0.44, 0.69, color='#dddddd', alpha=0.6, lw=0)
    b.axhline(0.56, color='#888888', ls=':', lw=1)
    b.text(0.02, 0.575, 'same complex, two articles: 0.56 (0.44–0.69)', fontsize=7,
           color='#555555', transform=b.get_yaxis_transform())
    b.legend(frameon=False)
    for ax in (a, b):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    out = OUT_FIG / f'{model}_domain.png'
    fig.savefig(out, dpi=200)
    print(f'\nsaved: {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='lightgbm')
    main(p.parse_args().model)
