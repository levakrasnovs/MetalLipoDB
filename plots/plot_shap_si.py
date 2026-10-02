"""
MetalLipoDB — plot_shap_si.py
=============================
SI figure: SHAP analysis of LightGBM in DOI splitting (out-of-fold, the same
frozen folds and hyperparameters as train/train_lightgbm.py).
  a) share of the total mean |SHAP| for each block of features
  b) beeswarm of the 15 most important features outside the fingerprint
Output: figures/figure_shap.png / .pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent.parent  # repository root
sys.path.insert(0, str(ROOT / 'train'))
from common import evaluation_folds, load_modeling_data  # noqa: E402
from features import COMMON_CATEGORICAL, COMMON_NUMERIC, feature_block  # noqa: E402
from train_lightgbm import FINGERPRINT_COLUMN, PARAMS, build_features, prepare_fold  # noqa: E402
from plot_figures_metallipodb import C_MAIN, RCPARAMS_STANDARD  # noqa: E402

TOP = 15
NAMES = {'rdkit_MolLogP': 'Crippen logP of ligands', 'rdkit_fr_benzene': 'benzene rings in ligands',
         'counterion_n_fragments': 'number of counterions', 'complex_charge_numeric': 'charge of complex',
         'rdkit_MinEStateIndex': 'min E-state index', 'rdkit_MolWt': 'molecular weight of ligands',
         'halide_conc_mM_value': 'halide concentration', 'metal_period_mean': 'period of metal',
         'rdkit_MaxEStateIndex': 'max E-state index', 'counterion_abs_charge_sum': 'total charge of counterions',
         'metal_oxidation_state_mean': 'oxidation state of metal', 'rdkit_FractionCSP3': 'fraction of sp3 carbons',
         'rdkit_TPSA': 'TPSA of ligands', 'counterion_MW': 'molecular weight of counterion',
         'rdkit_fr_COO': 'carboxylic groups in ligands', 'buffer_class_group': 'buffer',
         'donor_bonds_C': 'M–C bonds', 'counterion_MolLogP': 'Crippen logP of counterion',
         'metal_group_mean': 'group of metal', 'rdkit_NumHDonors': 'H-bond donors of ligands'}
BLOCK_NAMES = {'ligands: fingerprint': 'ligand fingerprint', 'ligands: descriptors': 'ligand descriptors',
               'ion pair': 'charge and counterion', 'metal': 'metal', 'conditions': 'measurement conditions',
               'coordination': 'coordination sphere'}


def compute():
    data = load_modeling_data()
    features, columns, morgan = build_features(data, FINGERPRINT_COLUMN['ligands'])
    y = pd.to_numeric(data['value']).to_numpy(float)
    values, inputs = [], []
    for fold, train_idx, test_idx in evaluation_folds(data, 'DOI'):
        X_train, X_test = prepare_fold(features, columns, morgan + COMMON_NUMERIC, train_idx, test_idx)
        model = LGBMRegressor(**PARAMS, random_state=42 + fold - 1)
        model.fit(X_train, y[train_idx], categorical_feature=COMMON_CATEGORICAL)
        values.append(pd.DataFrame(shap.TreeExplainer(model).shap_values(X_test), columns=columns))
        x = X_test.copy()
        for c in COMMON_CATEGORICAL:
            x[c] = x[c].cat.codes.replace(-1, np.nan)
        inputs.append(x.astype(float).reset_index(drop=True))
        print('fold', fold, flush=True)
    return pd.concat(values, ignore_index=True), pd.concat(inputs, ignore_index=True)


def main():
    sv, X = compute()
    mean_abs = sv.abs().mean()
    blocks = mean_abs.groupby(mean_abs.index.map(feature_block)).sum()
    blocks = (blocks / blocks.sum()).sort_values()
    print((100 * blocks).round(1).to_string())
    top = mean_abs[[c for c in sv.columns if not c.startswith('MorganBit_')]].nlargest(TOP).index[::-1]
    print(mean_abs[top[::-1]].round(3).to_string())

    plt.rcParams.update(RCPARAMS_STANDARD)
    fig = plt.figure(figsize=(7.4, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.9)
    a = fig.add_subplot(gs[0])
    a.barh([BLOCK_NAMES[b] for b in blocks.index], 100 * blocks.values, color=C_MAIN)
    for i, v in enumerate(blocks.values):
        a.text(100 * v + 1, i, f'{100 * v:.0f}%', va='center', fontsize=7)
    a.set(xlabel='share of total mean |SHAP| (%)', xlim=(0, 62))
    b = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    for i, c in enumerate(top):
        x, v = sv[c].to_numpy(), X[c].to_numpy()
        lo, hi = np.nanpercentile(v, [5, 95])
        colour = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        jitter = rng.normal(0, 0.12, len(x))
        b.scatter(x, i + jitter, c=colour, cmap='coolwarm', s=2, lw=0, alpha=0.6, rasterized=True)
    b.set_yticks(range(len(top)), [NAMES.get(c, c) for c in top])
    b.axvline(0, color='#555', lw=0.6)
    b.set(xlabel='SHAP value (effect on predicted logP)')
    sm = plt.cm.ScalarMappable(cmap='coolwarm')
    cb = fig.colorbar(sm, ax=b, fraction=0.04, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(['low', 'high'], fontsize=7)
    cb.set_label('feature value', fontsize=7)
    for k, ax in enumerate((a, b)):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.text(-0.02, 1.03, 'ab'[k] + ')', transform=ax.transAxes, fontsize=10, ha='right')
    out = ROOT / 'figures' / 'figure_shap'
    fig.savefig(out.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    main()
