"""
MetalLipoDB modeling — helpers shared by the scripts: paths, SMILES handling,
metrics and loading of the modeling table with its folds.
"""

import hashlib
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import COMMON_CATEGORICAL, COMMON_NUMERIC, METAL_GROUP, METALS

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
# release table (Zenodo, https://doi.org/10.5281/zenodo.22708388), 5190 rows
SOURCE_PATH = HERE.parent / "data" / "MetalLipoDB.csv"
DATA_DIR = HERE / "data"
MODELING_PATH = DATA_DIR / "MetalLipoDB_modeling.csv"
FOLD_PATH = DATA_DIR / "fold_manifest.csv"
DICTIONARY_PATH = DATA_DIR / "feature_dictionary.csv"
RESULTS_DIR = HERE / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
OOF_DIR = RESULTS_DIR / "oof_predictions"
FIGURE_DIR = RESULTS_DIR / "figures"
for _d in (DATA_DIR, METRICS_DIR, OOF_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SPLITS = [("SMILES", "SMILES_fold"), ("DOI", "DOI_fold")]
N_FOLDS = 10
# leave-one-metal-out and leave-one-group-out: every metal (group) with at least
# MIN_HELD_OUT single-metal rows is held out once; rows containing it,
# heterometallic ones included, and every article with a held-out row are
# removed from training, so a series cannot be recognised through analogues of
# another metal
METAL_SPLITS = ["LOMO", "LOGO"]
MIN_HELD_OUT = 20

METAL_NUMBERS = {Chem.GetPeriodicTable().GetAtomicNumber(s) for s in METALS}


def environment(**versions):
    print("Python:", sys.executable)
    print("Platform:", platform.platform())
    print("RDKit:", rdBase.rdkitVersion)
    for name, version in versions.items():
        print(f"{name}:", version)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── SMILES ────────────────────────────────────────────────────────────────────
def parsed(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def metal_atoms(mol):
    return [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in METAL_NUMBERS]


def canonical(smiles):
    return Chem.MolToSmiles(parsed(smiles), canonical=True)


def remove_all_metals(smiles):
    mol = parsed(smiles)
    indices = [atom.GetIdx() for atom in metal_atoms(mol)]
    if not indices:
        raise ValueError("No supported metal atom found")
    editable = Chem.RWMol(mol)
    for index in sorted(indices, reverse=True):
        editable.RemoveAtom(index)
    ligand_mol = editable.GetMol()
    ligand_mol.UpdatePropertyCache(strict=False)
    result = Chem.MolToSmiles(ligand_mol, canonical=True)
    if not result:
        raise ValueError("No ligand atoms remain")
    return result


def single_coordination_smiles(smiles):
    mol = Chem.Mol(parsed(smiles))
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DATIVE:
            bond.SetBondType(Chem.BondType.SINGLE)
    mol.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(mol, canonical=True)


# ── data and metrics ──────────────────────────────────────────────────────────
def load_modeling_data():
    """Modeling table with the frozen folds attached; one row per measurement."""
    data = pd.read_csv(MODELING_PATH, low_memory=False)
    manifest = pd.read_csv(FOLD_PATH)
    data = data.merge(manifest[["row_id", "SMILES_fold", "DOI_fold"]],
                      on="row_id", how="left", validate="one_to_one")
    assert len(data) == len(manifest) and data["row_id"].is_unique
    assert data[["SMILES_fold", "DOI_fold"]].notna().all().all()
    return data


def evaluation_folds(data, split, keep_articles=False):
    """Yield (fold, train_idx, test_idx) for SMILES, DOI, LOMO or LOGO.

    keep_articles (LOMO/LOGO): do not remove from training the articles with a
    held-out row, only the complexes containing the held-out metal (SI comparison).
    """
    if split in dict(SPLITS):
        column = dict(SPLITS)[split]
        for fold in range(1, N_FOLDS + 1):
            yield (fold, np.flatnonzero(data[column].ne(fold).to_numpy()),
                   np.flatnonzero(data[column].eq(fold).to_numpy()))
        return
    metals = data["metal"].str.split("-")
    if split == "LOMO":
        unit_of = {m: m for m in METALS}
    elif split == "LOGO":
        unit_of = {m: f"group {g}" for m, g in METAL_GROUP.items()}
    else:
        raise ValueError(split)
    units = metals.map(lambda ms: {unit_of[m] for m in ms})
    single = units.map(len).eq(1)
    counts = units[single].map(lambda u: next(iter(u))).value_counts()
    doi = data["doi"].str.lower().fillna("").to_numpy()
    for unit in counts[counts >= MIN_HELD_OUT].index:
        contains = units.map(lambda u: unit in u).to_numpy()
        test = contains & single.to_numpy()
        held_articles = np.zeros(len(doi), bool) if keep_articles else np.isin(doi, list(set(doi[test]) - {""}))
        yield unit, np.flatnonzero(~contains & ~held_articles), np.flatnonzero(test)


# ── shared by the neural models ────────────────────────────────────────────────
REPRESENTATIONS = ["DOT", "DATIVE", "SINGLE"]
VALIDATION_SHARE = 0.1


def single_graph(smiles):
    mol = Chem.Mol(parsed(smiles))
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DATIVE:
            bond.SetBondType(Chem.BondType.SINGLE)
    mol.UpdatePropertyCache(strict=False)
    return mol


def molecules(data, representation):
    if representation == "DOT":
        return [parsed(s) for s in data["canonical_ligands"]]
    if representation == "DATIVE":
        return [parsed(s) for s in data["canonical_complex_dative"]]
    return [single_graph(s) for s in data["canonical_complex_dative"]]


def fold_seed(fold):
    return 42 + fold if isinstance(fold, int) else 42 + sum(map(ord, fold))


def split_validation(data, split, train_idx, seed):
    """Early-stopping rows taken out of the training part, grouped like the split."""
    key = data["canonical_ligands"] if split == "SMILES" else data["doi"].str.lower().fillna(data["row_id"])
    inner = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SHARE, random_state=seed)
    fit, validation = next(inner.split(train_idx, groups=key.to_numpy()[train_idx]))
    return train_idx[fit], train_idx[validation]


def shared_features(data, train_idx, *other_idx):
    """Fold-local median imputation, scaling and one-hot encoding."""
    numeric = data[COMMON_NUMERIC].apply(pd.to_numeric, errors="coerce")
    categorical = data[COMMON_CATEGORICAL].fillna("MISSING").astype(str)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    scaler.fit(imputer.fit_transform(numeric.iloc[train_idx]))
    encoder.fit(categorical.iloc[train_idx])
    return [np.hstack([scaler.transform(imputer.transform(numeric.iloc[idx])),
                       encoder.transform(categorical.iloc[idx])]).astype(np.float32)
            for idx in (train_idx, *other_idx)]


def regression_metrics(y_true, prediction):
    y_true, prediction = np.asarray(y_true), np.asarray(prediction)
    return {
        "mae": mean_absolute_error(y_true, prediction),
        "rmse": mean_squared_error(y_true, prediction) ** 0.5,
        "r2": r2_score(y_true, prediction),
        "median_ae": np.median(np.abs(y_true - prediction)),
    }
