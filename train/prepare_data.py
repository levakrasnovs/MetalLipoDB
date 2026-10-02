"""
MetalLipoDB modeling — prepare_data.py
======================================
From the released MetalLipoDB.csv to the modeling table and the frozen folds
used by every model.

- keeps every exact measurement (value_relation "=" and a logP/logD label);
  frequent complexes are not collapsed: grouped folds keep repeated
  measurements of one complex on the same side, and a median would merge
  different media, counterions and labels
- row_id is the published record_id, so predictions map back to MetalLipoDB.csv
- computes the ligand descriptors, metal/coordination block, ion-pair block and
  condition columns; which of them enter the models is decided in features.py
- freezes two grouped 10-fold splits: by ligand set (SMILES) and by connected
  components of articles and complexes (DOI)

Imputation, scaling and category encoding stay fold-local in the model scripts.

Usage:
    python prepare_data.py            # reuses data/fold_manifest.csv if present
    python prepare_data.py --refold   # rebuilds the folds (GroupKFold orders groups
                                      # by canonical SMILES, so RDKit versions differ)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
import sys

from sklearn.model_selection import GroupKFold

from common import (DICTIONARY_PATH, FIGURE_DIR, FOLD_PATH, MODELING_PATH, N_FOLDS,
                    SOURCE_PATH, canonical, environment, metal_atoms, parsed,
                    remove_all_metals, sha256, single_coordination_smiles)
from features import (COMMON_CATEGORICAL, COMMON_NUMERIC, COUNTERION_ELEMENTS,
                      COUNTERION_FEATURES, METAL_EN, METAL_GROUP, METALS, RDKIT25,
                      feature_block)

DESCRIPTOR_FUNCTIONS = dict(Descriptors._descList)


# ── selection ─────────────────────────────────────────────────────────────────
def select_rows(raw):
    relation = raw["value_relation"].astype("string").fillna("=").str.strip().replace("", "=")
    valid_type = raw["type_authors"].isin(["logP", "logD"])
    print(pd.DataFrame({
        "quantity": ["all rows", "mononuclear", "multinuclear", "LogP", "LogD", ">", "<", "~"],
        "count": [len(raw), raw["multinuclear"].isna().sum(), raw["multinuclear"].notna().sum(),
                  raw["type_authors"].eq("logP").sum(), raw["type_authors"].eq("logD").sum(),
                  relation.eq(">").sum(), relation.eq("<").sum(), relation.eq("~").sum()],
    }).to_string(index=False))
    data = raw.loc[valid_type & relation.eq("=")].reset_index(drop=True)[raw.columns]
    assert data["record_id"].is_unique
    assert data["value_relation"].eq("=").all()
    assert data["type_authors"].isin(["logP", "logD"]).all()
    print(f"Exact measurements (modeling rows): {len(data):,} of {len(raw):,}")
    return data


def add_structures(data):
    data["domain"] = np.where(data["multinuclear"].isna(), "mono", "multi")
    data["row_id"] = data["record_id"]
    missing_mono = data["domain"].eq("mono") & (
        data["smiles_ligands"].isna() | data["smiles_ligands"].astype(str).str.strip().eq(""))
    assert not missing_mono.any(), "Every mononuclear row needs ligand SMILES."
    data["canonical_complex_dative"] = data["smiles_complex"].map(canonical)
    data["canonical_complex_single"] = data["smiles_complex"].map(single_coordination_smiles)
    # multinuclear rows have no curated ligand column: ligands are cut off the complex
    data["canonical_ligands"] = data["smiles_ligands"]
    multi = data["domain"].eq("multi")
    data.loc[multi, "canonical_ligands"] = data.loc[multi, "smiles_complex"].map(remove_all_metals)
    data["canonical_ligands"] = data["canonical_ligands"].map(canonical)
    for name, column in [("DOT ligand assemblies", "canonical_ligands"),
                         ("DATIVE complexes", "canonical_complex_dative"),
                         ("SINGLE complexes", "canonical_complex_single")]:
        print(f"Valid {name}:", data[column].notna().sum())
    return data


# ── descriptor blocks ─────────────────────────────────────────────────────────
def rdkit_record(smiles):
    mol = parsed(smiles)
    result = {}
    for name in RDKIT25:
        try:
            value = float(DESCRIPTOR_FUNCTIONS[name](mol))
            result[f"rdkit_{name}"] = value if np.isfinite(value) else np.nan
        except Exception:
            result[f"rdkit_{name}"] = np.nan
    return result


def coordination_record(smiles):
    mol = parsed(smiles)
    metals = metal_atoms(mol)
    if not metals:
        raise ValueError("Complex has no supported metal atom")
    metal_indices = {atom.GetIdx() for atom in metals}
    symbols = [atom.GetSymbol() for atom in metals]
    charges = np.asarray([atom.GetFormalCharge() for atom in metals], dtype=float)
    coordination = []
    donor_counts = {element: 0 for element in ["N", "O", "S", "P", "Cl", "C", "other"]}
    bridges = set()
    metal_metal_twice = 0
    for atom in metals:
        neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() not in metal_indices]
        coordination.append(len(neighbors))
        for neighbor in neighbors:
            symbol = neighbor.GetSymbol()
            donor_counts[symbol if symbol in donor_counts else "other"] += 1
            if sum(n.GetIdx() in metal_indices for n in neighbor.GetNeighbors()) >= 2:
                bridges.add(neighbor.GetIdx())
        metal_metal_twice += sum(n.GetIdx() in metal_indices for n in atom.GetNeighbors())
    coordination = np.asarray(coordination, dtype=float)
    result = {
        "metal_atomic_number_mean": float(np.mean([a.GetAtomicNum() for a in metals])),
        "metal_oxidation_state_mean": float(charges.mean()),
        "metal_valence_e_mean": float(np.mean([METAL_GROUP[a.GetSymbol()] - a.GetFormalCharge() for a in metals])),
        "metal_electronegativity_mean": float(np.mean([METAL_EN[a.GetSymbol()] for a in metals])),
        "metal_group_mean": float(np.mean([METAL_GROUP[a.GetSymbol()] for a in metals])),
        # period from the atomic number: 3d = 4, 4d = 5, 5d (and Gd) = 6
        "metal_period_mean": float(np.mean([4 if a.GetAtomicNum() <= 36 else 5 if a.GetAtomicNum() <= 54 else 6
                                            for a in metals])),
        # d-electron count = group - oxidation state; the metal formal charge in the
        # dative SMILES is the oxidation state (checked, incl. Fe(I) in Fe2Cp2 dimers)
        "metal_d_electrons_mean": float(np.mean([METAL_GROUP[a.GetSymbol()] - a.GetFormalCharge() for a in metals])),
        "n_metal_centers": float(len(metals)),
        "heterometallic": float(len(set(symbols)) > 1),
        "metal_formal_charge_sum": float(charges.sum()),
        "metal_formal_charge_range": float(charges.max() - charges.min()),
        "coordination_number_mean": float(coordination.mean()),
        "coordination_number_min": float(coordination.min()),
        "coordination_number_max": float(coordination.max()),
        "n_bridging_donor_atoms": float(len(bridges)),
        "n_metal_metal_bonds": float(metal_metal_twice / 2),
    }
    for symbol in METALS:
        result[f"metal_count_{symbol}"] = float(symbols.count(symbol))
    for element, count in donor_counts.items():
        result[f"donor_bonds_{element}"] = float(count)
    return result


def counterion_record(value):
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return {name: 0.0 for name in COUNTERION_FEATURES}
    mol = parsed(text)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    fragment_smiles = [Chem.MolToSmiles(f, canonical=True) for f in fragments]
    atoms = list(mol.GetAtoms())
    result = {
        "counterion_present": 1.0,
        "counterion_n_fragments": float(len(fragments)),
        "counterion_n_unique_fragments": float(len(set(fragment_smiles))),
        "counterion_formal_charge_sum": float(sum(a.GetFormalCharge() for a in atoms)),
        "counterion_abs_charge_sum": float(sum(abs(a.GetFormalCharge()) for a in atoms)),
        "counterion_MW": float(Descriptors.MolWt(mol)),
        "counterion_heavy_atoms": float(Descriptors.HeavyAtomCount(mol)),
        "counterion_TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "counterion_MolLogP": float(Crippen.MolLogP(mol)),
        "counterion_HBA": float(Lipinski.NumHAcceptors(mol)),
    }
    for element in COUNTERION_ELEMENTS:
        result[f"counterion_count_{element}"] = float(sum(a.GetSymbol() == element for a in atoms))
    return result


def add_descriptors(data):
    rdkit_cache = {s: rdkit_record(s) for s in data["canonical_ligands"].unique()}
    coord_cache = {s: coordination_record(s) for s in data["canonical_complex_dative"].unique()}
    keys = ["" if pd.isna(v) else str(v).strip() for v in data["counterion"]]
    counterion_cache = {k: counterion_record(k or np.nan) for k in dict.fromkeys(keys)}
    frames = [pd.DataFrame(data["canonical_ligands"].map(rdkit_cache).tolist()),
              pd.DataFrame(data["canonical_complex_dative"].map(coord_cache).tolist()),
              pd.DataFrame([counterion_cache[k] for k in keys])]
    return pd.concat([data.reset_index(drop=True)] + frames, axis=1)


def category(series):
    return series.astype("string").fillna("MISSING").str.strip().replace("", "MISSING")


def add_conditions(data):
    data["complex_charge_numeric"] = pd.to_numeric(data["complex_charge"], errors="coerce").fillna(0.0)
    data["is_logP"] = data["type_authors"].eq("logP").astype(float)
    data["pH_present"] = data["pH"].notna().astype(float)
    data["pH_value"] = pd.to_numeric(data["pH"], errors="coerce").fillna(7.0)
    data["temperature_present"] = data["temperature_C"].notna().astype(float)
    data["temperature_C_value"] = pd.to_numeric(data["temperature_C"], errors="coerce").fillna(0.0)
    data["halide_conc_present"] = data["halide_conc_mM"].notna().astype(float)
    # zero only where the article states there is no halide; a missing concentration
    # with Cl present, or with no halide type at all, is unknown, not zero
    data["halide_conc_mM_value"] = pd.to_numeric(data["halide_conc_mM"], errors="coerce")
    data.loc[data["halide_type"].eq("absent") & data["halide_conc_mM_value"].isna(),
             "halide_conc_mM_value"] = 0.0
    data["cosolvent_present"] = data["cosolvent"].notna().astype(float)
    data["detection_group"] = category(data["detection"])
    data["halide_type_group"] = category(data["halide_type"])
    data["buffer_class_group"] = category(data["buffer_class"])

    missing = [c for c in COMMON_NUMERIC + COMMON_CATEGORICAL if c not in data]
    assert not missing, missing
    assert not np.isinf(data[COMMON_NUMERIC].to_numpy(float)).any(), "infinite values"
    print("Shared numeric features:", len(COMMON_NUMERIC))
    print("Shared categorical features:", len(COMMON_CATEGORICAL))
    print("Rows requiring train-fold median imputation:",
          int(data[COMMON_NUMERIC].isna().any(axis=1).sum()))
    return data


# ── folds ─────────────────────────────────────────────────────────────────────
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def normalized_doi(value, row_id):
    if pd.isna(value) or not str(value).strip():
        return f"MISSING::{row_id}"
    return str(value).strip().lower()


def doi_complex_components(frame):
    """Articles linked through shared complexes: a complex measured in two
    articles puts both articles into one group."""
    uf = UnionFind(len(frame))
    doi_keys = [normalized_doi(v, r) for v, r in zip(frame["doi"], frame["row_id"])]
    for values in [doi_keys, frame["canonical_complex_dative"].tolist()]:
        first = {}
        for i, value in enumerate(values):
            if value in first:
                uf.union(i, first[value])
            else:
                first[value] = i
    roots = [uf.find(i) for i in range(len(frame))]
    mapping = {root: i for i, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([mapping[root] for root in roots])


def load_folds(data):
    """Reuse the frozen manifest: fold membership must not depend on the RDKit version."""
    manifest = pd.read_csv(FOLD_PATH)
    assert set(manifest["row_id"]) == set(data["row_id"]) and len(manifest) == len(data), \
        "rows changed; rebuild the folds with --refold"
    folds = data[["row_id", "canonical_ligands"]].merge(manifest, on="row_id", suffixes=("", "_frozen"))
    for fold in range(1, N_FOLDS + 1):
        test = folds["SMILES_fold"].eq(fold)
        assert not set(folds.loc[test, "canonical_ligands"]) & set(folds.loc[~test, "canonical_ligands"])
    print("Frozen folds reused:", FOLD_PATH)
    return manifest


def make_folds(data):
    smiles_groups = data["canonical_ligands"].to_numpy()
    doi_groups = doi_complex_components(data)
    manifest = data[["row_id", "canonical_ligands", "canonical_complex_dative", "doi", "domain"]].copy()
    for column, groups in [("SMILES_fold", smiles_groups), ("DOI_fold", doi_groups)]:
        assignments = np.zeros(len(data), dtype=int)
        for fold, (_, test_idx) in enumerate(GroupKFold(N_FOLDS).split(data, groups=groups), 1):
            assignments[test_idx] = fold
        manifest[column] = assignments
        for fold in range(1, N_FOLDS + 1):
            test = assignments == fold
            assert not set(groups[test]) & set(groups[~test])
    print("SMILES groups:", pd.Series(smiles_groups).nunique())
    print("DOI+complex components:", pd.Series(doi_groups).nunique())
    for column in ["SMILES_fold", "DOI_fold"]:
        print(manifest.groupby(column).size().rename("rows").to_frame().T.to_string())
    return manifest


# ── outputs ───────────────────────────────────────────────────────────────────
def write_dictionary():
    rows = [{"feature": f, "type": "numeric", "group": feature_block(f)} for f in COMMON_NUMERIC]
    rows += [{"feature": f, "type": "categorical", "group": feature_block(f)} for f in COMMON_CATEGORICAL]
    pd.DataFrame(rows).to_csv(DICTIONARY_PATH, index=False)


def overview(data):
    target = pd.to_numeric(data["value"], errors="raise")
    print(pd.DataFrame({
        "statistic": ["rows", "unique complexes", "unique DOI", "metals", "mean", "median", "std",
                      "min", "q05", "q25", "q75", "q95", "max"],
        "value": [len(data), data["canonical_complex_dative"].nunique(), data["doi"].nunique(dropna=True),
                  data["metal"].nunique(), target.mean(), target.median(), target.std(), target.min(),
                  target.quantile(.05), target.quantile(.25), target.quantile(.75),
                  target.quantile(.95), target.max()],
    }).to_string(index=False))
    print(pd.crosstab(data["domain"], data["type_authors"], margins=True).to_string())
    fields = ["pH", "temperature_C", "buffer_class", "buffer_conc_mM", "halide_type",
              "halide_conc_mM", "cosolvent", "detection", "counterion"]
    completeness = pd.DataFrame({"field": fields, "available": [data[f].notna().sum() for f in fields]})
    completeness["fraction"] = completeness["available"] / len(data)
    print(completeness.to_string(index=False))

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.histplot(data=data, x="value", hue="type_authors", bins=45, element="step", ax=axes[0, 0])
    axes[0, 0].set(title="Experimental lipophilicity", xlabel="LogP / LogD")
    metal_counts = data["metal"].value_counts().head(15).sort_values()
    axes[0, 1].barh(metal_counts.index, metal_counts.values, color="#267365")
    axes[0, 1].set(title="Most represented metals", xlabel="Measurements")
    sns.boxplot(data=data, x="domain", y="value", hue="type_authors", ax=axes[1, 0])
    axes[1, 0].set(title="Target by nuclearity", xlabel="", ylabel="LogP / LogD")
    sns.histplot(data.groupby("canonical_complex_dative").size(), discrete=True, ax=axes[1, 1], color="#C65D3B")
    axes[1, 1].set(title="Measurements per retained complex", xlabel="Measurements", ylabel="Complexes")
    fig.tight_layout()
    path = FIGURE_DIR / "dataset_overview.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():
    environment()
    print("Input SHA-256:", sha256(SOURCE_PATH))
    data = select_rows(pd.read_csv(SOURCE_PATH, low_memory=False))
    data = add_conditions(add_descriptors(add_structures(data)))
    if FOLD_PATH.exists() and "--refold" not in sys.argv:
        load_folds(data)
    else:
        make_folds(data).to_csv(FOLD_PATH, index=False, encoding="utf-8")
    data.to_csv(MODELING_PATH, index=False, encoding="utf-8")
    write_dictionary()
    print("Saved:", MODELING_PATH)
    print("Saved:", FOLD_PATH)
    overview(data)


if __name__ == "__main__":
    main()
