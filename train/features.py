"""
MetalLipoDB modeling — feature lists shared by every notebook.

The single place where the model inputs are defined: 00 computes these columns,
01-04 feed COMMON_NUMERIC and COMMON_CATEGORICAL to the models. A feature is
removed or added here, never in a notebook.
"""

RDKIT25 = [
    "MolWt", "HeavyAtomCount", "NumHeteroatoms", "MolLogP", "TPSA", "LabuteASA",
    "NumHAcceptors", "NumHDonors", "NHOHCount", "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "FractionCSP3", "BertzCT", "MaxPartialCharge",
    "MinPartialCharge", "MaxEStateIndex", "MinEStateIndex", "fr_benzene", "fr_pyridine",
    "fr_halogen", "fr_Ar_NH", "fr_amide", "fr_COO", "fr_ether",
]
METALS = [
    "Ti", "V", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Gd", "Re", "Os", "Ir", "Pt", "Au", "Hg",
]
METAL_GROUP = {
    "Ti": 4, "V": 5, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11,
    "Zn": 12, "Ga": 13, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11,
    "Cd": 12, "In": 13, "Gd": 3, "Re": 7, "Os": 8, "Ir": 9, "Pt": 10,
    "Au": 11, "Hg": 12,
}
METAL_EN = {
    "Ti": 1.54, "V": 1.63, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91,
    "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28,
    "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Gd": 1.20, "Re": 1.90,
    "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00,
}
COUNTERION_ELEMENTS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Na", "K"]
COUNTERION_FEATURES = [
    "counterion_present", "counterion_n_fragments", "counterion_n_unique_fragments",
    "counterion_formal_charge_sum", "counterion_abs_charge_sum", "counterion_MW",
    "counterion_heavy_atoms", "counterion_TPSA", "counterion_MolLogP", "counterion_HBA",
] + [f"counterion_count_{element}" for element in COUNTERION_ELEMENTS]
COORDINATION_FEATURES = (
    [f"metal_count_{symbol}" for symbol in METALS]
    + ["metal_formal_charge_sum", "metal_formal_charge_range"]
    + ["coordination_number_mean", "coordination_number_min", "coordination_number_max"]
    + ["donor_bonds_N", "donor_bonds_O", "donor_bonds_S", "donor_bonds_P",
       "donor_bonds_Cl", "donor_bonds_C", "donor_bonds_other"]
    + ["n_bridging_donor_atoms", "n_metal_metal_bonds"]
)
COMMON_NUMERIC = (
    [f"rdkit_{name}" for name in RDKIT25]
    + ["metal_atomic_number_mean", "metal_oxidation_state_mean", "metal_valence_e_mean",
       "metal_electronegativity_mean", "complex_charge_numeric", "is_logP",
       "pH_present", "pH_value", "temperature_present", "temperature_C_value",
       "halide_conc_present", "halide_conc_mM_value", "cosolvent_present"]
    + COUNTERION_FEATURES + COORDINATION_FEATURES
)
COMMON_CATEGORICAL = ["detection_group", "halide_type_group", "buffer_class_group"]
assert len(RDKIT25) == 25 and len(COUNTERION_FEATURES) == 22 and len(COORDINATION_FEATURES) == 37

# Step 1 of the feature clean-up (RETRAIN_PLAN.md, section 7): features that are
# zero in 97-100% of rows, duplicates by construction, and one of each pair with
# |r| >= 0.95. They are still computed, but left out of the model inputs.
DROPPED_UNINFORMATIVE = (
    [f"metal_count_{s}" for s in ["Ti", "V", "Mn", "Co", "Ni", "Zn", "Ga", "Rh",
                                  "Pd", "Ag", "Cd", "In", "Gd", "Re", "Os", "Hg"]]
    + [f"counterion_count_{e}" for e in ["Br", "I", "B", "Na", "K"]]
    + ["metal_formal_charge_sum", "counterion_formal_charge_sum",
       "metal_valence_e_mean", "counterion_n_unique_fragments"]
    + ["rdkit_HeavyAtomCount", "rdkit_LabuteASA", "coordination_number_min",
       "coordination_number_max", "counterion_HBA", "counterion_count_O",
       "counterion_heavy_atoms"]
)
assert len(DROPPED_UNINFORMATIVE) == 32
assert set(DROPPED_UNINFORMATIVE) <= set(COMMON_NUMERIC)
COMMON_NUMERIC = [c for c in COMMON_NUMERIC if c not in DROPPED_UNINFORMATIVE]

# Step 2: the authors' logP/logD label. The paper shows it does not match the
# measurement conditions for most rows, and its ablation contribution is zero
# (RETRAIN_PLAN.md, section 7). Still computed in 00, not a model input.
DROPPED_AUTHOR_LABEL = ["is_logP"]
assert set(DROPPED_AUTHOR_LABEL) <= set(COMMON_NUMERIC)
COMMON_NUMERIC = [c for c in COMMON_NUMERIC if c not in DROPPED_AUTHOR_LABEL]

# Step 3: how concentrations were measured. It does not change the partition
# coefficient, but it is constant within an article and acts as an article label:
# without it transfer to new articles is slightly better (RETRAIN_PLAN.md, section 7).
# Still computed in 00, not a model input.
DROPPED_ARTICLE_LABEL = ["detection_group"]
assert set(DROPPED_ARTICLE_LABEL) <= set(COMMON_CATEGORICAL)
COMMON_CATEGORICAL = [c for c in COMMON_CATEGORICAL if c not in DROPPED_ARTICLE_LABEL]

# Step 4: pH and temperature. Reported in ~28% of rows; 96% of reported pH values
# are 7.0-7.4 and temperatures are 25 or 37 C, so they carry almost no variation,
# and the missing ones were filled with 7.0 and 0 C. Whether the aqueous phase is
# buffered is kept through buffer_class. Still computed in 00, not model inputs.
DROPPED_SPARSE_CONDITIONS = ["pH_value", "pH_present", "temperature_C_value", "temperature_present"]
assert set(DROPPED_SPARSE_CONDITIONS) <= set(COMMON_NUMERIC)
COMMON_NUMERIC = [c for c in COMMON_NUMERIC if c not in DROPPED_SPARSE_CONDITIONS]

# Step 5: the metal as its periodic group instead of per-metal counts. Ligands
# predict the group far better than the metal within it (balanced accuracy 0.83
# for groups vs 0.66 for metals), and swapping a 4d metal for its 5d congener with
# the same ligands barely moves logP (median |Δ| 0.22; SI Figure S2). Counts of
# individual metals are still computed in 00, not model inputs. For heterometallic
# complexes the group is averaged over the metal centres, like the other metal_*_mean.
DROPPED_METAL_COUNTS = [c for c in COMMON_NUMERIC if c.startswith("metal_count_")]
ADDED_METAL_GROUP = ["metal_group_mean"]
COMMON_NUMERIC = [c for c in COMMON_NUMERIC if c not in DROPPED_METAL_COUNTS] + ADDED_METAL_GROUP

# Step 6: a chemically meaningful description of the metal. Dropped: the atomic
# number (meaningless as a number across periods; its mean over Fe-Ru is 35,
# bromine), the electronegativity (poorly defined for d metals, another encoding
# of the metal identity) and the charge range between metal centres (zero for the
# ~90% mononuclear rows). Added: period (with the group it fixes the metal),
# d-electron count (sets the geometry: d6 octahedral, d8 square planar, d10),
# number of metal centres, and a flag for heterometallic complexes, whose averaged
# metal features are not those of any real metal.
DROPPED_METAL_DESCRIPTORS = ["metal_atomic_number_mean", "metal_electronegativity_mean",
                             "metal_formal_charge_range"]
ADDED_METAL_DESCRIPTORS = ["metal_period_mean", "metal_d_electrons_mean",
                           "n_metal_centers", "heterometallic"]
assert set(DROPPED_METAL_DESCRIPTORS) <= set(COMMON_NUMERIC)
COMMON_NUMERIC = [c for c in COMMON_NUMERIC if c not in DROPPED_METAL_DESCRIPTORS] \
    + ADDED_METAL_DESCRIPTORS

# Step 7: review of the remaining inputs. Dropped: Gasteiger partial charges
# (ill-defined on the charged donor atoms of the ligand set, [c-], [O-], [Cl-]),
# fr_halogen (mixes an organic C-X with a coordinated chloride, which is counted in
# donor_bonds_Cl; organic halogens are in the Morgan fingerprint), and two near
# duplicates: counterion_present (r = 0.96 with a nonzero complex charge) and
# NHOHCount (r = 0.92 with NumHDonors).
DROPPED_REVIEW = ["rdkit_MaxPartialCharge", "rdkit_MinPartialCharge", "rdkit_fr_halogen",
                  "counterion_present", "rdkit_NHOHCount"]
assert set(DROPPED_REVIEW) <= set(COMMON_NUMERIC)
COMMON_NUMERIC = [c for c in COMMON_NUMERIC if c not in DROPPED_REVIEW]


# ── ligand fingerprint, shared by LightGBM and TabPFN ─────────────────────────
# Count Morgan fingerprint (radius 2) of the whole ligand set: each bit holds how
# many times the fragment occurs in the complex, over all ligands. It equals the
# sum of the per-ligand count fingerprints. Only the length differs between
# models (LightGBM 2048, TabPFN 512).
import numpy as _np
from rdkit import Chem as _Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator as _GetMorganGenerator

MORGAN_RADIUS = 2
_generators = {}


def ligand_count_fingerprint(smiles, fp_size):
    if fp_size not in _generators:
        _generators[fp_size] = _GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=fp_size)
    mol = _Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid ligand SMILES: {smiles}")
    return _generators[fp_size].GetCountFingerprintAsNumPy(mol).astype(_np.float32)


# ── feature blocks, for SHAP grouping and ablations ───────────────────────────
BLOCKS = ["ligands: fingerprint", "ligands: descriptors", "metal", "coordination",
          "ion pair", "conditions"]
_METAL = {"metal_oxidation_state_mean", "metal_group_mean", "metal_period_mean",
          "metal_d_electrons_mean", "n_metal_centers", "heterometallic"}
_CONDITIONS = {"halide_conc_present", "halide_conc_mM_value", "cosolvent_present",
               "halide_type_group", "buffer_class_group"}


def feature_block(name):
    if name.startswith("MorganBit_"):
        return "ligands: fingerprint"
    if name.startswith("rdkit_"):
        return "ligands: descriptors"
    if name in _METAL:
        return "metal"
    if name.startswith(("coordination_", "donor_bonds_")) or name in (
            "n_bridging_donor_atoms", "n_metal_metal_bonds"):
        return "coordination"
    if name == "complex_charge_numeric" or name.startswith("counterion_"):
        return "ion pair"
    if name in _CONDITIONS:
        return "conditions"
    raise KeyError(f"feature {name} has no block")


assert all(feature_block(c) in BLOCKS for c in COMMON_NUMERIC + COMMON_CATEGORICAL)
