"""
MetalLipoDB — release_corrections.py
====================================
Corrections to the published release, applied on top of the frozen Zenodo
table (release_v1/MetalLipoDB_v1.0.csv, 5190 rows) rather than by rebuilding
from the raw file: clean_values.py does not reproduce every step of the
release (record_id, formatting), so the published table is the base.

Every correction is keyed by record_id and guarded: the row must exist and
hold the expected DOI, abbreviation and value, otherwise the script stops.
record_id values are never reassigned; removed rows leave gaps.

Two kinds of corrections: FIX (field changes, old value checked) and DROP.
Writes MetalLipoDB.csv and release_v1/CHANGELOG.csv (one line per change).

Usage:
    python release_corrections.py
"""

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "release_v1" / "MetalLipoDB_v1.0.csv"
BASE_SHA256 = "0fa9870d01380484c0267057a0d3f57e8e95b6703342936f92fc341ccdd77191"
OUT = ROOT / "MetalLipoDB.csv"
ID_MAP = ROOT / "release_v1" / "record_id_map.csv"
CHANGELOG = ROOT / "release_v1" / "CHANGELOG.csv"

# Values quoted from an earlier paper instead of being measured: the dataset
# keeps only values measured in the article itself.
DROP = [
    # Ong et al., Angew. Chem. 2018 (Ang group) give logP of cP-(OH)2 and cP-(OBz)2
    # in one sentence, without table or SD. Both are identical to two decimals to
    # the group's 2012 values (10.1021/jm300580y: -2.12 +/- 0.24, 1.01 +/- 0.01),
    # cited next to them. The 2012 records stay.
    {"record_id": "MLDB-002080", "doi": "10.1002/anie.201810361",
     "abbreviation": "cP-(OH)₂", "value": -2.12,
     "reason": "value reused from 10.1021/jm300580y, not re-measured"},
    {"record_id": "MLDB-002081", "doi": "10.1002/anie.201810361",
     "abbreviation": "cP-(OBz)₂", "value": 1.01,
     "reason": "value reused from 10.1021/jm300580y, not re-measured"},

    # García-Moreno et al., Eur. J. Med. Chem. 2014 (Laguna group) list logD7.4 of
    # the four [AuX(PTA-R)]X precursors next to the new thiolates. The four numbers
    # are those of the group's 2013 paper (10.1002/ejic.201201411, Table 5: -0.40,
    # -0.34, -0.39, -0.14), reassigned between the compounds and with -0.39 given
    # as -0.40. Reused, not re-measured; the 2013 records stay.
    {"record_id": "MLDB-004244", "doi": "10.1016/j.ejmech.2014.04.001",
     "abbreviation": "1a", "value": -0.14,
     "reason": "precursor value reused from 10.1002/ejic.201201411, not re-measured"},
    {"record_id": "MLDB-004245", "doi": "10.1016/j.ejmech.2014.04.001",
     "abbreviation": "1a'", "value": -0.34,
     "reason": "precursor value reused from 10.1002/ejic.201201411, not re-measured"},
    {"record_id": "MLDB-004246", "doi": "10.1016/j.ejmech.2014.04.001",
     "abbreviation": "1b", "value": -0.4,
     "reason": "precursor value reused from 10.1002/ejic.201201411, not re-measured"},
    {"record_id": "MLDB-004247", "doi": "10.1016/j.ejmech.2014.04.001",
     "abbreviation": "1b'", "value": -0.4,
     "reason": "precursor value reused from 10.1002/ejic.201201411, not re-measured"},

    # Song et al., J. Inorg. Biochem. 2003 (Sohn group) include [Pt(dach)(OPr)4]
    # and JM216 in Table 3 as references. The text gives their logP with a citation
    # to the group's 2000 paper (ref. 12 = 10.1021/jm9904250, Table 3: 0.18, 0.1).
    # The paper's own complexes carry three decimals, these two do not.
    {"record_id": "MLDB-002041", "doi": "10.1016/s0162-0134(03)00149-1",
     "abbreviation": "[PtIV(dach)(OPr)4]", "value": 0.18,
     "reason": "reference value cited from 10.1021/jm9904250"},
    {"record_id": "MLDB-002042", "doi": "10.1016/s0162-0134(03)00149-1",
     "abbreviation": "JM216", "value": 0.1,
     "reason": "reference value cited from 10.1021/jm9904250"},

    # Buss et al., J. Biol. Inorg. Chem. 2012 (Jaehde group): oxaliplatin and
    # complexes 2-5 are "the previously studied complexes 2-5 [8]", ref. 8 being
    # the group's 2011 paper (10.1016/j.jinorgbio.2011.02.005, Table 1: -1.76,
    # -1.28, -0.68, -0.28, 0.42). Only complexes 6-8 are new; they stay.
    {"record_id": "MLDB-004835", "doi": "10.1007/s00775-012-0889-9",
     "abbreviation": "Oxaliplatin", "value": -1.76,
     "reason": "value from 10.1016/j.jinorgbio.2011.02.005 (previously studied complexes)"},
    {"record_id": "MLDB-004836", "doi": "10.1007/s00775-012-0889-9",
     "abbreviation": "2", "value": -1.28,
     "reason": "value from 10.1016/j.jinorgbio.2011.02.005 (previously studied complexes)"},
    {"record_id": "MLDB-004837", "doi": "10.1007/s00775-012-0889-9",
     "abbreviation": "3", "value": -0.68,
     "reason": "value from 10.1016/j.jinorgbio.2011.02.005 (previously studied complexes)"},
    {"record_id": "MLDB-004838", "doi": "10.1007/s00775-012-0889-9",
     "abbreviation": "4", "value": -0.28,
     "reason": "value from 10.1016/j.jinorgbio.2011.02.005 (previously studied complexes)"},
    {"record_id": "MLDB-004839", "doi": "10.1007/s00775-012-0889-9",
     "abbreviation": "5", "value": 0.42,
     "reason": "value from 10.1016/j.jinorgbio.2011.02.005 (previously studied complexes)"},

    # Buss et al., J. Inorg. Biochem. 2011 (Bonn): cisplatin -2.53 and carboplatin
    # -2.30 without SD, both identical to Screnci et al., Br. J. Cancer 2000
    # (Auckland, 10.1054/bjoc.1999.1026: -2.53 +/- 0.28, -2.3 +/- 0.10). Their own
    # oxaliplatin differs (-1.76 vs -1.65), so the two commercial references were
    # taken from the literature without citation. The 2000 records stay.
    {"record_id": "MLDB-001836", "doi": "10.1016/j.jinorgbio.2011.02.005",
     "abbreviation": "Cisplatin", "value": -2.53,
     "reason": "identical to 10.1054/bjoc.1999.1026, taken from the literature"},
    {"record_id": "MLDB-001837", "doi": "10.1016/j.jinorgbio.2011.02.005",
     "abbreviation": "Carboplatin", "value": -2.3,
     "reason": "identical to 10.1054/bjoc.1999.1026, taken from the literature"},

    # Bolitho et al., Angew. Chem. 2021 (Sadler group), SI Table S2: RR-1 and RR-4
    # carry footnote [a] "Literature Log P value of RR-1 and RR-4", i.e. the group's
    # Nat. Chem. 2018 values (10.1038/nchem.2918: 1.45 +/- 0.02, 0.30 +/- 0.03).
    # SS-2, RR-2, SS-3, RR-3 and SS-5 were measured and stay.
    {"record_id": "MLDB-000004", "doi": "10.1002/anie.202016456",
     "abbreviation": "R,R-1", "value": 1.45,
     "reason": "literature value (SI Table S2, footnote a) from 10.1038/nchem.2918"},
    {"record_id": "MLDB-000009", "doi": "10.1002/anie.202016456",
     "abbreviation": "R,R-4", "value": 0.3,
     "reason": "literature value (SI Table S2, footnote a) from 10.1038/nchem.2918"},

    # Varbanov et al., Eur. J. Med. Chem. 2011: shake-flask logP was determined
    # for compounds 4, 5, 6 and 8 only; R1 and R2 are comparison complexes
    # "described in Ref. [22]", whose literature logP served to calibrate RP-HPLC.
    {"record_id": "MLDB-001912", "doi": "10.1016/j.ejmech.2011.09.006",
     "abbreviation": "R1", "value": -0.81,
     "reason": "reference complex, logP from ref. 22 of the paper (not measured)"},
    {"record_id": "MLDB-001917", "doi": "10.1016/j.ejmech.2011.09.006",
     "abbreviation": "R2", "value": 1.69,
     "reason": "reference complex, logP from ref. 22 of the paper (not measured)"},

    # Giringer et al., Electrophoresis 2018, Table 2: NKP-1339 "0.27 a [55]", a
    # literature logD7.4 (Chang et al., Inorg. Chem. 2016, 10.1021/acs.inorgchem.6b00359,
    # measured there in 1% DMSO/PBS). The other shake-flask values of the table
    # without a reference stay.
    {"record_id": "MLDB-000216", "doi": "10.1002/elps.201700443",
     "abbreviation": "NKP-1339", "value": 0.27,
     "reason": "literature value (Table 2, ref. 55) from 10.1021/acs.inorgchem.6b00359"},

    # Zhang et al., J. Inorg. Biochem. 2026 (Xue group): Ir-NH2 0.32 is identical to
    # the group's 2025 value (10.1016/j.ica.2025.122601), measured by a verbatim
    # copy of the same protocol. Same authors and exactly the same number: the
    # later record is treated as reused (user decision, 2026-09-30).
    {"record_id": "MLDB-000570", "doi": "10.1016/j.jinorgbio.2025.113198",
     "abbreviation": "Ir-NH2", "value": 0.32,
     "reason": "same group, identical to 10.1016/j.ica.2025.122601"},

    # Zajac et al., J. Inorg. Biochem. 2016 (Brabec group): [Pt(dach)(OH)2(ox)] -2.38
    # is identical to the group's 2014 value for the same complex (10.1016/
    # j.jinorgbio.2014.07.004, -2.38 +/- 0.19) and given without SD. Same group and
    # exactly the same number: the later record is treated as reused (user rule).
    {"record_id": "MLDB-001548", "doi": "10.1016/j.jinorgbio.2015.12.003",
     "abbreviation": "[Pt(dach)(OH)2(ox)]", "value": -2.38,
     "reason": "same group, identical to 10.1016/j.jinorgbio.2014.07.004"},

    # Satraplatin as the reference of the Gao/Yang group's CBD-Pt(IV) series:
    # -0.11 +/- 0.02 in J. Inorg. Biochem. (online March 2024, 10.1016/
    # j.jinorgbio.2024.112515), then -0.11 without SD in Appl. Organomet. Chem.
    # (online October 2024, same line of work, protocol shared with the group's
    # 10.1002/aoc.70416). Same group, exactly the same number: the later record goes.
    # The 2025 value (-0.12) differs and stays.
    {"record_id": "MLDB-004728", "doi": "10.1002/aoc.7865",
     "abbreviation": "Satraplatin", "value": -0.11,
     "reason": "same group, identical to 10.1016/j.jinorgbio.2024.112515"},

    # Sakellariou et al., Dalton Trans. 2026 (Lazarides group) quote [Ru(bpy)3]2+
    # log Po/w = -2.00 in the discussion only; it is not among the measured
    # compounds of Fig. 2 or the method. The value is the group's 2025 measurement
    # (10.1021/acs.inorgchem.5c03244, Tris-HCl), which stays.
    {"record_id": "MLDB-004409", "doi": "10.1039/d6dt00689b",
     "abbreviation": "[Ru(bpy)3]2+", "value": -2.0,
     "reason": "same group, value from 10.1021/acs.inorgchem.5c03244, not measured here"},

    # Sakthikumar et al., RSC Med. Chem. 2023, Table 11: logD was computed from the
    # same shake-flask experiment as logP (aqueous ions from conductivity, pKa via
    # Henderson-Hasselbalch), not measured separately; one experiment, one record.
    {"record_id": "MLDB-002409", "doi": "10.1039/d2md00394e",
     "abbreviation": "2", "value": 1.107,
     "reason": "logD derived from the same experiment as the logP record"},
    {"record_id": "MLDB-002410", "doi": "10.1039/d2md00394e",
     "abbreviation": "3", "value": -1.497,
     "reason": "logD derived from the same experiment as the logP record"},
    {"record_id": "MLDB-002411", "doi": "10.1039/d2md00394e",
     "abbreviation": "4", "value": 1.158,
     "reason": "logD derived from the same experiment as the logP record"},
    # 10.1002/cmdc.202300347 (MacDonnell group, Alatrash co-author): P3 and D3 repeat the
    # 10.1002/cmdc.201700240 values of the same group in both phases, without a new SD.
    {"record_id": "MLDB-003895", "doi": "10.1002/cmdc.202300347",
     "abbreviation": "P3", "value": -1.5,
     "reason": "same group, identical to 10.1002/cmdc.201700240"},
    {"record_id": "MLDB-003902", "doi": "10.1002/cmdc.202300347",
     "abbreviation": "P3", "value": -1.1,
     "reason": "same group, identical to 10.1002/cmdc.201700240"},
    {"record_id": "MLDB-003900", "doi": "10.1002/cmdc.202300347",
     "abbreviation": "D3", "value": 1.4,
     "reason": "same group, identical to 10.1002/cmdc.201700240"},
    {"record_id": "MLDB-003907", "doi": "10.1002/cmdc.202300347",
     "abbreviation": "D3", "value": 1.9,
     "reason": "same group, identical to 10.1002/cmdc.201700240"},
    # 10.1016/j.jinorgbio.2003.09.004 (Sohn group): the carboplatin control repeats
    # P = 0.04, log P -1.398 of the group's earlier paper (their ref. 13).
    {"record_id": "MLDB-002131", "doi": "10.1016/j.jinorgbio.2003.09.004",
     "abbreviation": "Carboplatin", "value": -1.398,
     "reason": "same group, identical to 10.1016/s0162-0134(97)00109-8"},
]

# Field corrections: (record_id, doi, abbreviation, value) identify the row,
# "set" maps column -> (old, new); the old value must be there.
FIX = [
    # 10.1002/ejic.201201411, Table 5: 4b is [AuBr(PTA-CH2CO2Me)]Br and 5a is
    # [AuCl(PTA-CH2Ph)]Cl. The table had the two structures swapped.
    {"record_id": "MLDB-004277", "doi": "10.1002/ejic.201201411", "abbreviation": "4b", "value": -0.34,
     "set": {"smiles_complex": ("[Cl-]->[Au+]<-[P]12CN3CN(C[N+](Cc4ccccc4)(C3)C1)C2", "COC(=O)C[N+]12CN3CN(C1)C[P](->[Au+]<-[Br-])(C3)C2"),
             "smiles_ligands": ("[Cl-].c1ccc(C[N+]23CN4CN(CP(C4)C2)C3)cc1", "COC(=O)C[N+]12CN3CN(CP(C3)C1)C2.[Br-]"),
             "counterion": ("[Cl-]", "[Br-]")},
     "reason": "structure of 5a entered for 4b"},
    {"record_id": "MLDB-004278", "doi": "10.1002/ejic.201201411", "abbreviation": "5a", "value": -0.39,
     "set": {"smiles_complex": ("COC(=O)C[N+]12CN3CN(C1)C[P](->[Au+]<-[Br-])(C3)C2", "[Cl-]->[Au+]<-[P]12CN3CN(C[N+](Cc4ccccc4)(C3)C1)C2"),
             "smiles_ligands": ("COC(=O)C[N+]12CN3CN(CP(C3)C1)C2.[Br-]", "[Cl-].c1ccc(C[N+]23CN4CN(CP(C4)C2)C3)cc1"),
             "counterion": ("[Br-]", "[Cl-]")},
     "reason": "structure of 4b entered for 5a"},
    # 10.1021/jm9904250, Table 3: the carboxylate is written O2CR in the paper;
    # the abbreviations lost the C (the structures are right).
    {"record_id": "MLDB-002020", "doi": "10.1021/jm9904250", "abbreviation": "[Pt(O2C2H5)4(dach)]", "value": 0.18,
     "set": {"abbreviation_in_the_article": ("[Pt(O2C2H5)4(dach)]", "[Pt(O2CC2H5)4(dach)]")},
     "reason": "abbreviation as printed in the paper"},
    {"record_id": "MLDB-002021", "doi": "10.1021/jm9904250", "abbreviation": "[Pt(O2C2H7)4(dach)]", "value": 1.54,
     "set": {"abbreviation_in_the_article": ("[Pt(O2C2H7)4(dach)]", "[Pt(O2CC3H7)4(dach)]")},
     "reason": "abbreviation as printed in the paper"},
    {"record_id": "MLDB-002022", "doi": "10.1021/jm9904250", "abbreviation": "[Pt(O2C4H9)4(dach)]", "value": 2.54,
     "set": {"abbreviation_in_the_article": ("[Pt(O2C4H9)4(dach)]", "[Pt(O2CC4H9)4(dach)]")},
     "reason": "abbreviation as printed in the paper"},
    {"record_id": "MLDB-002023", "doi": "10.1021/jm9904250", "abbreviation": "[Pt(O2C5H11)4(dach)]", "value": 3.03,
     "set": {"abbreviation_in_the_article": ("[Pt(O2C5H11)4(dach)]", "[Pt(O2CC5H11)4(dach)]")},
     "reason": "abbreviation as printed in the paper"},
    # 10.1016/j.jinorgbio.2015.12.003: the carrier ligand is R,R-dach (the paper:
    # "dach = R,R-1,2-diaminocyclohexane", derivatives of oxaliplatin); the table
    # had the S,S enantiomer in both SMILES.
    {"record_id": "MLDB-001549", "doi": "10.1016/j.jinorgbio.2015.12.003", "abbreviation": "[Pt(dach)(DCA)(OH)(ox)]", "value": -1.17,
     "set": {"smiles_complex": ("O=C1[O-]->[Pt+4]2(<-[OH-])(<-[NH2][C@H]3CCCC[C@@H]3[NH2]->2)(<-[O-]C1=O)<-[O-]C(=O)C(Cl)Cl", "O=C1[O-]->[Pt+4]2(<-[OH-])(<-[NH2][C@@H]3CCCC[C@H]3[NH2]->2)(<-[O-]C1=O)<-[O-]C(=O)C(Cl)Cl"),
             "smiles_ligands": ("N[C@H]1CCCC[C@@H]1N.O=C([O-])C(=O)[O-].O=C([O-])C(Cl)Cl.[OH-]", "N[C@@H]1CCCC[C@H]1N.O=C([O-])C(=O)[O-].O=C([O-])C(Cl)Cl.[OH-]")},
     "reason": "S,S-dach entered for R,R-dach"},
    {"record_id": "MLDB-001550", "doi": "10.1016/j.jinorgbio.2015.12.003", "abbreviation": "[Pt(dach)(DCA)2(ox)]", "value": -0.35,
     "set": {"smiles_complex": ("O=C1[O-]->[Pt+4]2(<-[NH2][C@H]3CCCC[C@@H]3[NH2]->2)(<-[O-]C1=O)(<-[O-]C(=O)C(Cl)Cl)<-[O-]C(=O)C(Cl)Cl", "O=C1[O-]->[Pt+4]2(<-[NH2][C@@H]3CCCC[C@H]3[NH2]->2)(<-[O-]C1=O)(<-[O-]C(=O)C(Cl)Cl)<-[O-]C(=O)C(Cl)Cl"),
             "smiles_ligands": ("N[C@H]1CCCC[C@@H]1N.O=C([O-])C(=O)[O-].O=C([O-])C(Cl)Cl.O=C([O-])C(Cl)Cl", "N[C@@H]1CCCC[C@H]1N.O=C([O-])C(=O)[O-].O=C([O-])C(Cl)Cl.O=C([O-])C(Cl)Cl")},
     "reason": "S,S-dach entered for R,R-dach"},
    # 10.1016/j.jinorgbio.2025.113198: the method dissolves the compounds "in PBS
    # buffer" (same protocol as the group's 10.1016/j.ica.2025.122601); the medium
    # was labelled water. Conditions set as for the PBS rows of 10.1016/j.ica.2025.122601.
    {"record_id": "MLDB-000569", "doi": "10.1016/j.jinorgbio.2025.113198", "abbreviation": "Ir-FA", "value": 0.59,
     "set": {"aqueous_phase": ("water", "PBS"), "medium_class": ("water", "buffer + salt"),
             "buffer_class": ("absent", "Phosphate"), "halide_type": ("absent", "Cl"),
             "halide_conc_mM": ("0", "140")},
     "reason": "medium is PBS per the method, not water"},
    # 10.1039/d6dt00689b: Ru-bpy-O3/O4 were not detected in octanol and "a tentative
    # log Po/w value of <= -3 was assigned"; a bound, not a value.
    {"record_id": "MLDB-004407", "doi": "10.1039/d6dt00689b", "abbreviation": "Ru-bpy-O4", "value": -3.0,
     "set": {"value_relation": ("=", "<"), "value_raw": ("-3.00", "<-3")},
     "reason": "upper bound (not detected in octanol), not a measured value"},
    {"record_id": "MLDB-004408", "doi": "10.1039/d6dt00689b", "abbreviation": "Ru-bpy-O3", "value": -3.0,
     "set": {"value_relation": ("=", "<"), "value_raw": ("-3.00", "<-3")},
     "reason": "upper bound (not detected in octanol), not a measured value"},
    # 10.1016/j.molstruc.2024.138997, 2.14: "The chloride salts of Ru(II) complexes
    # (200 uM) were dissolved in water ..." - logP was measured on the chlorides,
    # the table had the PF6 salts (the synthetic counterion).
    {"record_id": "MLDB-000643", "doi": "10.1016/j.molstruc.2024.138997", "abbreviation": "RC1", "value": -2.81,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F", "[Cl-].[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    {"record_id": "MLDB-000644", "doi": "10.1016/j.molstruc.2024.138997", "abbreviation": "RC2", "value": 0.25,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F", "[Cl-].[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    {"record_id": "MLDB-000645", "doi": "10.1016/j.molstruc.2024.138997", "abbreviation": "RC3", "value": 0.53,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F", "[Cl-].[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    # 10.1002/chem.201500561: the method measures the chloride salts ("Cisplatin and chloride salts of VR52, VR54 and VR63 were dissolved in n-octanol-saturated water");
    # the table had the PF6 salts.
    {"record_id": "MLDB-000095", "doi": "10.1002/chem.201500561", "abbreviation": "VR52", "value": -3.85,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F", "[Cl-].[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    {"record_id": "MLDB-000096", "doi": "10.1002/chem.201500561", "abbreviation": "VR54", "value": -1.9,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F", "[Cl-].[Cl-].[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    {"record_id": "MLDB-000097", "doi": "10.1002/chem.201500561", "abbreviation": "VR63", "value": -2.24,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F", "[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    # 10.1039/d0sc03008b: the method measures the chloride salts ("50 uM complex (chloride salt)");
    # the table had the PF6 salts.
    {"record_id": "MLDB-001319", "doi": "10.1039/d0sc03008b", "abbreviation": "[Os(phen)3]2+", "value": -1.697,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F", "[Cl-].[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    {"record_id": "MLDB-001320", "doi": "10.1039/d0sc03008b", "abbreviation": "Os-0T", "value": -2.243,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F", "[Cl-].[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    {"record_id": "MLDB-001321", "doi": "10.1039/d0sc03008b", "abbreviation": "Os-1T", "value": -0.794,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F", "[Cl-].[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    {"record_id": "MLDB-001322", "doi": "10.1039/d0sc03008b", "abbreviation": "Os-2T", "value": -0.341,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F", "[Cl-].[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    {"record_id": "MLDB-001323", "doi": "10.1039/d0sc03008b", "abbreviation": "Os-3T", "value": 0.726,
     "set": {"counterion": ("F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F", "[Cl-].[Cl-]")},
     "reason": "logP measured on the chloride salt"},
    # 10.1016/j.jinorgbio.2017.12.017: D7.4 at 4, 24 and 100 mM chloride in 20 mM
    # phosphate buffer; the halide columns are right but the medium label omitted
    # the chloride, unlike the same design in other articles ("+ 4 mM Cl-").
    {"record_id": "MLDB-003735", "doi": "10.1016/j.jinorgbio.2017.12.017", "abbreviation": "2", "value": -2.4,
     "set": {"aqueous_phase": ("20 mM phosphate buffer", "20 mM phosphate buffer + 4 mM Cl-")},
     "reason": "medium label without the chloride concentration"},
    {"record_id": "MLDB-003736", "doi": "10.1016/j.jinorgbio.2017.12.017", "abbreviation": "3", "value": -1.4,
     "set": {"aqueous_phase": ("20 mM phosphate buffer", "20 mM phosphate buffer + 4 mM Cl-")},
     "reason": "medium label without the chloride concentration"},
    {"record_id": "MLDB-003737", "doi": "10.1016/j.jinorgbio.2017.12.017", "abbreviation": "2", "value": -1.6,
     "set": {"aqueous_phase": ("20 mM phosphate buffer", "20 mM phosphate buffer + 24 mM Cl-")},
     "reason": "medium label without the chloride concentration"},
    {"record_id": "MLDB-003738", "doi": "10.1016/j.jinorgbio.2017.12.017", "abbreviation": "3", "value": -1.16,
     "set": {"aqueous_phase": ("20 mM phosphate buffer", "20 mM phosphate buffer + 24 mM Cl-")},
     "reason": "medium label without the chloride concentration"},
    {"record_id": "MLDB-003739", "doi": "10.1016/j.jinorgbio.2017.12.017", "abbreviation": "2", "value": -1.16,
     "set": {"aqueous_phase": ("20 mM phosphate buffer", "20 mM phosphate buffer + 100 mM Cl-")},
     "reason": "medium label without the chloride concentration"},
    {"record_id": "MLDB-003740", "doi": "10.1016/j.jinorgbio.2017.12.017", "abbreviation": "3", "value": -0.97,
     "set": {"aqueous_phase": ("20 mM phosphate buffer", "20 mM phosphate buffer + 100 mM Cl-")},
     "reason": "medium label without the chloride concentration"},
    # 10.1016/j.jinorgbio.2025.113080, Table 1: Ru-1 1.02 +/- 0.07, Ru-2 1.21 +/- 0.05.
    # The records took the range quoted in the text ("between 0.32 and 1.21"), which
    # does not match the table (0.32 appears nowhere else); the table is the source.
    {"record_id": "MLDB-004467", "doi": "10.1016/j.jinorgbio.2025.113080", "abbreviation": "Ru-1", "value": 1.21,
     "set": {"value": ("1.21", "1.02"), "value_raw": ("1.21", "1.02 ± 0.07"), "std": ("", "0.07")},
     "reason": "value from Table 1, not from the range in the text"},
    {"record_id": "MLDB-004468", "doi": "10.1016/j.jinorgbio.2025.113080", "abbreviation": "Ru-2", "value": 0.32,
     "set": {"value": ("0.32", "1.21"), "value_raw": ("0.32", "1.21 ± 0.05"), "std": ("", "0.05")},
     "reason": "value from Table 1, not from the range in the text"},
    # 10.1016/j.jinorgbio.2022.112058, Table 2, footnote a: PBS' here is 15 mM
    # phosphate with 102 mM chloride (1.5 mM KCl + 100.5 mM NaCl), not standard PBS.
    {"record_id": "MLDB-000509", "doi": "10.1016/j.jinorgbio.2022.112058", "abbreviation": "1", "value": -1.73,
     "set": {"halide_conc_mM": ("140", "102"), "buffer_conc_mM": ("", "15")},
     "reason": "modified PBS of the article: 15 mM phosphate, 102 mM chloride"},
    {"record_id": "MLDB-000510", "doi": "10.1016/j.jinorgbio.2022.112058", "abbreviation": "2", "value": 1.48,
     "set": {"halide_conc_mM": ("140", "102"), "buffer_conc_mM": ("", "15")},
     "reason": "modified PBS of the article: 15 mM phosphate, 102 mM chloride"},
    # 10.1039/d2md00394e: the complexes are numbered 1-3 in the paper, "M = Mn (1),
    # Co (2) or Ni (3)"; the records carried 2-4 (values and metals were right).
    {"record_id": "MLDB-002406", "doi": "10.1039/d2md00394e", "abbreviation": "2", "value": 1.166,
     "set": {"abbreviation_in_the_article": ("2", "1")},
     "reason": "numbering of the paper"},
    {"record_id": "MLDB-002407", "doi": "10.1039/d2md00394e", "abbreviation": "3", "value": -1.477,
     "set": {"abbreviation_in_the_article": ("3", "2")},
     "reason": "numbering of the paper"},
    {"record_id": "MLDB-002408", "doi": "10.1039/d2md00394e", "abbreviation": "4", "value": 1.17,
     "set": {"abbreviation_in_the_article": ("4", "3")},
     "reason": "numbering of the paper"},
    # 10.1021/acs.jmedchem.5c00086: the main text swaps the two values ("Pt-BisDHA
    # (1.19 +/- 0.06) and Pt-DHA (1.60 +/- 0.07)"), whereas SI Table S2 gives Pt-DHA
    # 1.19 and Pt-BisDHA 1.60, and the discussion states twice that Pt-BisDHA is the
    # more lipophilic one. The records followed the main text; set as in Table S2.
    {"record_id": "MLDB-002137", "doi": "10.1021/acs.jmedchem.5c00086", "abbreviation": "Pt-BisDHA", "value": 1.19,
     "set": {"value": ("1.19", "1.60"), "value_raw": ("1.19 ± 0.06", "1.60 ± 0.07"), "std": ("0.06", "0.07")},
     "reason": "values swapped in the main text; SI Table S2 and the discussion"},
    {"record_id": "MLDB-002138", "doi": "10.1021/acs.jmedchem.5c00086", "abbreviation": "Pt-DHA", "value": 1.6,
     "set": {"value": ("1.6", "1.19"), "value_raw": ("1.60 ± 0.07", "1.19 ± 0.06"), "std": ("0.07", "0.06")},
     "reason": "values swapped in the main text; SI Table S2 and the discussion"},
    # van Rijt et al., Metallomics 2014 (10.1039/c4mt00034j): "log P ... increase in
    # the order 3 < 2 < 1 with values of -0.57 for 3, -0.30 for 2 and 0.26 for 1"
    # (the minus of 2 was lost), and the chloride complex 3 was partitioned in
    # 0.3 M NaCl, only the iodides 1 and 2 in 0.3 M NaI.
    {"record_id": "MLDB-001079", "doi": "10.1039/c4mt00034j", "abbreviation": "2", "value": 0.3,
     "set": {"value": ("0.3", "-0.30"), "value_raw": ("0.30 ± 0.01", "-0.30 ± 0.01")},
     "reason": "sign lost: -0.30 in the paper (order 3 < 2 < 1)"},
    {"record_id": "MLDB-001080", "doi": "10.1039/c4mt00034j", "abbreviation": "3", "value": -0.57,
     "set": {"aqueous_phase": ("300 mM NaI", "300 mM NaCl"), "halide_type": ("I", "Cl")},
     "reason": "chloride complex partitioned in 0.3 M NaCl"},
    # 10.1021/acs.jmedchem.9b00489: 6-8 are cct-[Pt(IV)(1S,2S-DACH)(5,6-Me2phen)(RCOO)2]2+,
    # Pt(IV) derivatives of 56MeSS (5); the records had Pt(II) and a neutral complex.
    # 8 was isolated as its trifluoroacetate salt; for 6 and 7 (prepared as in ref. 66)
    # the salt is not stated in the paper, so no counterion is given.
    {"record_id": "MLDB-002212", "doi": "10.1021/acs.jmedchem.9b00489", "abbreviation": "6", "value": -1.37,
     "set": {"smiles_complex": ("CC(=O)[O-]->[Pt+2]12(<-[NH2][C@H]3CCCC[C@@H]3[NH2]->1)(<-[O-]C(=O)CCCc1ccccc1)<-[n]1cccc3c(C)c(C)c4ccc[n]->2c4c31", "CC(=O)[O-]->[Pt+4]12(<-[NH2][C@H]3CCCC[C@@H]3[NH2]->1)(<-[O-]C(=O)CCCc1ccccc1)<-[n]1cccc3c(C)c(C)c4ccc[n]->2c4c31"), "complex_charge": ("0", "2")},
     "reason": "Pt(IV) dication, not a neutral Pt(II) complex"},
    {"record_id": "MLDB-002213", "doi": "10.1021/acs.jmedchem.9b00489", "abbreviation": "7", "value": -0.75,
     "set": {"smiles_complex": ("CCCCCCCC(=O)[O-]->[Pt+2]12(<-[NH2][C@H]3CCCC[C@@H]3[NH2]->1)(<-[O-]C(C)=O)<-[n]1cccc3c(C)c(C)c4ccc[n]->2c4c31", "CCCCCCCC(=O)[O-]->[Pt+4]12(<-[NH2][C@H]3CCCC[C@@H]3[NH2]->1)(<-[O-]C(C)=O)<-[n]1cccc3c(C)c(C)c4ccc[n]->2c4c31"), "complex_charge": ("0", "2")},
     "reason": "Pt(IV) dication, not a neutral Pt(II) complex"},
    {"record_id": "MLDB-002214", "doi": "10.1021/acs.jmedchem.9b00489", "abbreviation": "8", "value": 0.01,
     "set": {"smiles_complex": ("CCCCCCCC(=O)[O-]->[Pt+2]12(<-[NH2][C@H]3CCCC[C@@H]3[NH2]->1)(<-[O-]C(=O)CCCc1ccccc1)<-[n]1cccc3c(C)c(C)c4ccc[n]->2c4c31", "CCCCCCCC(=O)[O-]->[Pt+4]12(<-[NH2][C@H]3CCCC[C@@H]3[NH2]->1)(<-[O-]C(=O)CCCc1ccccc1)<-[n]1cccc3c(C)c(C)c4ccc[n]->2c4c31"), "complex_charge": ("0", "2"), "counterion": ("", "O=C([O-])C(F)(F)F.O=C([O-])C(F)(F)F")},
     "reason": "Pt(IV) dication, not a neutral Pt(II) complex"},
    # 10.1039/c4dt03679d: Table 3 gives an SD for every log P (lost on entry), 3d is printed as 0.1512.
    {"record_id": "MLDB-002618", "doi": "10.1039/c4dt03679d", "abbreviation": "Cisplatin", "value": -2.36,
     "set": {"std": ("", "0.01"), "value_raw": ("-2.36", "-2.36 ± 0.01")},
     "reason": "SD from Table 3"},
    {"record_id": "MLDB-002619", "doi": "10.1039/c4dt03679d", "abbreviation": "3a", "value": -0.01,
     "set": {"std": ("", "0.03"), "value_raw": ("-0.01", "-0.01 ± 0.03")},
     "reason": "SD from Table 3"},
    {"record_id": "MLDB-002620", "doi": "10.1039/c4dt03679d", "abbreviation": "3b", "value": 0.53,
     "set": {"std": ("", "0.02"), "value_raw": ("0.53", "0.53 ± 0.02")},
     "reason": "SD from Table 3"},
    {"record_id": "MLDB-002621", "doi": "10.1039/c4dt03679d", "abbreviation": "3c", "value": 0.56,
     "set": {"std": ("", "0.07"), "value_raw": ("0.56", "0.56 ± 0.07")},
     "reason": "SD from Table 3"},
    {"record_id": "MLDB-002622", "doi": "10.1039/c4dt03679d", "abbreviation": "3d", "value": 0.15,
     "set": {"std": ("", "0.05"), "value": ("0.15", "0.1512"), "value_raw": ("0.15", "0.1512 ± 0.05")},
     "reason": "SD from Table 3"},
    {"record_id": "MLDB-002623", "doi": "10.1039/c4dt03679d", "abbreviation": "3e", "value": -1.96,
     "set": {"std": ("", "0.06"), "value_raw": ("-1.96", "-1.96 ± 0.06")},
     "reason": "SD from Table 3"},
]


def target(d, rec):
    """The single row a correction is keyed to, checked against DOI, abbreviation and value."""
    row = d[d["record_id"] == rec["record_id"]]
    if len(row) != 1:
        raise SystemExit(f"{rec['record_id']}: found {len(row)} rows")
    r = row.iloc[0]
    if (r["doi"].lower() != rec["doi"] or r["abbreviation_in_the_article"] != rec["abbreviation"]
            or abs(float(r["value"]) - rec["value"]) > 1e-9):
        raise SystemExit(f"{rec['record_id']}: row does not match {rec}")
    return row.index[0], r


def main():
    digest = hashlib.sha256(BASE.read_bytes()).hexdigest()
    if digest != BASE_SHA256:
        raise SystemExit(f"{BASE.name} is not the published v1.0 table (sha256 {digest})")
    d = pd.read_csv(BASE, dtype=str, keep_default_na=False)
    log = []
    for rec in FIX:
        i, r = target(d, rec)
        for col, (old, new) in rec["set"].items():
            if r[col] != old:
                raise SystemExit(f"{rec['record_id']}: {col} is {r[col]!r}, expected {old!r}")
            d.at[i, col] = new
            log.append({"record_id": rec["record_id"], "action": f"{col}: {old} -> {new}", "doi": r["doi"],
                        "abbreviation": r["abbreviation_in_the_article"], "value": r["value"],
                        "reason": rec["reason"]})
    for rec in DROP:
        _, r = target(d, rec)
        log.append({"record_id": rec["record_id"], "action": "removed", "doi": r["doi"],
                    "abbreviation": r["abbreviation_in_the_article"], "value": r["value"],
                    "reason": rec["reason"]})
    d = d[~d["record_id"].isin([r["record_id"] for r in DROP])].copy()
    # consecutive ids for the new release; record_id in the changelog stays the v1.0 id
    new_id = {old: f"MLDB-{k:06d}" for k, old in enumerate(d["record_id"], 1)}
    pd.DataFrame({"record_id_v1.0": list(new_id), "record_id": list(new_id.values())}).to_csv(ID_MAP, index=False)
    d["record_id"] = d["record_id"].map(new_id)
    d.to_csv(OUT, index=False)
    log = pd.DataFrame(log).rename(columns={"record_id": "record_id_v1.0"})
    log.insert(1, "record_id", log["record_id_v1.0"].map(new_id).fillna(""))
    log.to_csv(CHANGELOG, index=False)
    print(f"{OUT.name}: {len(d)} rows, {d['doi'].nunique()} articles "
          f"({len(log)} changes, see {CHANGELOG.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
