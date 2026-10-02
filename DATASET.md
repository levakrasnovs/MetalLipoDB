# MetalLipoDB: the dataset

A curated dataset of experimental octanol–water lipophilicity values (log P / log D)
for metal complexes, compiled from the peer-reviewed literature.

- **5,158** measurements from **1,192** articles published 1997–2026
- **4,714** distinct complexes (**4,784** distinct complex–counterion pairs)
- **23** metals; **556** multinuclear complexes
- All values obtained by the shake-flask method
- Every record carries the structure, the measurement conditions as reported, the original
  method text and the source DOI

**Zenodo:** [10.5281/zenodo.23084573](https://doi.org/10.5281/zenodo.23084573) &nbsp;·&nbsp; **License:** CC BY 4.0

## Files

| File | Content |
|---|---|
| `MetalLipoDB.csv` | The dataset: one row per reported measurement, 28 columns, UTF-8, comma-separated |

## Loading

Read every column as text and treat only empty cells as missing:

```python
import pandas as pd

df = pd.read_csv("MetalLipoDB.csv", dtype=str,
                 keep_default_na=False, na_values=[""])
df["value"] = df["value"].astype(float)
```

With default settings pandas converts strings such as `NA` or `None` to missing values.
The dataset does not rely on those strings, but explicit settings keep the file as written.

All structures (`smiles_complex`, `smiles_ligands`, `counterion`) parse with RDKit using
default settings. To keep explicit hydride ligands (`[H-]`), parse without removing hydrogens:

```python
from rdkit import Chem

params = Chem.SmilesParserParams()
params.removeHs = False
mol = Chem.MolFromSmiles(df.loc[0, "smiles_complex"], params)
```

## Scope and inclusion criteria

A value is included when the article reports its **own** octanol–water shake-flask
measurement for a metal-containing compound and describes how it was obtained.

Not included:

- free organic ligands;
- values cited from other work, including a group's own earlier articles: when the same
  value and the same stated uncertainty for the same complex appear in several articles
  of one group, only the original article is kept;
- values without a traceable experimental procedure;
- articles whose Supporting Information contradicts the main text on the measured
  compounds or conditions;
- methods other than shake-flask (e.g. slow-stirring).

Partition coefficients reported as P<sub>ow</sub> were converted to log<sub>10</sub>.
A small number of values were read from figures where no numerical table was given.

## Columns

Missing values follow one rule across the table: an **empty cell** means the article does
not report the quantity; `absent` (in text columns only) means it is confirmed to be absent.

### Identifier

| Column | Description |
|---|---|
| `record_id` | Identifier `MLDB-000001` … `MLDB-005158`, one per measurement. The mapping to the identifiers of the first, withdrawn upload is given in `release_v1/record_id_map.csv`. |

### Structure

| Column | Description |
|---|---|
| `metal` | Metal element; heterometallic complexes list the metals joined by `-` (e.g. `Fe-Ru`) |
| `multinuclear` | `YES` for complexes with more than one metal centre, empty otherwise |
| `ccdc` | CCDC reference code of the crystal structure, when available |
| `smiles_complex` | SMILES of the complex ion or neutral complex. Metal–ligand bonds are written as dative bonds (`->`), η-bound rings as dative bonds from every ring atom. Canonicalised with RDKit. |
| `smiles_ligands` | SMILES of the ligands, dot-separated and sorted, with donor atoms in their bound charge state (e.g. `[c-]` for a cyclometalated carbon). Given for mononuclear complexes; empty for most multinuclear ones. |
| `counterion` | SMILES of the counterion(s) with stoichiometry, e.g. `[Cl-].[Cl-]` |
| `complex_charge` | Charge of the complex ion (integer) |
| `abbreviation_in_the_article` | Compound label used in the source article |

### Measurement

| Column | Description |
|---|---|
| `value` | Reported log P or log D (decimal log<sub>10</sub>) |
| `value_relation` | `=` (5,079), `>` (41), `<` (36), `~` (2) |
| `std` | Reported uncertainty, same units as `value` |
| `value_raw` | The value exactly as written in the article, including the uncertainty |
| `type_authors` | `logP` (4,189) or `logD` (969) — the name the authors gave the quantity. It is the authors' label, not a classification of the measurement (see *Notes*). |

### Conditions

| Column | Description |
|---|---|
| `method` | `shake-flask`; empty when the article does not name the method (73) |
| `detection` | How concentrations were determined: `UV-Vis`, `ICP-MS`, `Gamma counter`, `ICP-OES`, `GFAAS`, `FAAS`, `Fluorescence`, `HPLC`, `AAS`, `RP-UPLC`, `Solid scintillation counter`, `ICP spectroscopy`, `Dose calibrator` |
| `aqueous_phase` | Aqueous phase as described in the article, normalised in spelling (e.g. `water`, `PBS`, `0.9% NaCl`, `10 mM phosphate buffer`) |
| `medium_class` | Class of the aqueous phase: `water`, `salt / saline`, `buffer`, `buffer + salt`, `acid` |
| `pH` | pH of the aqueous phase |
| `temperature_C` | Temperature, °C |
| `buffer_class` | `Phosphate`, `Tris`, `HEPES`, `MOPS`, `Acetate`, `MES`, or `absent` |
| `buffer_conc_mM` | Buffer concentration, mmol/L |
| `halide_type` | Halide in the aqueous phase: `Cl`, `I`, `Br`, or `absent` |
| `halide_conc_mM` | Halide concentration, mmol/L (see *Notes*) |
| `cosolvent` | Co-solvent and its fraction as reported, e.g. `DMSO 1%` |
| `method_description` | The experimental procedure quoted from the article, without curator edits |

### Source

| Column | Description |
|---|---|
| `doi` | DOI of the source article, lower case |
| `year` | Year of first publication (Crossref `issued` date; for articles published online ahead of print, the online year) |

Numeric columns (`complex_charge`, `value`, `std`, `pH`, `temperature_C`, `buffer_conc_mM`,
`halide_conc_mM`, `year`) are written in their shortest exact decimal form (`7.4`, `154`),
without trailing zeros. The original notation of each value is kept in `value_raw`.

## Notes

**log P and log D.** `type_authors` records the authors' wording. In the literature "log P"
is often used for any octanol–water value, including those measured in buffers or for
charged complexes, where the measured quantity is a distribution coefficient or depends on
the counterion. The conditions needed to tell these cases apart are given in `pH`,
`buffer_class`, `medium_class` and `complex_charge`. For example, values that correspond to
the strict definition of log P (neutral complex, unbuffered aqueous phase) can be selected as:

```python
strict_logP = df[
    df["medium_class"].notna()
    & df["pH"].isna()
    & df["buffer_class"].isin(["absent"])
    & df["complex_charge"].eq("0")
]
```

**Counterions.** The same complex cation with different counterions can differ in measured
lipophilicity; records are identified by the pair `smiles_complex` + `counterion`. For 103
charged complexes the counterion is empty. Almost all of them are Tc (89) and Re (11) complexes,
mostly radiometric tracer measurements, for which no stoichiometric counterion exists.

**Halide concentration.** Where an article names the aqueous phase only as PBS or DPBS
without its composition, the chloride concentration is taken from the standard
recipe (137 mM NaCl + 2.7 mM KCl, recorded as 140 mM). Values of 0 mM for water, 154 mM for 0.9% NaCl and
6,100 mM for saturated NaCl are derived from the name of the aqueous phase. Other values are
as reported.

**Zero uncertainty.** `std = 0` (15 records) means the article reports "± 0.00" or
"± 0.0", i.e. an uncertainty below the reported rounding.

**Identifiers.** Standard InChI does not represent dative metal–ligand bonds, so no InChI
or InChIKey is given for the complexes. The canonical `smiles_complex` together with
`counterion` identifies a compound.

## Citation

If you use MetalLipoDB, please cite the accompanying article:

> Krasnov, L.; Malikov, D.; Kiseleva, M. A.; Lvov, A.; Tatarin, S. V.; Bezzubov, S. I.
> Beyond Organic Molecules: MetalLipoDB, a Lipophilicity Dataset for Metal Complexes and
> Machine Learning Benchmarks (reference will be added on publication).

## Contact

N.S. Kurnakov Institute of General and Inorganic Chemistry, Russian Academy of Sciences,
Moscow, Russia.
Lev Krasnov — lewa.krasnovs@gmail.com; Stanislav I. Bezzubov — bezzubov@igic.ras.ru
