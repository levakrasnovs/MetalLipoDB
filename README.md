# MetalLipoDB

Code and data for the article *Beyond Organic Molecules: MetalLipoDB, a Lipophilicity
Dataset for Metal Complexes and Machine Learning Benchmarks*.

MetalLipoDB contains 5158 experimental octanol–water logP and logD values of complexes of
23 metals from 1192 articles. The dataset is deposited on Zenodo
([10.5281/zenodo.23084573](https://doi.org/10.5281/zenodo.23084573)) and can be searched at
https://biometaldb.streamlit.app/. The columns are described in [DATASET.md](DATASET.md).

## Setup

```bash
conda env create -f environment.yml
conda activate metallipodb
python data/get_sangster.py   # SangsterLogP from Zenodo (10.5281/zenodo.19387552)
```

The scripts resolve their paths relative to the repository, so they can be run from any
directory. The figures are written to `figures/`.

## Repository

| Path | Content |
|---|---|
| `data/MetalLipoDB.csv` | The dataset, identical to the Zenodo file (MD5 `1e059644c5e16c42fe9b60cb293b9d55`) |
| `data/release_v1/` | First upload (`MetalLipoDB_v1.0.csv`), the corrections applied to it (`CHANGELOG.csv`) and the mapping of record identifiers (`record_id_map.csv`) |
| `data/release_corrections.py` | Applies the corrections to the first upload and writes `data/MetalLipoDB.csv` |
| `data/get_sangster.py` | Downloads SangsterLogP |
| `train/` | Machine learning models, validation folds, out-of-fold predictions and metrics (see [train/README.md](train/README.md)) |
| `plots/` | Analyses and figures of the article |
| `figures/` | Output of `plots/`; `figures/tsne/` holds the t-SNE embedding used in Figure 3 |

## Figures and tables

| Article | Script |
|---|---|
| Figure 2 (dataset composition, panels assembled manually) | `plots/plot_figures_metallipodb.py` |
| Figure 3 (chemical space) | `plots/metallipo_tsne_pipeline.py data/MetalLipoDB.csv --unique --outdir figures/tsne --top-n-legend 9 --value-col none --skip-html --skip-structure-grids`, then `plots/plot_tsne_annotated.py` |
| Figure 4 (inter-laboratory reproducibility) | `plots/plot_interlab.py` (pairs in `plots/interlab_pairs.py`) |
| Figure 5 (substituent increments) | `plots/plot_substituent_attenuation.py --rebuild` |
| Figure 6 (model comparison) | `train/stat_compare.py`, then `plots/plot_model_stats.py` |
| Figure 7 (unseen metals) | `plots/plot_ml_metal.py` |
| Figure 8 (time split, learning curve) | `train/time_split.py`, `train/learning_curve.py`, then `plots/plot_time_learning.py` |
| Figure 9 (practical use) | `train/analyze_domain.py`, `train/analyze_practical.py`, then `plots/plot_ml_practical.py` |
| Figure 10 (cytotoxicity) | `plots/plot_cytotox_logp.py` (see below) |
| SI figures | `plots/plot_si_years.py`, `plots/plot_counterion.py`, `plots/plot_metal_swap.py`, `plots/plot_conditions.py`, `plots/analyze_conditions.py`, `plots/classify_metal_from_ligands.py`, `plots/plot_substituent_examples.py`, `plots/plot_parity.py`, `plots/plot_shap_si.py` |

The models are trained by the scripts in `train/`. Their out-of-fold predictions are
included, so the figures can be rebuilt without retraining. `plots/plot_parity.py` and
`plots/plot_shap_si.py` need `train/data/MetalLipoDB_modeling.csv`, which is created by
`python prepare_data.py` in `train/`. TabPFN is used through the Prior Labs API, and
the token is read from the `TABPFN_TOKEN` environment variable.

`plots/plot_cytotox_logp.py` uses the IC50 values of MetalCytoToxDB (download `MetalCytoToxDB.csv`
from its Zenodo record into `data/`) and IC50 values extracted from the
MetalLipoDB articles, which will be released with the next version of MetalCytoToxDB.
Without the second file the figure is built from MetalCytoToxDB alone, and the counts are
smaller than in the article.

## License

Code: MIT. Data: CC BY 4.0.

## Citation

If you use MetalLipoDB, please cite the article (reference will be added on publication)
and the dataset ([10.5281/zenodo.23084573](https://doi.org/10.5281/zenodo.23084573)).
