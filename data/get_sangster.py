"""
MetalLipoDB — get_sangster.py
=============================
Downloads the SangsterLogP dataset of organic compounds from Zenodo
(10.5281/zenodo.19387552) and saves its main sheet as SangsterLogP.csv, which is
used by plot_figures_metallipodb.py and plot_substituent_attenuation.py.

Usage:
    python get_sangster.py
"""

import io
import urllib.request
from pathlib import Path

import pandas as pd

URL = 'https://zenodo.org/api/records/19387552/files/Datasets.xlsx/content'
SHEET = 'SangsterLogP (N=23,520)'
OUT = Path(__file__).resolve().parent / 'SangsterLogP.csv'


def main():
    with urllib.request.urlopen(URL) as r:
        x = pd.read_excel(io.BytesIO(r.read()), sheet_name=SHEET)
    x.to_csv(OUT, index=False)
    print(f'saved: {OUT} ({len(x)} rows)')


if __name__ == '__main__':
    main()
