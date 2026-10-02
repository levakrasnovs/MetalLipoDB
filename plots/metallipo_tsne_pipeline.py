#!/usr/bin/env python3
"""
metallipo_tsne_pipeline.py
==========================
Map of the chemical space of metal complexes (or any molecules given as SMILES
with a categorical and an optional numeric label):

  1. count Morgan fingerprints (unlike binary ones they keep the stoichiometry:
     one and three bipyridines give the same binary but different count vectors)
  2. log1p scaling and PCA to N components (removes noise before t-SNE)
  3. t-SNE (metric='cosine', init='pca'), 2D layout
  4. KMeans clustering in the same PCA space, not on the t-SNE coordinates,
     since distances between t-SNE clusters are not meaningful
  5. export: t-SNE map coloured by label and value, cluster map, structure grids
     per cluster (RDKit) and a self-contained interactive HTML

Usage (Figure 2 of the paper, one point per unique complex):
  python plots/metallipo_tsne_pipeline.py data/MetalLipoDB.csv --unique --outdir figures/tsne \
      --top-n-legend 9 --value-col none --skip-html --skip-structure-grids

Required columns: SMILES (fragments separated by '.'), a categorical label
(e.g. metal) and, optionally, a numeric value for the second colouring.
"""

import argparse
import base64
import io
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from PIL import Image

RDLogger.DisableLog("rdApp.*")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path", help="Input CSV")
    p.add_argument("--smiles-col", default="smiles_ligands",
                    help="SMILES column (default: smiles_ligands)")
    p.add_argument("--label-col", default="metal",
                    help="Categorical column for colouring (default: metal)")
    p.add_argument("--value-col", default="value",
                    help="Numeric column for the second colouring (default: value)")
    p.add_argument("--outdir", default="./tsne_output", help="Output directory")
    p.add_argument("--fp-size", type=int, default=2048, help="Morgan fingerprint size")
    p.add_argument("--fp-radius", type=int, default=2, help="Morgan fingerprint radius")
    p.add_argument("--pca-components", type=int, default=50,
                    help="Number of PCA components before t-SNE")
    p.add_argument("--perplexity", type=float, default=40, help="t-SNE perplexity")
    p.add_argument("--k-clusters", type=int, default=14, help="Number of KMeans clusters")
    p.add_argument("--value-label", default=None,
                    help="Colour bar label of panel B (default: name of value-col)")
    p.add_argument("--top-n-legend", type=int, default=9,
                    help="Number of label-col categories shown in their own colour (the rest -> 'other')")
    p.add_argument("--structures-per-cluster", type=int, default=12,
                    help="Structures per cluster grid")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--unique", action="store_true",
                   help="One point per unique complex (metal + ligand set), value = median of exact values")
    p.add_argument("--skip-html", action="store_true", help="Do not build the interactive HTML (large)")
    p.add_argument("--skip-structure-grids", action="store_true",
                    help="Do not draw the per-cluster structure grids")
    return p.parse_args()


def compute_fingerprints(smiles_list, radius, fp_size):
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps, keep_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(mg.GetCountFingerprintAsNumPy(mol))
        keep_idx.append(i)
    X = np.vstack(fps).astype(np.float32)
    return X, keep_idx


def build_embedding(X, pca_components, perplexity, random_state):
    Xl = np.log1p(X)
    n_comp = min(pca_components, Xl.shape[0] - 1, Xl.shape[1])
    Xp = PCA(n_components=n_comp, random_state=random_state).fit_transform(Xl)
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca",
               random_state=random_state, metric="cosine").fit_transform(Xp)
    return emb, Xp


def plot_cluster_map(emb, labels, outpath, k):
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab20")
    for c in range(k):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=8, color=cmap(c % 20), lw=0)
        if m.sum() > 0:
            cx, cy = emb[m, 0].mean(), emb[m, 1].mean()
            ax.text(cx, cy, str(c), fontsize=13, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle", fc="white", ec="black", alpha=0.85, pad=0.15))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"t-SNE + KMeans (K={k}; clustering performed in PCA space)", fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close(fig)


METAL_ORDER = ['Ru', 'Pt', 'Ir', 'Au', 'Cu', 'Tc', 'Rh', 'Re', 'Os']


def plot_tsne_map(emb, dd, label_col, value_col, top_n_legend, outpath,
                     value_label=None):
    top = dd[label_col].value_counts().index[:top_n_legend]
    # fixed order, so that a metal keeps its colour when the counts change
    top = sorted(top, key=lambda m: METAL_ORDER.index(m) if m in METAL_ORDER else len(METAL_ORDER))
    # tab10 without its grey (index 7): grey is reserved for "other"
    tab = plt.get_cmap("tab10").colors
    palette = [c for i, c in enumerate(tab) if i != 7]
    cols = lambda i: palette[i % len(palette)]
    # one panel per thing actually drawn: A always, B only when there is a value
    # column. An extra axis here would render as blank space on the right.
    has_value = value_col in dd.columns
    fig, axes = plt.subplots(1, 2 if has_value else 1,
                             figsize=(13 if has_value else 6.8, 6.2))
    if not hasattr(axes, "__len__"):
        axes = [axes]

    a = axes[0]
    oth = ~dd[label_col].isin(top)
    a.scatter(emb[oth.values, 0], emb[oth.values, 1], s=7, c="#d8d8d8", lw=0, label="other")
    for k_i, name in enumerate(top):
        msk = (dd[label_col] == name).values
        a.scatter(emb[msk, 0], emb[msk, 1], s=9, color=cols(k_i), lw=0,
                  label=f"{name} ({msk.sum()})", alpha=0.8)
    a.legend(fontsize=8, markerscale=2, loc="best", frameon=False)
    if has_value:
        a.text(-0.02, 1.02, "a)", transform=a.transAxes, fontsize=18, ha="left", va="bottom")

    if has_value and len(axes) > 1:
        b = axes[1]
        vals = pd.to_numeric(dd[value_col], errors="coerce").values
        nanmask = np.isnan(vals)
        b.scatter(emb[nanmask, 0], emb[nanmask, 1], s=7, c="#eee", lw=0)
        sc = b.scatter(emb[~nanmask, 0], emb[~nanmask, 1], s=9,
                       c=vals[~nanmask], cmap="coolwarm", vmin=-3, vmax=3, lw=0)
        plt.colorbar(sc, ax=b, label=value_label or value_col)
        b.text(-0.02, 1.02, "b)", transform=b.transAxes, fontsize=18, ha="left", va="bottom")

    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=400)
    plt.close(fig)


def save_cluster_structure_grids(dd, labels, k, outdir, smiles_col, label_col, value_col, n_per_cluster, random_state):
    for c in range(k):
        sub = dd[labels == c]
        if len(sub) == 0:
            continue
        n_show = min(n_per_cluster, len(sub))
        sample = sub.sample(n_show, random_state=random_state) if len(sub) > n_show else sub
        mols, legends = [], []
        for _, row in sample.iterrows():
            mol = Chem.MolFromSmiles(row[smiles_col])
            if mol is None:
                continue
            mols.append(mol)
            label_txt = str(row[label_col])
            if value_col in dd.columns and pd.notna(row.get(value_col, np.nan)):
                try:
                    label_txt += f" | {value_col}={float(row[value_col]):.2f}"
                except (TypeError, ValueError):
                    pass
            legends.append(label_txt)
        if not mols:
            continue
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(260, 220),
                                    legends=legends, useSVG=False)
        img.save(f"{outdir}/cluster_{c:02d}_structures.png")


def build_interactive_html(dd, emb, labels, smiles_col, label_col, value_col, outpath,
                            img_size=(90, 70), palette_colors=8):
    x = emb[:, 0]; y = emb[:, 1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-9) * 1000
    y = (y - y.min()) / (y.max() - y.min() + 1e-9) * 800

    records = []
    for i, (_, row) in enumerate(dd.iterrows()):
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            continue
        d2d = Draw.rdMolDraw2D.MolDraw2DCairo(*img_size)
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        png_bytes = d2d.GetDrawingText()
        im = Image.open(io.BytesIO(png_bytes)).convert("P", palette=Image.ADAPTIVE, colors=palette_colors)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        v = None
        if value_col in dd.columns:
            try:
                vv = float(row[value_col])
                if not np.isnan(vv):
                    v = round(vv, 2)
            except (TypeError, ValueError):
                pass

        records.append({
            "x": round(float(x[i]), 2),
            "y": round(float(y[i]), 2),
            "c": int(labels[i]),
            "m": str(row[label_col]),
            "v": v,
            "img": b64,
        })

    data_str = json.dumps(records)

    html_template = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Interactive t-SNE map</title>
<style>
  body { margin:0; font-family: -apple-system, Segoe UI, Arial, sans-serif; background:#fff; color:#222; }
  #wrap { display:flex; }
  #canvasWrap { position:relative; flex: 1 1 auto; }
  svg { width:100%; height:100vh; display:block; cursor: crosshair; }
  .pt { transition: opacity .1s; }
  #tooltip {
    position:absolute; pointer-events:none; display:none;
    background:#fff; border:1px solid #ccc; border-radius:6px;
    box-shadow:0 4px 14px rgba(0,0,0,.15); padding:6px 8px; font-size:12px;
    z-index:10; text-align:center;
  }
  #tooltip img { display:block; margin:0 auto 4px auto; }
  #panel {
    width:230px; flex: 0 0 230px; padding:14px; border-left:1px solid #eee;
    height:100vh; overflow-y:auto; box-sizing:border-box; font-size:12.5px;
  }
  #panel h3 { margin:6px 0 4px 0; font-size:13px; }
  .legend-item { display:flex; align-items:center; gap:6px; margin:3px 0; cursor:pointer; user-select:none; }
  .legend-item.dim { opacity:.35; }
  .swatch { width:11px; height:11px; border-radius:50%; flex:0 0 auto; }
  select { width:100%; margin-bottom:10px; }
  .stat { color:#666; font-size:11px; margin-top:8px;}
</style>
</head>
<body>
<div id="wrap">
  <div id="canvasWrap">
    <svg id="svg" viewBox="-40 -40 1080 880"></svg>
    <div id="tooltip"></div>
  </div>
  <div id="panel">
    <h3>Colour by</h3>
    <select id="colorMode">
      <option value="cluster">Cluster (chemistry)</option>
      <option value="metal">Label</option>
      <option value="value">Value</option>
    </select>
    <div id="legend"></div>
    <div class="stat" id="stat"></div>
  </div>
</div>
<script>
const DATA = __DATA__;

const svg = document.getElementById('svg');
const tooltip = document.getElementById('tooltip');
const legend = document.getElementById('legend');
const colorModeSel = document.getElementById('colorMode');
const stat = document.getElementById('stat');

const CLUSTER_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
"#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78","#98df8a","#ff9896",
"#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5","#393b79"];

const metalSet = [...new Set(DATA.map(d=>d.m))].sort();
const METAL_COLORS = {};
const palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2",
"#7f7f7f","#bcbd22","#17becf","#f781bf","#a6d854","#fc8d62","#8da0cb","#e78ac3",
"#66c2a5","#ffd92f","#e5c494","#b3b3b3","#1b9e77","#d95f02","#7570b3","#e7298a",
"#66a61e","#e6ab02"];
metalSet.forEach((m,i)=> METAL_COLORS[m] = palette[i % palette.length]);

function valueColor(v){
  if (v===null) return '#dddddd';
  const t = Math.max(-3, Math.min(3, v));
  const norm = (t+3)/6;
  function lerp(a,b,f){return Math.round(a+(b-a)*f);}
  let r,g,b;
  if(norm<0.5){
    const f = norm/0.5;
    r=lerp(60,240,f); g=lerp(90,240,f); b=lerp(220,240,f);
  } else {
    const f=(norm-0.5)/0.5;
    r=lerp(240,190,f); g=lerp(240,50,f); b=lerp(240,45,f);
  }
  return `rgb(${r},${g},${b})`;
}

function colorFor(d, mode){
  if (mode==='cluster') return CLUSTER_COLORS[d.c % CLUSTER_COLORS.length];
  if (mode==='metal') return METAL_COLORS[d.m];
  return valueColor(d.v);
}

let activeFilter = null;

function render(){
  const mode = colorModeSel.value;
  svg.innerHTML = '';
  const frag = document.createDocumentFragment();
  DATA.forEach((d, idx) => {
    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx', d.x);
    c.setAttribute('cy', 800 - d.y);
    c.setAttribute('r', 3.2);
    c.setAttribute('fill', colorFor(d, mode));
    c.setAttribute('fill-opacity', 0.8);
    c.classList.add('pt');
    c.dataset.idx = idx;

    let visible = true;
    if (activeFilter){
      if (activeFilter.mode==='cluster' && d.c !== activeFilter.key) visible=false;
      if (activeFilter.mode==='metal' && d.m !== activeFilter.key) visible=false;
    }
    c.style.opacity = visible ? 1 : 0.06;

    c.addEventListener('mouseenter', (e)=>{
      tooltip.style.display = 'block';
      tooltip.innerHTML = `<img src="data:image/png;base64,${d.img}" width="90" height="70">
        <div><b>${d.m}</b> ${d.v!==null ? ('| value='+d.v) : ''}</div>
        <div style="color:#888">cluster ${d.c}</div>`;
    });
    c.addEventListener('mousemove', (e)=>{
      tooltip.style.left = (e.pageX + 14) + 'px';
      tooltip.style.top = (e.pageY + 14) + 'px';
    });
    c.addEventListener('mouseleave', ()=>{ tooltip.style.display='none'; });

    frag.appendChild(c);
  });
  svg.appendChild(frag);
  renderLegend(mode);
  stat.textContent = `${DATA.length} points`;
}

function renderLegend(mode){
  legend.innerHTML = '';
  if (mode==='value'){
    legend.innerHTML = '<div style="font-size:11px;color:#666">gradient: blue (low) → red (high)</div>';
    return;
  }
  if (mode==='cluster'){
    const clusters = [...new Set(DATA.map(d=>d.c))].sort((a,b)=>a-b);
    clusters.forEach(cl=>{
      const n = DATA.filter(d=>d.c===cl).length;
      const item = document.createElement('div');
      item.className = 'legend-item';
      if (activeFilter && activeFilter.mode==='cluster' && activeFilter.key!==cl) item.classList.add('dim');
      item.innerHTML = `<span class="swatch" style="background:${CLUSTER_COLORS[cl % CLUSTER_COLORS.length]}"></span> cluster ${cl} (${n})`;
      item.onclick = ()=>{
        activeFilter = (activeFilter && activeFilter.mode==='cluster' && activeFilter.key===cl) ? null : {mode:'cluster', key:cl};
        render();
      };
      legend.appendChild(item);
    });
  } else if (mode==='metal'){
    const top = metalSet.map(m=>[m, DATA.filter(d=>d.m===m).length]).sort((a,b)=>b[1]-a[1]);
    top.forEach(([m,n])=>{
      const item = document.createElement('div');
      item.className = 'legend-item';
      if (activeFilter && activeFilter.mode==='metal' && activeFilter.key!==m) item.classList.add('dim');
      item.innerHTML = `<span class="swatch" style="background:${METAL_COLORS[m]}"></span> ${m} (${n})`;
      item.onclick = ()=>{
        activeFilter = (activeFilter && activeFilter.mode==='metal' && activeFilter.key===m) ? null : {mode:'metal', key:m};
        render();
      };
      legend.appendChild(item);
    });
  }
}

colorModeSel.addEventListener('change', ()=>{ activeFilter=null; render(); });
render();
</script>
</body>
</html>
"""
    html = html_template.replace("__DATA__", data_str)
    with open(outpath, "w") as f:
        f.write(html)


def main():
    args = parse_args()
    import os
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    required = [args.smiles_col, args.label_col]
    for col in required:
        if col not in df.columns:
            sys.exit(f"Column '{col}' not found in the CSV. Available columns: {list(df.columns)}")

    d = df[df[args.smiles_col].notna() & df[args.label_col].notna()].copy()
    if args.value_col in d.columns:
        d[args.value_col] = pd.to_numeric(d[args.value_col], errors="coerce")
        # a bound (">3.7") is not the number 3.7: leave these points uncoloured on panel B
        if "value_relation" in d.columns:
            d.loc[d["value_relation"].isin([">", "<"]), args.value_col] = np.nan

    if args.unique:
        exact = d["value_relation"].isin(["=", "~"]) if "value_relation" in d.columns else True
        med = d[exact].groupby([args.label_col, args.smiles_col])[args.value_col].median() \
            if args.value_col in d.columns else None
        d = d.drop_duplicates([args.label_col, args.smiles_col]).copy()
        if med is not None:
            d[args.value_col] = [med.get(k) for k in zip(d[args.label_col], d[args.smiles_col])]
    print(f"Rows with SMILES and label: {len(d)}")

    X, keep_idx = compute_fingerprints(d[args.smiles_col].tolist(), args.fp_radius, args.fp_size)
    dd = d.iloc[keep_idx].reset_index(drop=True)
    print(f"SMILES parsed by RDKit: {len(dd)}")

    print("t-SNE embedding (log1p -> PCA -> t-SNE cosine)...")
    emb, Xp = build_embedding(X, args.pca_components, args.perplexity, args.random_state)

    print(f"KMeans (K={args.k_clusters}) in PCA space...")
    km = KMeans(n_clusters=args.k_clusters, random_state=args.random_state, n_init=10).fit(Xp)
    labels = km.labels_
    dd["cluster"] = labels
    dd["tsne_x"] = emb[:, 0]
    dd["tsne_y"] = emb[:, 1]

    dd.to_csv(f"{args.outdir}/embedding_and_clusters.csv", index=False)
    print(f"Saved: {args.outdir}/embedding_and_clusters.csv")

    plot_cluster_map(emb, labels, f"{args.outdir}/clusters_map.png", args.k_clusters)
    print(f"Saved: {args.outdir}/clusters_map.png")

    plot_tsne_map(emb, dd, args.label_col, args.value_col, args.top_n_legend,
                      f"{args.outdir}/tsne_map.png", value_label=args.value_label)
    print(f"Saved: {args.outdir}/tsne_map.png")

    if not args.skip_structure_grids:
        print("Drawing structure grids per cluster (RDKit)...")
        save_cluster_structure_grids(dd, labels, args.k_clusters, args.outdir,
                                      args.smiles_col, args.label_col, args.value_col,
                                      args.structures_per_cluster, args.random_state)
        print(f"Saved: {args.outdir}/cluster_XX_structures.png ({args.k_clusters} files)")

    if not args.skip_html:
        print("Building the interactive HTML (slowest step, one image per point)...")
        build_interactive_html(dd, emb, labels, args.smiles_col, args.label_col, args.value_col,
                                f"{args.outdir}/interactive_map.html")
        print(f"Saved: {args.outdir}/interactive_map.html")

    print("\nDone.")


if __name__ == "__main__":
    main()
