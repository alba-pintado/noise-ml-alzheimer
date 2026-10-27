"""Tablas de features a nivel de DONANTE.

Clave del diseño: el modelo predice el estado de cada donante, no de cada
célula. Las células de un mismo donante no son muestras independientes; usar
células como filas inflaría el rendimiento (pseudorreplicación).

Conjuntos de features que se comparan, todos sobre el mismo panel de genes:
    mean        expresión media (log) por gen x tipo celular  -> línea base
    noise       residuo de CV² por gen x tipo celular
    noise_global ruido global por tipo celular (mediana del residuo y
                 distancia al centroide): pocas features, muy interpretables
    combined    mean + noise: ¿el ruido aporta algo ADEMÁS de la media?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset
from .noise import downsample_counts, group_stats, residual_cv2


def build_features(ds: Dataset, target_umis: int, min_cells: int, n_cells: int,
                   min_mean: float, n_genes_panel: int, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    counts, keep = downsample_counts(ds.counts, target_umis, rng)
    obs = ds.obs.loc[keep].reset_index(drop=True)

    stats, cdist = group_stats(counts, obs, min_cells=min_cells, n_cells=n_cells, rng=rng)
    st = residual_cv2(stats, min_mean=min_mean)
    st["gene"] = np.asarray(ds.genes)[st["gene_idx"].to_numpy()]

    # Panel: genes presentes en todos los grupos de su tipo celular y con mayor
    # media; se elige sin mirar la etiqueta.
    n_groups = st.groupby("celltype", observed=True)["donor"].nunique()
    cover = st.groupby(["celltype", "gene"], observed=True).agg(n=("donor", "nunique"), m=("mean", "mean")).reset_index()
    cover = cover[cover["n"] == cover["celltype"].map(n_groups)]
    panel = (cover.sort_values("m", ascending=False).groupby("celltype", observed=True)
             .head(n_genes_panel)[["celltype", "gene"]])
    st = st.merge(panel, on=["celltype", "gene"])

    def wide(value: str, prefix: str) -> pd.DataFrame:
        w = st.pivot_table(index="donor", columns=["celltype", "gene"], values=value, observed=True)
        w.columns = [f"{prefix}|{ct}|{g}" for ct, g in w.columns]
        return w

    mean = wide("log_mean", "mean")
    noise = wide("resid_cv2", "noise")

    glob = st.groupby(["donor", "celltype"], observed=True)["resid_cv2"].median().unstack()
    glob.columns = [f"noise_median|{c}" for c in glob.columns]
    cd = pd.Series(cdist).rename_axis(["donor", "celltype"]).unstack()
    cd.columns = [f"centroid_dist|{c}" for c in cd.columns]
    noise_global = glob.join(cd)

    donors = ds.donors.index.intersection(mean.index)
    feats = {
        "mean": mean.loc[donors],
        "noise": noise.loc[donors],
        "noise_global": noise_global.loc[donors],
        "combined": mean.loc[donors].join(noise.loc[donors]),
    }
    feats["_gene_stats"] = st  # para figuras y diagnóstico
    return feats
