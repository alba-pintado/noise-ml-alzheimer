"""Métricas de ruido transcripcional por donante y tipo celular.

Controles técnicos (la parte que más se mira en una entrevista):

1. Igualar profundidad: cada célula se submuestrea (binomial thinning) a la
   misma cantidad de UMIs. Así un donante secuenciado con más profundidad no
   parece "menos ruidoso" por artefacto.
2. Igualar número de células: cada grupo donante x tipo celular se submuestrea
   al mismo número de células, porque la varianza estimada depende de n.
3. Quitar la dependencia media-varianza: el CV² baja con la media de forma
   esperada. Ajustamos esa tendencia con todos los grupos y usamos el
   residuo (ruido "extra" respecto a lo esperado para esa media).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def downsample_counts(counts: sp.csr_matrix, target: int, rng: np.random.Generator) -> sp.csr_matrix:
    """Submuestrea cada célula a `target` UMIs. Las células con menos UMIs se descartan
    (devuelve también la máscara de células conservadas)."""
    counts = sp.csr_matrix(counts, dtype=np.int64)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    keep = totals >= target
    sub = counts[keep].copy()
    p = target / np.repeat(totals[keep], np.diff(sub.indptr))
    sub.data = rng.binomial(sub.data, p)
    sub.eliminate_zeros()
    return sub, keep


def group_stats(counts: sp.csr_matrix, obs: pd.DataFrame, min_cells: int,
                n_cells: int | None, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Media, varianza y distancia al centroide por grupo donante x tipo celular.

    Devuelve un DataFrame largo (donor, celltype, gene_idx, mean, var) y un
    diccionario {(donor, celltype): distancia media al centroide}.
    """
    rows, centroid_dist = [], {}
    for (donor, ct), idx in obs.groupby(["donor", "celltype"], observed=True).indices.items():
        if len(idx) < min_cells:
            continue
        if n_cells is not None:
            idx = rng.choice(idx, n_cells, replace=False) if len(idx) >= n_cells else None
            if idx is None:
                continue
        X = counts[idx].toarray().astype(float)
        mean, var = X.mean(0), X.var(0, ddof=1)
        rows.append(pd.DataFrame({"donor": donor, "celltype": ct,
                                  "gene_idx": np.arange(X.shape[1]), "mean": mean, "var": var}))
        # Ruido global estilo Enge et al. 2017: distancia euclídea de cada célula al
        # centroide del grupo en espacio log1p (todas con la misma profundidad).
        L = np.log1p(X)
        centroid_dist[(donor, ct)] = float(np.linalg.norm(L - L.mean(0), axis=1).mean())
    return pd.concat(rows, ignore_index=True), centroid_dist


def residual_cv2(stats: pd.DataFrame, min_mean: float, deg: int = 3) -> pd.DataFrame:
    """Añade log CV² y su residuo respecto a la tendencia media-CV² de cada tipo celular.

    La tendencia se ajusta por tipo celular con todos los donantes juntos (un
    polinomio en log-media). Nota: la tendencia no usa la etiqueta, así que no
    hay fuga de información hacia el modelo.
    """
    st = stats[stats["mean"] >= min_mean].copy()
    st["log_mean"] = np.log10(st["mean"])
    st["log_cv2"] = np.log10(st["var"].clip(lower=1e-12) / st["mean"] ** 2)
    st["resid_cv2"] = np.nan
    for ct, sub in st.groupby("celltype", observed=True):
        coef = np.polyfit(sub["log_mean"], sub["log_cv2"], deg)
        st.loc[sub.index, "resid_cv2"] = sub["log_cv2"] - np.polyval(coef, sub["log_mean"])
    return st
