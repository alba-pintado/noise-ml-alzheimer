"""Transcriptional noise metrics per donor and cell type.

Technical controls (the part interviewers look at most):

1. Equal depth: every cell is downsampled (binomial thinning) to the same
   number of UMIs, so a more deeply sequenced donor does not look "less
   noisy" as an artefact.
2. Equal cell numbers: every donor x cell type group is subsampled to the
   same number of cells, because the estimated variance depends on n.
3. Remove the mean-variance dependence: CV² falls with the mean in an
   expected way. We fit that trend across all groups and use the residual
   (noise "in excess" of what is expected for that mean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def downsample_counts(counts: sp.csr_matrix, target: int, rng: np.random.Generator) -> sp.csr_matrix:
    """Downsample every cell to `target` UMIs. Cells with fewer UMIs are dropped
    (also returns the mask of kept cells)."""
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
    """Mean, variance and distance to centroid per donor x cell type group.

    Returns a long DataFrame (donor, celltype, gene_idx, mean, var) and a
    dictionary {(donor, celltype): mean distance to centroid}.
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
        # Global noise in the style of Enge et al. 2017: Euclidean distance of each
        # cell to the group centroid in log1p space (all cells at the same depth).
        L = np.log1p(X)
        centroid_dist[(donor, ct)] = float(np.linalg.norm(L - L.mean(0), axis=1).mean())
    return pd.concat(rows, ignore_index=True), centroid_dist


def residual_cv2(stats: pd.DataFrame, min_mean: float, deg: int = 3) -> pd.DataFrame:
    """Add log CV² and its residual from the mean-CV² trend of each cell type.

    The trend is fitted per cell type with all donors pooled (a polynomial in
    log mean). It does not use the label, so no information leaks into the
    model.
    """
    st = stats[stats["mean"] >= min_mean].copy()
    st["log_mean"] = np.log10(st["mean"])
    st["log_cv2"] = np.log10(st["var"].clip(lower=1e-12) / st["mean"] ** 2)
    st["resid_cv2"] = np.nan
    for ct, sub in st.groupby("celltype", observed=True):
        coef = np.polyfit(sub["log_mean"], sub["log_cv2"], deg)
        st.loc[sub.index, "resid_cv2"] = sub["log_cv2"] - np.polyval(coef, sub["log_mean"])
    return st
