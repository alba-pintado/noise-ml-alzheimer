"""Data loading and simulation.

The core pipeline works with three simple objects so that it does not depend
on any particular single-cell library:

    counts : (cells x genes) raw count matrix, dense or scipy.sparse
    obs    : DataFrame with one row per cell (donor, cell type, ...)
    genes  : list of gene names

`load_h5ad` converts a real AnnData file to that format. `simulate` builds a
toy dataset with a known noise effect to test the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Dataset:
    counts: sp.csr_matrix          # cells x genes, raw counts
    obs: pd.DataFrame              # per-cell metadata
    genes: list[str]
    donors: pd.DataFrame           # per-donor metadata (index = donor id)


def load_h5ad(path: str, donor_col: str, celltype_col: str, label_col: str,
              covariate_cols: list[str] | None = None, layer: str | None = None) -> Dataset:
    """Read an .h5ad file (e.g. SEA-AD downloaded from CELLxGENE).

    Uses raw counts: `layer` if given, otherwise `adata.raw.X` when present,
    otherwise `adata.X`. Checks that values are integers, because noise
    metrics are computed on counts, not on already-normalised data.
    """
    import anndata as ad  # local import: only needed for real data

    adata = ad.read_h5ad(path)
    if layer:
        X, genes = adata.layers[layer], list(adata.var_names)
    elif adata.raw is not None:
        X, genes = adata.raw.X, list(adata.raw.var_names)
    else:
        X, genes = adata.X, list(adata.var_names)
    X = sp.csr_matrix(X)
    sample = X.data[: min(10_000, X.data.size)]
    if sample.size and not np.allclose(sample, np.round(sample)):
        raise ValueError("The matrix does not look like raw counts (non-integer values found). "
                         "Point `layer` to the raw count layer.")

    obs = adata.obs[[donor_col, celltype_col]].copy()
    obs.columns = ["donor", "celltype"]
    obs["donor"] = obs["donor"].astype(str)
    obs["celltype"] = obs["celltype"].astype(str)

    cols = [label_col] + list(covariate_cols or [])
    donors = (adata.obs.assign(donor=obs["donor"].values)
              .groupby("donor", observed=True)[cols].first())
    donors = donors.rename(columns={label_col: "label"})
    return Dataset(X, obs.reset_index(drop=True), genes, donors)


def simulate(n_donors: int = 60, celltypes=("Neuron_exc", "Neuron_inh", "Microglia"),
             n_genes: int = 300, cells_per_group: int = 120, n_noisy_genes: int = 40,
             noisy_celltype: str = "Microglia", noise_fold: float = 1.6,
             seed: int = 0) -> Dataset:
    """Synthetic dataset with a variability-ONLY effect.

    In Alzheimer's donors (label=1), `n_noisy_genes` genes in `noisy_celltype`
    have higher overdispersion (more noise) but the SAME mean. A sound pipeline
    must detect the effect with the noise features and not with the mean
    features: this is the project's positive control.
    """
    rng = np.random.default_rng(seed)
    genes = [f"G{i:04d}" for i in range(n_genes)]
    base_mean = rng.lognormal(mean=1.0, sigma=1.2, size=(len(celltypes), n_genes))
    base_disp = rng.uniform(0.05, 0.4, size=n_genes)           # NB overdispersion
    noisy = rng.choice(n_genes, n_noisy_genes, replace=False)

    labels = rng.permutation(np.r_[np.zeros(n_donors // 2), np.ones(n_donors - n_donors // 2)]).astype(int)
    donors = pd.DataFrame({
        "label": labels,
        "age": rng.normal(82, 6, n_donors).round(),
        "sex": rng.choice(["F", "M"], n_donors),
    }, index=[f"D{i:03d}" for i in range(n_donors)])

    blocks, obs_rows = [], []
    for d, (donor, row) in enumerate(donors.iterrows()):
        depth = rng.lognormal(0, 0.35)                           # donor-specific sequencing depth
        for c, ct in enumerate(celltypes):
            n = int(rng.integers(int(cells_per_group * 0.6), int(cells_per_group * 1.4)))
            mu = base_mean[c] * depth * rng.lognormal(0, 0.1, n_genes)  # between-donor biological variation
            disp = base_disp.copy()
            if row.label == 1 and ct == noisy_celltype:
                disp[noisy] *= noise_fold
            size_f = rng.lognormal(0, 0.3, n)[:, None]             # per-cell depth
            m = mu[None, :] * size_f
            r = 1.0 / disp[None, :]
            lam = rng.gamma(shape=r, scale=m / r)                  # Gamma-Poisson = negative binomial
            blocks.append(sp.csr_matrix(rng.poisson(lam)))
            obs_rows.append(pd.DataFrame({"donor": donor, "celltype": ct}, index=range(n)))

    counts = sp.vstack(blocks).tocsr()
    obs = pd.concat(obs_rows, ignore_index=True)
    donors.attrs["noisy_genes"] = [genes[i] for i in noisy]
    return Dataset(counts, obs, genes, donors)
