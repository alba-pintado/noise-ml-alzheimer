"""Carga de datos y simulación.

El núcleo del pipeline trabaja con tres objetos sencillos para no depender de
una librería concreta:

    counts : matriz (células x genes) de cuentas crudas, densa o scipy.sparse
    obs    : DataFrame con una fila por célula (donante, tipo celular, ...)
    genes  : lista con los nombres de los genes

`load_h5ad` convierte un AnnData real a ese formato. `simulate` genera un
dataset de juguete con un efecto de ruido conocido para probar el pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Dataset:
    counts: sp.csr_matrix          # células x genes, cuentas crudas
    obs: pd.DataFrame              # metadatos por célula
    genes: list[str]
    donors: pd.DataFrame           # metadatos por donante (índice = id de donante)


def load_h5ad(path: str, donor_col: str, celltype_col: str, label_col: str,
              covariate_cols: list[str] | None = None, layer: str | None = None) -> Dataset:
    """Lee un .h5ad (p. ej. SEA-AD descargado de CELLxGENE).

    Usa las cuentas crudas: `layer` si se indica, si no `adata.raw.X` cuando
    exista, y si no `adata.X`. Comprueba que sean enteros, porque las métricas
    de ruido se calculan sobre cuentas, no sobre datos ya normalizados.
    """
    import anndata as ad  # import local: solo hace falta con datos reales

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
        raise ValueError("La matriz no parece de cuentas crudas (hay valores no enteros). "
                         "Indica la capa de cuentas con `layer`.")

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
    """Dataset sintético con un efecto SOLO de variabilidad.

    En los donantes con Alzheimer (label=1), `n_noisy_genes` genes de
    `noisy_celltype` tienen mayor sobredispersión (más ruido) pero la MISMA
    media. Un buen pipeline debe detectar el efecto con las features de ruido
    y no con las de media: es el control positivo del proyecto.
    """
    rng = np.random.default_rng(seed)
    genes = [f"G{i:04d}" for i in range(n_genes)]
    base_mean = rng.lognormal(mean=1.0, sigma=1.2, size=(len(celltypes), n_genes))
    base_disp = rng.uniform(0.05, 0.4, size=n_genes)           # sobredispersión NB
    noisy = rng.choice(n_genes, n_noisy_genes, replace=False)

    labels = rng.permutation(np.r_[np.zeros(n_donors // 2), np.ones(n_donors - n_donors // 2)]).astype(int)
    donors = pd.DataFrame({
        "label": labels,
        "age": rng.normal(82, 6, n_donors).round(),
        "sex": rng.choice(["F", "M"], n_donors),
    }, index=[f"D{i:03d}" for i in range(n_donors)])

    blocks, obs_rows = [], []
    for d, (donor, row) in enumerate(donors.iterrows()):
        depth = rng.lognormal(0, 0.35)                           # profundidad propia de cada donante
        for c, ct in enumerate(celltypes):
            n = int(rng.integers(int(cells_per_group * 0.6), int(cells_per_group * 1.4)))
            mu = base_mean[c] * depth * rng.lognormal(0, 0.1, n_genes)  # ruido biológico entre donantes
            disp = base_disp.copy()
            if row.label == 1 and ct == noisy_celltype:
                disp[noisy] *= noise_fold
            size_f = rng.lognormal(0, 0.3, n)[:, None]             # profundidad por célula
            m = mu[None, :] * size_f
            r = 1.0 / disp[None, :]
            lam = rng.gamma(shape=r, scale=m / r)                  # Gamma-Poisson = binomial negativa
            blocks.append(sp.csr_matrix(rng.poisson(lam)))
            obs_rows.append(pd.DataFrame({"donor": donor, "celltype": ct}, index=range(n)))

    counts = sp.vstack(blocks).tocsr()
    obs = pd.concat(obs_rows, ignore_index=True)
    donors.attrs["noisy_genes"] = [genes[i] for i in noisy]
    return Dataset(counts, obs, genes, donors)
