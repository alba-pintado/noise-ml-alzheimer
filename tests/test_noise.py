"""Quick tests. Run with `pytest` or `python tests/test_noise.py`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from noisead import data, features, noise  # noqa: E402


def test_downsample_keeps_target_depth():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(5, size=(50, 200)))
    sub, keep = noise.downsample_counts(X, 500, rng)
    totals = np.asarray(sub.sum(1)).ravel()
    assert keep.sum() == sub.shape[0]
    assert abs(totals.mean() - 500) < 25            # binomial: ~500 on average


def test_residual_removes_mean_trend():
    rng = np.random.default_rng(1)
    m = 10 ** rng.uniform(-1, 1.5, 2000)
    st = pd.DataFrame({"donor": "D", "celltype": "A", "gene_idx": range(2000),
                       "mean": m, "var": m + 0.2 * m ** 2})  # NB variance
    r = noise.residual_cv2(st, min_mean=0.0)
    assert abs(np.corrcoef(r["log_mean"], r["resid_cv2"])[0, 1]) < 0.05


def test_features_are_per_donor_and_have_no_label_columns():
    ds = data.simulate(n_donors=12, n_genes=80, cells_per_group=80, seed=3)
    f = features.build_features(ds, target_umis=400, min_cells=40, n_cells=40,
                                min_mean=0.05, n_genes_panel=50)
    f.pop("_gene_stats")
    for name, X in f.items():
        assert X.index.is_unique, name
        assert X.shape[0] <= 12, name
        assert not any("label" in c for c in X.columns), name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
