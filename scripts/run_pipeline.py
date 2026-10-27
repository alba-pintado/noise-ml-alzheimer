"""Ejecuta el pipeline completo.

    python scripts/run_pipeline.py --demo                 # datos simulados
    python scripts/run_pipeline.py --config config.yaml   # datos reales
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from noisead import data, features, model, plots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--demo", action="store_true", help="usar datos simulados")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(cfg["output_dir"]) / ("demo" if args.demo else "real")
    out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        ds = data.simulate(**cfg["demo"])
    else:
        d = cfg["data"]
        ds = data.load_h5ad(d["path"], d["donor_col"], d["celltype_col"], d["label_col"],
                            d.get("covariate_cols"), d.get("counts_layer"))
        pos = d.get("positive_values")
        if pos is not None:  # p. ej. ADNC "High"/"Intermediate" -> 1
            ds.donors["label"] = ds.donors["label"].astype(str).isin([str(v) for v in pos]).astype(int)
        keep_ct = d.get("celltypes")
        if keep_ct:
            mask = ds.obs["celltype"].isin(keep_ct).to_numpy()
            ds = data.Dataset(ds.counts[mask], ds.obs[mask].reset_index(drop=True), ds.genes, ds.donors)

    f = {**cfg["features"], **(cfg.get("demo_features", {}) if args.demo else {})}
    print(f"Células: {ds.counts.shape[0]:,} · genes: {ds.counts.shape[1]:,} · donantes: {len(ds.donors)}")
    feats = features.build_features(ds, f["target_umis"], f["min_cells"], f["n_cells"],
                                    f["min_mean"], f["n_genes_panel"], seed=cfg["seed"])
    st = feats.pop("_gene_stats")
    y = ds.donors.loc[feats["mean"].index, "label"].astype(int)
    print(f"Donantes con datos suficientes: {len(y)} (positivos: {int(y.sum())})")
    for k, v in feats.items():
        print(f"  {k:<13} {v.shape[1]:>6} features")

    covs = [c for c in (cfg.get("data", {}).get("covariate_cols") or ["age", "sex"]) if c in ds.donors]
    if covs:  # línea base de covariables: ¿el modelo aprende algo más que edad y sexo?
        feats["covariates"] = pd.get_dummies(ds.donors.loc[y.index, covs], drop_first=True).astype(float)

    m = cfg["model"]
    res = model.nested_cv(feats, y, kind=m["kind"], n_splits=m["n_splits"],
                          n_repeats=m["n_repeats"], seed=cfg["seed"])
    res.to_csv(out / "cv_results.csv", index=False)
    summary = (res.groupby(["features", "repeat"])["auc"].mean().groupby("features")
               .agg(["mean", "std"]).sort_values("mean", ascending=False))
    print("\nAUC (media ± sd entre repeticiones):")
    print(summary.round(3).to_string())

    # Comparación por pares en las mismas particiones: ¿el ruido mejora a la media?
    wide = res.pivot_table(index=["repeat", "fold"], columns="features", values="auc")
    paired = {}
    for other in ("noise", "noise_global", "combined"):
        diff = wide[other] - wide["mean"]
        paired[other] = {"mean_delta_auc": round(float(diff.mean()), 3),
                         "share_folds_better": round(float((diff > 0).mean()), 3)}
    print("\nFrente a la línea base de media:", json.dumps(paired, indent=2, ensure_ascii=False))

    best = summary.index[0]
    report = {"summary": summary.round(4).reset_index().to_dict("records"), "paired_vs_mean": paired}
    if m.get("n_perm", 0) > 0:
        p = model.permutation_pvalue(feats[best], y, summary.loc[best, "mean"], kind=m["kind"],
                                     n_perm=m["n_perm"], seed=cfg["seed"])
        report["permutation_p"] = {best: p}
        print(f"\nTest de permutación ({best}): p = {p:.3f}")

    top = model.top_features(feats["noise"], y, seed=cfg["seed"])
    top.to_csv(out / "top_noise_features.csv", header=["coef"])
    print("\nFeatures de ruido con más peso:\n" + top.round(3).to_string())
    if "noisy_genes" in ds.donors.attrs:  # en la demo sabemos la verdad
        truth = set(ds.donors.attrs["noisy_genes"])
        hits = sum(name.split("|")[2] in truth for name in top.index)
        report["demo_true_noisy_genes_in_top15"] = hits
        print(f"(demo) genes realmente ruidosos entre los 15 primeros: {hits}/15")

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plots.auc_comparison(res[res["features"] != "covariates"], str(out / "auc_comparison.png"),
                         "¿El ruido predice la patología mejor que la expresión media?")
    ct = cfg.get("plot_celltype") or st["celltype"].iloc[0]
    plots.mean_cv2(st, ct, str(out / f"mean_cv2_{ct}.png"))
    print(f"\nResultados en {out}/")


if __name__ == "__main__":
    main()
