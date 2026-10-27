"""Modelos y evaluación.

Con pocas decenas de donantes, lo importante es la honestidad de la
evaluación, no el modelo más complejo:

- Validación cruzada anidada: el bucle interno elige hiperparámetros, el
  externo estima el rendimiento. Escalado e imputación van dentro del
  Pipeline, así que se ajustan solo con los datos de entrenamiento.
- Repeticiones con distintas particiones para ver la variabilidad del AUC.
- Las mismas particiones para todos los conjuntos de features, lo que permite
  compararlos por pares (¿el ruido mejora a la media en la misma partición?).
- Test de permutación de etiquetas para saber si el AUC supera al azar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str, seed: int = 0) -> GridSearchCV:
    if kind == "logreg":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("clf", LogisticRegression(solver="liblinear", max_iter=2000))])
        # l1_ratio=1 (L1) selecciona pocos genes, interpretable; 0 (L2) reparte el peso.
        grid = {"clf__C": [0.03, 0.3, 3.0], "clf__l1_ratio": [0.0, 1.0]}
    elif kind == "rf":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))])
        grid = {"clf__max_features": ["sqrt", 0.1], "clf__min_samples_leaf": [1, 3]}
    else:
        raise ValueError(kind)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=1)


def nested_cv(feature_sets: dict[str, pd.DataFrame], y: pd.Series, kind: str = "logreg",
              n_splits: int = 5, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """AUC por conjunto de features, repetición y fold (mismas particiones para todos)."""
    donors = y.index
    out = []
    for rep in range(n_repeats):
        outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for fold, (tr, te) in enumerate(outer.split(donors, y)):
            for name, X in feature_sets.items():
                X = X.loc[donors]
                m = make_model(kind, seed)
                m.fit(X.iloc[tr], y.iloc[tr])
                p = m.predict_proba(X.iloc[te])[:, 1]
                out.append({"features": name, "repeat": rep, "fold": fold,
                            "auc": roc_auc_score(y.iloc[te], p)})
    return pd.DataFrame(out)


def permutation_pvalue(X: pd.DataFrame, y: pd.Series, observed_auc: float, kind: str = "logreg",
                       n_perm: int = 50, seed: int = 0) -> float:
    """p-valor empírico: proporción de permutaciones con AUC >= observado."""
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_perm):
        yp = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        res = nested_cv({"perm": X}, yp, kind=kind, n_repeats=1, seed=seed + 1000 + i)
        null.append(res["auc"].mean())
    null = np.asarray(null)
    return float((1 + (null >= observed_auc).sum()) / (1 + n_perm))


def top_features(X: pd.DataFrame, y: pd.Series, k: int = 15, seed: int = 0) -> pd.Series:
    """Coeficientes del modelo ajustado con todos los donantes (solo para interpretar,
    nunca para evaluar)."""
    m = make_model("logreg", seed).fit(X, y)
    coef = pd.Series(m.best_estimator_.named_steps["clf"].coef_.ravel(), index=X.columns)
    return coef.reindex(coef.abs().sort_values(ascending=False).index).head(k)
