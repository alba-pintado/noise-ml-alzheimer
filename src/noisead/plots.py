"""Project figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["mean", "noise_global", "noise", "combined"]
LABELS = {"mean": "Mean\n(baseline)", "noise_global": "Global noise",
          "noise": "Per-gene noise", "combined": "Mean + noise"}
COLORS = {"mean": "#8a94a6", "noise_global": "#5b8def", "noise": "#1f5fbf", "combined": "#123a75"}


def auc_comparison(res: pd.DataFrame, path: str, title: str) -> None:
    per_rep = res.groupby(["features", "repeat"])["auc"].mean().reset_index()
    names = [n for n in ORDER if n in per_rep["features"].unique()]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, n in enumerate(names):
        v = per_rep.loc[per_rep["features"] == n, "auc"].to_numpy()
        ax.scatter(np.full(v.size, i) + np.random.default_rng(i).uniform(-0.08, 0.08, v.size),
                   v, s=22, color=COLORS[n], alpha=0.6, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=COLORS[n], lw=2.5, zorder=4)
        ax.text(i, 1.02, f"{v.mean():.2f}", ha="center", fontsize=10, color=COLORS[n], weight="bold")
    ax.axhline(0.5, ls="--", color="#b0b6c2", lw=1)
    ax.text(len(names) - 0.5, 0.505, "chance", fontsize=8, color="#8a94a6", ha="right", va="bottom")
    ax.set_xticks(range(len(names)), [LABELS[n] for n in names])
    ax.set_ylim(0.3, 1.08)
    ax.set_ylabel("AUC (nested CV, mean per repeat)")
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def mean_cv2(st: pd.DataFrame, celltype: str, path: str) -> None:
    sub = st[st["celltype"] == celltype]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.scatter(sub["log_mean"], sub["log_cv2"], s=3, alpha=0.15, color="#5b8def", rasterized=True)
    xs = np.linspace(sub["log_mean"].min(), sub["log_mean"].max(), 200)
    coef = np.polyfit(sub["log_mean"], sub["log_cv2"], 3)
    ax.plot(xs, np.polyval(coef, xs), color="#123a75", lw=2, label="expected trend")
    ax.set_xlabel("log10 mean")
    ax.set_ylabel("log10 CV²")
    ax.set_title(f"{celltype}: the residual from the curve is the 'excess' noise", fontsize=10, loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
