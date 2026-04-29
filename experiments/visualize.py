"""Visualization: CD diagrams, bar charts, PR distributions, calibration, k-sensitivity."""

from __future__ import annotations

import io as _io
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging

logger = setup_logging()

LOWER_IS_BETTER = {"log_loss", "brier_score", "ece"}

# ---------------------------------------------------------------------------
# Journal style constants
# ---------------------------------------------------------------------------

# Two-class palette (used for any head-to-head comparison)
BASE_COLOR = "#1f77b4"   # muted blue  — GaussianNB / baseline
PWNB_COLOR = "#d94801"   # deep orange — PW-NB (proposed method)

# Per-classifier colors: baselines → blue family; fixed-k PW-NB → orange family;
# PW-NB(auto) → purple so it reads as its own family.
CLF_PALETTE = {
    "GaussianNB":    "#1f77b4",
    "BernoulliNB":   "#6baed6",
    "MultinomialNB": "#9ecae1",
    "ComplementNB":  "#c6dbef",
    "PW-NB(k=5)":   "#fdae6b",
    "PW-NB(k=15)":  "#f16913",
    "PW-NB(k=30)":  "#d94801",
    "PW-NB(k=45)":  "#8c2d04",
    "PW-NB(auto)":  "#7b2d8b",
}

# ---------------------------------------------------------------------------
# Font detection
# Priority 1 — Computer Modern Roman via actual LaTeX (text.usetex=True)
#   requires: latex + dvipng (or dvisvgm) in PATH, AND a working install
# Priority 2 — STIX Two Text: designed for scientific publishing, closely
#   matches CM metrics, bundled with matplotlib >= 3.2, no LaTeX needed
# ---------------------------------------------------------------------------

def _probe_latex() -> bool:
    """Return True only if a full matplotlib→LaTeX render actually succeeds.

    Checks binary presence first (fast), then does a tiny test render so
    broken LaTeX installs (e.g. MiKTeX out-of-sync) are detected at import
    time rather than silently crashing every figure save.
    """
    if shutil.which("latex") is None:
        return False
    if shutil.which("dvipng") is None and shutil.which("dvisvgm") is None:
        return False
    try:
        import matplotlib
        import matplotlib.figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        with matplotlib.rc_context({"text.usetex": True}):
            fig = matplotlib.figure.Figure(figsize=(1, 1))
            ax  = fig.add_subplot(111)
            ax.set_xlabel("test")          # forces LaTeX to compile real text
            FigureCanvasAgg(fig).print_png(_io.BytesIO())
        return True
    except Exception as exc:
        logger.warning(
            "LaTeX binary found but render test failed (%s). "
            "Falling back to STIX Two. "
            "To enable CM Roman: fix your LaTeX install "
            "(e.g. run 'miktex update' or open MiKTeX Console).",
            exc,
        )
        return False


_HAS_LATEX: bool = _probe_latex()

_FONT_RC: dict = {
    "font.family":  "serif",
    "font.serif":   (
        ["Computer Modern Roman"]
        if _HAS_LATEX
        else ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"]
    ),
    "mathtext.fontset": "cm" if _HAS_LATEX else "stix",
}
if _HAS_LATEX:
    _FONT_RC["text.usetex"]         = True
    _FONT_RC["text.latex.preamble"] = r"\usepackage{amsmath}"

plt.rcParams.update({
    **_FONT_RC,
    # Font sizes — journal specification
    "font.size":             9,     # base / tick fallback
    "axes.titlesize":       10,     # panel titles
    "axes.labelsize":        9,     # axis labels
    "xtick.labelsize":       8,     # tick labels
    "ytick.labelsize":       8,
    "legend.fontsize":       8,     # legend entries
    "legend.title_fontsize": 8,
    # Aesthetics
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "0.8",
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.alpha":            0.3,
    "grid.color":            "0.75",
    "grid.linewidth":        0.5,
    "lines.linewidth":       1.5,
    "lines.markersize":      5,
    "patch.linewidth":       0.5,
})


def _tex(s: str) -> str:
    """Make a label string safe for the active rendering mode.

    When text.usetex=True (LaTeX / CM path): converts Unicode typographic
    symbols to their LaTeX equivalents so pdflatex does not choke.
    When STIX Two path (usetex=False): returns the string unchanged —
    Unicode renders natively through matplotlib's text engine.
    """
    if not _HAS_LATEX:
        return s
    s = s.replace("—", "---")       # em dash
    s = s.replace("–", "--")        # en dash
    s = s.replace("±", r"$\pm$")    # plus-minus sign
    s = s.replace("−", r"$-$")      # Unicode minus (U+2212)
    s = s.replace("%", r"\%")
    return s


def _clf_color(name: str) -> str:
    return CLF_PALETTE.get(name, "#888888")


def _save_fig(fig, path: Path, name: str):
    """Save figure as both PNG and PDF."""
    for ext in ["png", "pdf"]:
        fig.savefig(path / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved figure: %s", name)


# ---------------------------------------------------------------------------
# 1. Critical Difference diagram
# ---------------------------------------------------------------------------

def plot_cd_diagram(mean_std: pd.DataFrame, metric: str, fig_dir: Path):
    """Critical Difference diagram using Nemenyi post-hoc test."""
    subset = mean_std[mean_std["metric"] == metric]
    pivot = subset.pivot(index="dataset", columns="classifier", values="mean").dropna()

    ascending = metric in LOWER_IS_BETTER
    ranks = pivot.rank(axis=1, ascending=ascending, method="average")
    avg_ranks = ranks.mean().sort_values()

    n_clf = len(avg_ranks)
    n_datasets = len(pivot)

    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
               6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_alpha = q_table.get(n_clf, 3.0)
    cd = q_alpha * np.sqrt(n_clf * (n_clf + 1) / (6 * n_datasets))

    fig, ax = plt.subplots(figsize=(8, max(2.8, n_clf * 0.44)))

    names     = list(avg_ranks.index)
    rank_vals = list(avg_ranks.values)
    mid       = np.median(rank_vals)

    # Grey bars connecting non-significantly different classifiers
    for i in range(n_clf):
        for j in range(i + 1, n_clf):
            if abs(rank_vals[i] - rank_vals[j]) < cd:
                ax.plot(
                    [rank_vals[i], rank_vals[j]],
                    [n_clf - i - 0.12, n_clf - j + 0.12],
                    color="0.6", linewidth=2.2, alpha=0.45, solid_capstyle="round",
                )

    # Classifier dots and labels
    for i, (name, rank) in enumerate(zip(names, rank_vals)):
        y = n_clf - i
        ax.plot(rank, y, "o", color=_clf_color(name), markersize=8, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
        on_left = rank > mid
        ha      = "right" if on_left else "left"
        offset  = -0.20 if on_left else 0.20
        ax.text(rank + offset, y, f"{name}  ({rank:.2f})",
                ha=ha, va="center", fontsize=8)

    # CD bracket at top-left
    cd_y = n_clf + 0.9
    ax.annotate("", xy=(1 + cd, cd_y), xytext=(1.0, cd_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(1 + cd / 2, cd_y + 0.22,
            f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlim(0.3, n_clf + 0.7)
    ax.set_ylim(0, n_clf + 1.7)
    ax.set_xlabel("Average Rank  (lower = better)", labelpad=6)
    ax.set_title(_tex(f"Critical Difference \u2014 {metric.replace('_', ' ').title()}"))
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)

    _save_fig(fig, fig_dir, f"cd_diagram_{metric}")


# ---------------------------------------------------------------------------
# 2. Per-dataset accuracy bar chart
# ---------------------------------------------------------------------------

def plot_accuracy_bar_chart(
    mean_std: pd.DataFrame,
    fig_dir: Path,
    pwnb_clf: str = "PW-NB(auto)",
    baseline_clf: str = "GaussianNB",
):
    """Horizontal bar chart: proposed method vs GaussianNB per dataset (with error bars)."""
    subset     = mean_std[mean_std["metric"] == "accuracy"]
    pivot_mean = subset.pivot(index="dataset", columns="classifier", values="mean")
    pivot_std  = subset.pivot(index="dataset", columns="classifier", values="std")

    # Fall back to PW-NB(k=15) if adaptive variant is absent
    if pwnb_clf not in pivot_mean.columns:
        fallback = next((c for c in pivot_mean.columns if "PW-NB" in c), None)
        if fallback is None:
            logger.warning("No PW-NB classifier found — skipping bar chart.")
            return
        pwnb_clf = fallback
        logger.info("Bar chart: falling back to %s", pwnb_clf)

    if baseline_clf not in pivot_mean.columns:
        logger.warning("%s not found — skipping bar chart.", baseline_clf)
        return

    df = pd.DataFrame({
        "pwnb":     pivot_mean[pwnb_clf],
        "pwnb_err": pivot_std[pwnb_clf].fillna(0),
        "base":     pivot_mean[baseline_clf],
        "base_err": pivot_std[baseline_clf].fillna(0),
    }).dropna(subset=["pwnb", "base"])
    df["gain"] = df["pwnb"] - df["base"]
    df = df.sort_values("gain")

    n  = len(df)
    h  = 0.34
    y  = np.arange(n)
    _err_kw = dict(elinewidth=0.7, ecolor="0.25", capsize=2)

    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.33)))
    ax.barh(y - h / 2, df["base"], h, xerr=df["base_err"],
            color=BASE_COLOR, label=baseline_clf, error_kw=_err_kw)
    ax.barh(y + h / 2, df["pwnb"], h, xerr=df["pwnb_err"],
            color=PWNB_COLOR, label=pwnb_clf, error_kw=_err_kw)

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=7.5)
    ax.set_xlabel(_tex("Accuracy  (mean \u00b1 std, 10-fold CV)"))
    ax.set_title(_tex(f"{pwnb_clf} vs {baseline_clf} \u2014 Accuracy per Dataset"))
    ax.legend(loc="lower right")
    lo = max(0.0, df[["base", "pwnb"]].min().min() - 0.06)
    ax.set_xlim(left=lo)

    _save_fig(fig, fig_dir, "bar_accuracy_per_dataset")


# ---------------------------------------------------------------------------
# 3. PR score distribution
# ---------------------------------------------------------------------------

def plot_pr_distribution(dataset_name: str, X, y, fig_dir: Path, k: int = 15):
    """Histogram of PR scores coloured by class."""
    from src.proximal_ratio import ProximalRatio
    from sklearn.preprocessing import StandardScaler

    X_sc = StandardScaler().fit_transform(X)
    pr = ProximalRatio(k=k)
    pr.fit(X_sc, y)
    scores = pr.pr_scores_
    classes = np.unique(y)

    # Use tab10 for classes (works for up to 10 classes)
    cmap = plt.cm.get_cmap("tab10", max(len(classes), 3))

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for idx, c in enumerate(classes):
        ax.hist(scores[y == c], bins=20, range=(0, 1),
                alpha=0.65, color=cmap(idx),
                label=f"Class {c}", edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Proximal Ratio Score")
    ax.set_ylabel("Count")
    ax.set_title(_tex(f"PR Score Distribution \u2014 {dataset_name}  (k = {k})"))
    ax.legend(title="Class", framealpha=0.9, fontsize=7.5)

    _save_fig(fig, fig_dir, f"pr_distribution_{dataset_name}")


# ---------------------------------------------------------------------------
# 4. ECE comparison (replaces simplified reliability bar)
# ---------------------------------------------------------------------------

def plot_ece_comparison(dataset_name: str, mean_std: pd.DataFrame, fig_dir: Path):
    """Horizontal bar chart of ECE for every classifier on one dataset."""
    subset = mean_std[
        (mean_std["dataset"] == dataset_name) & (mean_std["metric"] == "ece")
    ].copy()
    if subset.empty:
        logger.warning("No ECE data for %s — skipping.", dataset_name)
        return

    subset = subset.sort_values("mean", ascending=False)   # worst → top
    colors = [_clf_color(c) for c in subset["classifier"]]

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(subset) * 0.34)))
    y = np.arange(len(subset))
    ax.barh(y, subset["mean"].values,
            xerr=subset["std"].fillna(0).values,
            color=colors, capsize=2,
            error_kw=dict(elinewidth=0.7, ecolor="0.25"))
    ax.set_yticks(y)
    ax.set_yticklabels(subset["classifier"].values, fontsize=8)
    ax.set_xlabel("ECE  (lower = better)")
    ax.set_title(_tex(f"Expected Calibration Error \u2014 {dataset_name}"))

    _save_fig(fig, fig_dir, f"ece_comparison_{dataset_name}")


# ---------------------------------------------------------------------------
# 5. k-sensitivity (fixed-k variants + PW-NB(auto) reference)
# ---------------------------------------------------------------------------

def plot_k_sensitivity(mean_std: pd.DataFrame, fig_dir: Path):
    """Accuracy and macro_f1 vs k, with PW-NB(auto) and GaussianNB reference lines."""
    representative = ["iris", "wine", "breast_cancer", "glass", "ionosphere", "sonar"]
    k_values = [5, 15, 30, 45]

    has_auto = "PW-NB(auto)" in mean_std["classifier"].unique()

    for metric in ["accuracy", "macro_f1"]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.flatten()

        for idx, ds_name in enumerate(representative):
            ax = axes[idx]

            means, stds = [], []
            for k in k_values:
                row = mean_std[
                    (mean_std["dataset"] == ds_name)
                    & (mean_std["classifier"] == f"PW-NB(k={k})")
                    & (mean_std["metric"] == metric)
                ]
                if not row.empty:
                    means.append(float(row["mean"].values[0]))
                    stds.append(float(row["std"].values[0]))
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            means = np.array(means, dtype=float)
            stds  = np.array(stds,  dtype=float)

            ax.fill_between(k_values, means - stds, means + stds,
                            color=PWNB_COLOR, alpha=0.15)
            ax.plot(k_values, means, "o-", color=PWNB_COLOR,
                    linewidth=1.6, markersize=5, label="Fixed-k PW-NB",
                    markeredgecolor="white", markeredgewidth=0.4)

            # PW-NB(auto) reference band
            if has_auto:
                ar = mean_std[
                    (mean_std["dataset"] == ds_name)
                    & (mean_std["classifier"] == "PW-NB(auto)")
                    & (mean_std["metric"] == metric)
                ]
                if not ar.empty:
                    am, asd = float(ar["mean"].values[0]), float(ar["std"].values[0])
                    ax.axhline(am, color=CLF_PALETTE["PW-NB(auto)"],
                               linestyle="--", linewidth=1.2, label="PW-NB(auto)")
                    ax.axhspan(am - asd, am + asd,
                               color=CLF_PALETTE["PW-NB(auto)"], alpha=0.10)

            # GaussianNB reference line
            gr = mean_std[
                (mean_std["dataset"] == ds_name)
                & (mean_std["classifier"] == "GaussianNB")
                & (mean_std["metric"] == metric)
            ]
            if not gr.empty:
                ax.axhline(float(gr["mean"].values[0]), color=BASE_COLOR,
                           linestyle=":", linewidth=1.0, label="GaussianNB")

            ax.set_title(ds_name, fontsize=9)
            ax.set_xlabel("k")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_xticks(k_values)
            if idx == 0:
                ax.legend(fontsize=7, loc="lower right")

        fig.suptitle(
            f"k-Sensitivity: {metric.replace('_', ' ').title()}", fontsize=11
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save_fig(fig, fig_dir, f"k_sensitivity_{metric}")


# ---------------------------------------------------------------------------
# 6. PR score vs accuracy gain scatter  [NEEDS TRAINING DATA]
# ---------------------------------------------------------------------------

def plot_pr_gain_scatter(mean_std: pd.DataFrame, fig_dir: Path):
    """Scatter: mean training PR score (x) vs accuracy gain over GaussianNB (y).

    Each point is one dataset.  Orange = PW-NB wins, blue = GaussianNB wins.
    This figure REQUIRES mean_pr to have been logged during training
    (captured in run_experiment.py from clf.pr_scores_).
    """
    # Prefer adaptive; fall back to best available fixed-k
    pwnb_clf = None
    for candidate in ["PW-NB(auto)", "PW-NB(k=15)", "PW-NB(k=30)",
                      "PW-NB(k=45)", "PW-NB(k=5)"]:
        if candidate in mean_std["classifier"].unique():
            pwnb_clf = candidate
            break
    if pwnb_clf is None:
        logger.warning("No PW-NB classifier found — skipping PR gain scatter.")
        return

    pr_rows = mean_std[
        (mean_std["classifier"] == pwnb_clf) & (mean_std["metric"] == "mean_pr")
    ]
    if pr_rows.empty:
        logger.warning(
            "mean_pr not found in mean_std.csv — was it logged during training? "
            "Skipping PR gain scatter."
        )
        return

    acc_pwnb = (
        mean_std[(mean_std["classifier"] == pwnb_clf) & (mean_std["metric"] == "accuracy")]
        .set_index("dataset")["mean"]
    )
    acc_gnb = (
        mean_std[(mean_std["classifier"] == "GaussianNB") & (mean_std["metric"] == "accuracy")]
        .set_index("dataset")["mean"]
    )
    pr_mean = pr_rows.set_index("dataset")["mean"]

    datasets = pr_mean.index.intersection(acc_pwnb.index).intersection(acc_gnb.index)
    if len(datasets) == 0:
        logger.warning("No overlapping datasets for PR gain scatter.")
        return

    x      = pr_mean.loc[datasets].values.astype(float)
    y_gain = (acc_pwnb.loc[datasets] - acc_gnb.loc[datasets]).values.astype(float)
    labels = list(datasets)
    colors = np.where(y_gain > 0, PWNB_COLOR, BASE_COLOR)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y_gain, c=colors, s=60, alpha=0.88,
               edgecolors="white", linewidths=0.5, zorder=3)

    for ds, xi, yi in zip(labels, x, y_gain):
        ax.annotate(ds, (xi, yi), fontsize=6.5, alpha=0.75,
                    textcoords="offset points", xytext=(5, 3))

    # Trend line
    if len(x) > 3:
        mask = ~(np.isnan(x) | np.isnan(y_gain))
        if mask.sum() > 3:
            coef  = np.polyfit(x[mask], y_gain[mask], 1)
            xline = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(xline, np.polyval(coef, xline),
                    color="0.4", linestyle="--", linewidth=1.0, alpha=0.7,
                    label="Trend")

    ax.axhline(0, color="0.55", linestyle="-", linewidth=0.9)
    ax.set_xlabel("Mean PR Score  (training fold average)")
    ax.set_ylabel(_tex(f"Accuracy Gain  ({pwnb_clf} \u2212 GaussianNB)"))
    ax.set_title("Does PW-NB Help More When Data Is Noisier?\n"
                 "(Lower PR ≈ more class overlap in training fold)")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PWNB_COLOR,
               markersize=7, label=f"{pwnb_clf} wins"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BASE_COLOR,
               markersize=7, label="GaussianNB wins"),
    ]
    ax.legend(handles=handles, fontsize=8)

    _save_fig(fig, fig_dir, "pr_gain_scatter")


# ---------------------------------------------------------------------------
# 7. Best-k distribution for PW-NB(auto)  [NEEDS TRAINING DATA]
# ---------------------------------------------------------------------------

def plot_best_k_distribution(
    mean_std: pd.DataFrame,
    all_folds: pd.DataFrame | None,
    fig_dir: Path,
):
    """Bar chart: how often each k is selected by PW-NB(auto) across all folds.

    Uses per-fold data from all_folds.csv when available (exact integer counts);
    falls back to the rounded mean from mean_std.csv.
    """
    k_series = None

    if all_folds is not None:
        src = all_folds[
            (all_folds["classifier"] == "PW-NB(auto)")
            & (all_folds["metric"] == "best_k")
        ]
        if not src.empty:
            k_series = src["value"].round().astype(int)

    if k_series is None:
        src = mean_std[
            (mean_std["classifier"] == "PW-NB(auto)")
            & (mean_std["metric"] == "best_k")
        ]
        if src.empty:
            logger.warning(
                "best_k not found — was it logged during training? "
                "Skipping k-distribution plot."
            )
            return
        k_series = src["mean"].round().astype(int)

    k_counts = k_series.value_counts().sort_index()
    total    = k_counts.sum()

    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(
        k_counts.index.astype(str), k_counts.values,
        color=PWNB_COLOR, edgecolor="white", linewidth=0.5,
    )
    for bar, cnt in zip(bars, k_counts.values):
        pct = 100 * cnt / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{cnt}  ({pct:.0f}%)",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Selected k")
    ax.set_ylabel("Selection count  (datasets × folds)")
    ax.set_title("PW-NB(auto): Inner-CV k Selection Distribution")
    ax.set_ylim(top=k_counts.max() * 1.18)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    _save_fig(fig, fig_dir, "best_k_distribution")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: Path, cache_dir: Path | None = None):
    """Generate all publication figures."""
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    mean_std = pd.read_csv(results_dir / "summary" / "mean_std.csv")
    all_folds_path = results_dir / "raw" / "all_folds.csv"
    all_folds = pd.read_csv(all_folds_path) if all_folds_path.exists() else None

    # 1. CD diagrams (all metrics)
    for metric in [
        "accuracy", "macro_f1", "auc_roc", "ece", "brier_score", "log_loss",
        "balanced_accuracy", "geometric_mean", "mcc", "weighted_f1",
    ]:
        try:
            plot_cd_diagram(mean_std, metric, fig_dir)
        except Exception as e:
            logger.error("CD diagram failed for %s: %s", metric, e)

    # 2. Accuracy bar chart (proposed vs GaussianNB)
    try:
        plot_accuracy_bar_chart(mean_std, fig_dir, pwnb_clf="PW-NB(auto)")
    except Exception as e:
        logger.error("Bar chart failed: %s", e)

    # 3. PR score distributions (uses raw dataset files — reproducible post-hoc)
    from src.datasets import load_dataset
    for ds_name in ["iris", "breast_cancer", "glass", "ionosphere", "sonar", "yeast"]:
        try:
            X, y, _ = load_dataset(ds_name, cache_dir)
            plot_pr_distribution(ds_name, X, y, fig_dir)
        except Exception as e:
            logger.error("PR distribution failed for %s: %s", ds_name, e)

    # 4. ECE comparison per representative dataset
    for ds_name in ["iris", "breast_cancer", "page_blocks", "letter"]:
        try:
            plot_ece_comparison(ds_name, mean_std, fig_dir)
        except Exception as e:
            logger.error("ECE comparison failed for %s: %s", ds_name, e)

    # 5. k-sensitivity with PW-NB(auto) reference
    try:
        plot_k_sensitivity(mean_std, fig_dir)
    except Exception as e:
        logger.error("k-sensitivity failed: %s", e)

    # 6. PR score vs accuracy gain scatter  [requires mean_pr from training]
    try:
        plot_pr_gain_scatter(mean_std, fig_dir)
    except Exception as e:
        logger.error("PR gain scatter failed: %s", e)

    # 7. Best-k distribution for PW-NB(auto)  [requires best_k from training]
    try:
        plot_best_k_distribution(mean_std, all_folds, fig_dir)
    except Exception as e:
        logger.error("Best-k distribution failed: %s", e)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "cache"),
    )
    args = parser.parse_args()
    generate_all_figures(Path(args.results_dir), Path(args.cache_dir))


if __name__ == "__main__":
    main()
