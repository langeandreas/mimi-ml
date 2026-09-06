"""Matplotlib visualizations for behavioral signature analysis.

Provides a single public function, ``plot_behavioral_signatures``, that takes
the trajectory metrics dict produced by ``TreeAnalyzer.compute_trajectory_metrics``
and renders a 2×2 diagnostic figure.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def plot_behavioral_signatures(
    trajectory_metrics: Dict,
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[str] = None,
    selected_features: Optional[List[int]] = None,
    points_per_feature: Optional[int] = None,
    confusion_metric: str = "hessian",
):
    """Plot Feature Acceleration vs. confusion-signal behavioral signatures.

    Produces a 2×2 figure:
    - Top-left:  Acceleration vs. FCI scatter, color-coded by boosting iteration,
                 with quadrant regions interpreted jointly with color
                 (early vs. late phase).
    - Top-right: Feature Acceleration trajectory over iterations.
    - Bottom-left:  FCI trajectory over iterations.
    - Bottom-right: Cumulative optimization gain over training.

    Args:
        trajectory_metrics: Output of ``TreeAnalyzer.compute_trajectory_metrics()``.
        figsize: Matplotlib figure size ``(width, height)`` in inches.
        save_path: File path to save the figure.  When *None* the figure is
                   displayed but not saved.
        selected_features: Feature indices to include.  When *None* all features
                           are plotted.
        points_per_feature: Number of phase-contracted points to display per
                            feature across training. When *None* all iterations
                            are shown.

    Returns:
        ``(fig, axes)`` — the matplotlib Figure and 2×2 Axes array.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")
    confusion_metric = _normalize_confusion_metric(confusion_metric)
    confusion_label = _confusion_label(confusion_metric)

    metrics_to_plot = (
        {k: v for k, v in trajectory_metrics.items() if k in selected_features}
        if selected_features is not None
        else trajectory_metrics
    )

    # ── Top-left: Acceleration vs. FCI scatter ──────────────────────────────
    ax1 = axes[0, 0]
    all_fcis, all_accelerations, all_iterations = [], [], []

    for _, metrics in metrics_to_plot.items():
        accelerations = metrics["acceleration"]
        fcis = _confusion_series(metrics, confusion_metric)
        iterations = metrics["iterations"]
        feature_name = metrics["feature_name"]

        sample_idxs = _phase_sample_indices(len(iterations), points_per_feature)
        sampled_accelerations = [accelerations[i] for i in sample_idxs]
        sampled_fcis = [fcis[i] for i in sample_idxs]
        sampled_iterations = [iterations[i] for i in sample_idxs]

        all_fcis.extend(sampled_fcis)
        all_accelerations.extend(sampled_accelerations)
        all_iterations.extend(sampled_iterations)

        ax1.scatter(
            sampled_fcis, sampled_accelerations,
            c=sampled_iterations, s=150, alpha=0.7,
            label=feature_name, cmap="cool",
            edgecolors="black", lw=0.5,
        )
        ax1.plot(sampled_fcis, sampled_accelerations, alpha=0.3, linestyle="--", linewidth=1)

        if sampled_fcis:
            ax1.annotate(
                feature_name,
                xy=(sampled_fcis[-1], sampled_accelerations[-1]),
                xytext=(5, 5), textcoords="offset points", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=0.5),
            )

    if all_fcis and all_accelerations:
        max_fci = max(abs(min(all_fcis)), abs(max(all_fcis)))
        max_accel = max(abs(min(all_accelerations)), abs(max(all_accelerations)))
        padding = 0.1
        fci_limit = max_fci * (1 + padding)
        accel_limit = max_accel * (1 + padding)
        ax1.set_xlim(0, fci_limit)
        ax1.set_ylim(-accel_limit, accel_limit)

        q = accel_limit * 0.7
        _quadrant_label(
            ax1,
            0.9 * max_fci,
            q,
            "CONFLICT RESOLVER\n(High FCI, +Acceleration)\nLight=early emergence, dark=late escalation",
            "lightblue",
        )
        _quadrant_label(
            ax1,
            0.1 * max_fci,
            q,
            "HIGH-VARIANCE PATCH\n(Low FCI, +Acceleration)\nLight=early micro-patching, dark=late cleanup",
            "lightyellow",
            ha="left",
        )
        _quadrant_label(
            ax1,
            0.9 * max_fci,
            -q,
            "EASY\n(High FCI, -Acceleration)\nLight=early coarse-fit decay, dark=late stabilization",
            "lightgreen",
        )
        _quadrant_label(
            ax1,
            0.1 * max_fci,
            -q,
            "OUTLIER SPECIALIST\n(Low FCI, -Acceleration)\nLight=early pruning, dark=late tail calibration",
            "lightcoral",
            ha="left",
        )

    ax1.set_xlabel(confusion_label, fontsize=12, weight="bold")
    ax1.set_ylabel("Feature Acceleration (A_f)", fontsize=12, weight="bold")
    ax1.set_title("Behavioral Signatures: Acceleration vs Confusion", fontsize=13, weight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5, linewidth=1)

    if all_iterations:
        sm = plt.cm.ScalarMappable(
            cmap="cool",
            norm=mcolors.Normalize(vmin=min(all_iterations), vmax=max(all_iterations)),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, pad=0.02)
        cbar.set_label("Boosting Iteration (Early→Late)", fontsize=10, weight="bold")

    # ── Top-right: Acceleration over iterations ──────────────────────────────
    ax2 = axes[0, 1]
    for _, metrics in metrics_to_plot.items():
        sample_idxs = _phase_sample_indices(len(metrics["iterations"]), points_per_feature)
        sampled_iterations = [metrics["iterations"][i] for i in sample_idxs]
        sampled_acceleration = [metrics["acceleration"][i] for i in sample_idxs]
        ax2.plot(
            sampled_iterations, sampled_acceleration,
            marker="o", label=metrics["feature_name"], linewidth=2, markersize=6,
        )
    ax2.set_xlabel("Boosting Iteration", fontsize=12, weight="bold")
    ax2.set_ylabel("Feature Acceleration (A_f)", fontsize=12, weight="bold")
    ax2.set_title("Feature Acceleration Over Time", fontsize=13, weight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.95, ncol=1)

    # ── Bottom-left: confusion signal over iterations ────────────────────────
    ax3 = axes[1, 0]
    for _, metrics in metrics_to_plot.items():
        sample_idxs = _phase_sample_indices(len(metrics["iterations"]), points_per_feature)
        sampled_iterations = [metrics["iterations"][i] for i in sample_idxs]
        conf = _confusion_series(metrics, confusion_metric)
        sampled_fci = [conf[i] for i in sample_idxs]
        ax3.plot(
            sampled_iterations, sampled_fci,
            marker="s", label=metrics["feature_name"], linewidth=2, markersize=6,
        )
    ax3.set_xlabel("Boosting Iteration", fontsize=12, weight="bold")
    ax3.set_ylabel(confusion_label, fontsize=12, weight="bold")
    ax3.set_title(f"{confusion_label} Over Time", fontsize=13, weight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.95, ncol=1)

    # ── Bottom-right: Cumulative gain over iterations ─────────────────────────
    ax4 = axes[1, 1]
    for _, metrics in metrics_to_plot.items():
        sample_idxs = _phase_sample_indices(len(metrics["iterations"]), points_per_feature)
        sampled_iterations = [metrics["iterations"][i] for i in sample_idxs]
        sampled_cumulative_gain = [metrics["cumulative_gain"][i] for i in sample_idxs]
        ax4.plot(
            sampled_iterations, sampled_cumulative_gain,
            marker="^", label=metrics["feature_name"], linewidth=2.5, markersize=6,
        )
    ax4.set_xlabel("Boosting Iteration", fontsize=12, weight="bold")
    ax4.set_ylabel("Cumulative Gain (Γ)", fontsize=12, weight="bold")
    ax4.set_title("Cumulative Optimization Gain Over Training", fontsize=13, weight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left", fontsize=8, framealpha=0.95, ncol=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig, axes


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _quadrant_label(ax, x: float, y: float, text: str, color: str, ha: str = "center") -> None:
    ax.text(
        x, y, text,
        ha=ha, va="center", fontsize=10, weight="bold",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=color, alpha=0.3),
    )


def _phase_sample_indices(series_length: int, points_per_feature: Optional[int]) -> List[int]:
    """Return phase-balanced indices for a 1D trajectory."""
    if series_length <= 0:
        return []
    if points_per_feature is None or points_per_feature >= series_length:
        return list(range(series_length))
    if points_per_feature <= 1:
        return [series_length // 2]

    bins = min(points_per_feature, series_length)
    sampled: List[int] = []
    for i in range(bins):
        start = (i * series_length) // bins
        end = ((i + 1) * series_length) // bins - 1
        if end < start:
            end = start
        sampled.append((start + end) // 2)

    # Keep stable order and guard against accidental duplicates.
    return sorted(set(sampled))


def _normalize_confusion_metric(value: str) -> str:
    metric = str(value or "hessian").strip().lower()
    if metric not in {"hessian", "fci"}:
        raise ValueError("confusion_metric must be 'hessian' or 'fci'")
    return metric


def _confusion_label(confusion_metric: str) -> str:
    return "Raw Hessian Mean (HVI signal)" if confusion_metric == "hessian" else "Feature Confusion Index (FCI)"


def _confusion_series(metrics: Dict, confusion_metric: str) -> List[float]:
    if confusion_metric == "hessian" and metrics.get("hessian_mean_raw"):
        return list(metrics.get("hessian_mean_raw", []))
    return list(metrics.get("fci", []))
