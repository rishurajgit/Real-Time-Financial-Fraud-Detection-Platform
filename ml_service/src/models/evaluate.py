"""Imbalanced binary fraud model evaluation and threshold optimization.

Uses Precision-Recall AUC (Average Precision) as the primary metric rather than
accuracy, matching severe class imbalance in real-world payment fraud (< 4% / < 1%).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    block_threshold: float
    review_threshold: float
    best_f1: float
    precision_at_block: float
    recall_at_block: float


def compute_confusion_counts(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
) -> dict[str, int]:
    """Pure-Python confusion matrix counts at a given probability threshold."""
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have identical length.")

    tn = fp = fn = tp = 0
    for truth, prob in zip(y_true, y_prob):
        pred = 1 if prob >= threshold else 0
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def optimize_thresholds_from_pr(
    precisions: Sequence[float],
    recalls: Sequence[float],
    thresholds: Sequence[float],
    min_precision_for_block: float = 0.50,
) -> ThresholdOptimizationResult:
    """Select block_threshold (maximizing F1, preferring precision >= floor) and review_threshold."""
    if len(thresholds) == 0:
        return ThresholdOptimizationResult(
            block_threshold=0.80,
            review_threshold=0.50,
            best_f1=0.0,
            precision_at_block=0.0,
            recall_at_block=0.0,
        )

    best_idx = 0
    best_f1 = -1.0
    constrained_idx: int | None = None
    constrained_f1 = -1.0

    for i, thr in enumerate(thresholds):
        p = float(precisions[i])
        r = float(recalls[i])
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0.0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_idx = i
        if p >= min_precision_for_block and f1 > constrained_f1:
            constrained_f1 = f1
            constrained_idx = i

    chosen = constrained_idx if constrained_idx is not None else best_idx
    block_thr = float(thresholds[chosen])
    chosen_f1 = constrained_f1 if constrained_idx is not None else best_f1

    # Choose review threshold at a higher-recall operating point below block_threshold
    review_thr = max(0.20, round(block_thr * 0.68, 4))
    if review_thr >= block_thr:
        review_thr = max(0.10, round(block_thr - 0.15, 4))

    return ThresholdOptimizationResult(
        block_threshold=round(block_thr, 4),
        review_threshold=round(review_thr, 4),
        best_f1=round(chosen_f1, 4),
        precision_at_block=round(float(precisions[chosen]), 4),
        recall_at_block=round(float(recalls[chosen]), 4),
    )


def evaluate_binary_predictions(
    y_true: Any,
    y_prob: Any,
    chart_output_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Compute full imbalanced classification report, optimal thresholds, and PR curve points."""
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
    opt = optimize_thresholds_from_pr(precisions, recalls, thresholds)

    y_pred = (y_prob_arr >= opt.block_threshold).astype(np.int32)
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    cm = compute_confusion_counts(y_true_arr.tolist(), y_prob_arr.tolist(), opt.block_threshold)

    # Precision at 80% and 90% recall operating points
    def _precision_at_recall(target_recall: float) -> float:
        valid = [float(p) for p, r in zip(precisions, recalls) if r >= target_recall]
        return round(max(valid), 4) if valid else 0.0

    # Downsample PR curve to ~40 sorted points by recall for lightweight JSON charting
    step = max(1, len(recalls) // 40)
    pr_points = [
        {
            "recall": round(float(recalls[i]), 4),
            "precision": round(float(precisions[i]), 4),
            "threshold": round(float(thresholds[min(i, len(thresholds) - 1)]), 4)
            if len(thresholds) > 0
            else 0.5,
        }
        for i in range(0, len(recalls), step)
    ]
    pr_points.sort(key=lambda pt: pt["recall"])

    if chart_output_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        ax1.plot(recalls, precisions, color="#1D4ED8", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
        baseline = float(y_true_arr.mean())
        ax1.axhline(baseline, color="#94A3B8", linestyle="--", label=f"Baseline = {baseline:.4f}")
        ax1.set_title("Validation Precision-Recall Curve")
        ax1.set_xlabel("Recall")
        ax1.set_ylabel("Precision")
        ax1.legend(loc="lower left")
        ax1.grid(alpha=0.3)

        ax2.plot(fpr, tpr, color="#0F172A", linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
        ax2.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--")
        ax2.set_title("Validation ROC Curve")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        chart_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chart_output_path, dpi=150)
        plt.close(fig)

    metrics: dict[str, Any] = {
        "primary_metric": "pr_auc",
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1_score": round(f1, 5),
        "precision_at_80_recall": _precision_at_recall(0.80),
        "precision_at_90_recall": _precision_at_recall(0.90),
        "optimized_block_threshold": opt.block_threshold,
        "optimized_review_threshold": opt.review_threshold,
        "confusion_matrix": cm,
        "validation_positive_rate_pct": round(float(y_true_arr.mean() * 100.0), 4),
    }
    return metrics, pr_points
