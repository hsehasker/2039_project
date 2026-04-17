"""Evaluation metrics for the Digital Forgetting prototype.

Covers the four groups required by §6.2 of the ТЗ:

  1. **Classification quality**: Accuracy, Precision, Recall, F1, ROC AUC.
  2. **Regression-style error**: MSE on the probability outputs (useful as a
     continuous proxy for "how much did the decision surface move?").
  3. **Resource usage**: wall time, peak memory (RSS) — reported by the CLI
     around each run, not by the pure-prediction metrics here.
  4. **Unlearning-specific diagnostics**: delta-metrics and prediction
     disagreement are assembled in ``unlearning.Unlearner`` from these
     primitive reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    mse: float
    positive_rate_true: float
    positive_rate_pred: float
    n_samples: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationReport:
    """Compute standard binary classification metrics.

    Implemented without scikit-learn dependency for the core math so the
    tests don't break on a partial install; ROC AUC uses the Mann-Whitney
    U equivalence for a clean numpy-only implementation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = tp + tn + fp + fn

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    auc = _roc_auc(y_true, y_score)
    mse = float(np.mean((y_score - y_true) ** 2))

    return ClassificationReport(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auc=float(auc),
        mse=mse,
        positive_rate_true=float(y_true.mean() if n else 0.0),
        positive_rate_pred=float(y_pred.mean() if n else 0.0),
        n_samples=n,
    )


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Numpy-only ROC AUC via the Mann-Whitney U identity.

    AUC = P(score(pos) > score(neg)) + 0.5 * P(score(pos) == score(neg)).
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5  # undefined, return chance level
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    scores = np.concatenate([pos, neg])
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    rank_sum = np.zeros(len(counts))
    np.add.at(rank_sum, inv, ranks)
    avg_ranks = (rank_sum / counts)[inv]
    rank_pos = avg_ranks[: len(pos)]
    u = rank_pos.sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


# ---- delta helpers ---------------------------------------------------------

def delta_report(
    before: ClassificationReport,
    after: ClassificationReport,
) -> dict[str, float]:
    """Return a dict of ``metric -> (after - before)`` for the scalar metrics."""
    keys = ("accuracy", "precision", "recall", "f1", "auc", "mse")
    return {k: float(getattr(after, k) - getattr(before, k)) for k in keys}
