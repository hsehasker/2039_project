"""High-level Machine Unlearning driver.

Two request modes are supported (mandated by §6.2.2 of the TЗ):

  * **delete-by-ID**       — drop every row whose ``user_id`` is in a list
  * **delete-by-filter**   — drop every row matching a boolean pandas query

The procedure is:

  1. resolve the requested rows into row-indices of the training DataFrame;
  2. identify which SISA shards contain those rows (``rebuild_after_deletion``);
  3. drop the rows from the DataFrame and the feature matrix;
  4. re-train only the affected shards;
  5. re-evaluate on a held-out test set and return a delta-report.

The ensemble retains exactly the same shard identifiers, so saved metric
histories can be compared point-by-point before vs. after unlearning.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import ClassificationReport, compute_classification_metrics
from .sisa import SISAEnsemble, SISAReport


@dataclass
class UnlearnRequest:
    user_ids: list[str] | None = None         # delete-by-ID
    filter_query: str | None = None           # delete-by-filter (pandas .query)
    reason: str = "user_request"              # e.g. "152-ФЗ ст.14"

    def is_empty(self) -> bool:
        return not self.user_ids and not self.filter_query


@dataclass
class UnlearnResult:
    deleted_rows: int
    affected_shards: list[int]
    retrain_report: SISAReport
    metrics_before: ClassificationReport
    metrics_after: ClassificationReport
    wall_time_sec: float
    # Δ = after - before (negative means degradation for accuracy-like metrics)
    delta: dict[str, float] = field(default_factory=dict)
    # Proportion of test predictions that flipped (label stability proxy).
    prediction_disagreement: float = 0.0


class Unlearner:
    """Drives delete -> retrain-affected -> re-evaluate cycle."""

    def __init__(self, ensemble: SISAEnsemble) -> None:
        self.ensemble = ensemble

    # ---- request resolution -------------------------------------------------

    def resolve_rows(self, df: pd.DataFrame, request: UnlearnRequest) -> np.ndarray:
        if request.is_empty():
            return np.empty(0, dtype=np.int64)
        masks: list[np.ndarray] = []
        if request.user_ids:
            if "user_id" not in df.columns:
                raise KeyError("Cannot delete by ID: 'user_id' column missing")
            masks.append(df["user_id"].isin(request.user_ids).values)
        if request.filter_query:
            masks.append(df.eval(request.filter_query).astype(bool).values)
        combined = np.zeros(len(df), dtype=bool)
        for m in masks:
            combined |= m
        return np.where(combined)[0].astype(np.int64)

    # ---- main entry ---------------------------------------------------------

    def unlearn(
        self,
        request: UnlearnRequest,
        df_train: pd.DataFrame,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[UnlearnResult, pd.DataFrame, np.ndarray, np.ndarray]:
        """Execute one unlearning round and return (result, new df/X/y)."""
        start = time.time()

        # 1. Baseline metrics before any change
        preds_before = self.ensemble.predict_proba(X_test)
        metrics_before = compute_classification_metrics(y_test, preds_before)

        # 2. Resolve + delete rows
        deleted_rows = self.resolve_rows(df_train, request)
        if len(deleted_rows) == 0:
            # Nothing to do; still report a no-op
            metrics_after = metrics_before
            return (
                UnlearnResult(
                    deleted_rows=0,
                    affected_shards=[],
                    retrain_report=SISAReport(
                        n_shards=self.ensemble.plan.n_shards,
                        shard_sizes=self.ensemble.plan.sizes(),
                        per_shard_time_sec=[0.0] * self.ensemble.plan.n_shards,
                        per_shard_loss=[None] * self.ensemble.plan.n_shards,
                        total_wall_time_sec=0.0,
                    ),
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                    wall_time_sec=time.time() - start,
                ),
                df_train, X_train, y_train,
            )

        # 3. Update shard plan and row-indexed arrays
        new_plan, affected = self.ensemble.plan.rebuild_after_deletion(deleted_rows)
        keep_mask = ~np.isin(np.arange(len(df_train)), deleted_rows)
        df_train_new = df_train.iloc[keep_mask].reset_index(drop=True)
        X_train_new = X_train[keep_mask]
        y_train_new = y_train[keep_mask]

        # 4. Swap plan and retrain only affected shards
        self.ensemble.replace_plan(new_plan)
        retrain_report = self.ensemble.retrain_shards(sorted(affected), X_train_new, y_train_new)

        # 5. Evaluate after
        preds_after = self.ensemble.predict_proba(X_test)
        metrics_after = compute_classification_metrics(y_test, preds_after)

        hard_before = (preds_before >= 0.5).astype(int)
        hard_after = (preds_after >= 0.5).astype(int)
        disagreement = float((hard_before != hard_after).mean())

        delta = {
            "accuracy":  metrics_after.accuracy  - metrics_before.accuracy,
            "precision": metrics_after.precision - metrics_before.precision,
            "recall":    metrics_after.recall    - metrics_before.recall,
            "f1":        metrics_after.f1        - metrics_before.f1,
            "auc":       metrics_after.auc       - metrics_before.auc,
            "mse":       metrics_after.mse       - metrics_before.mse,
        }

        return (
            UnlearnResult(
                deleted_rows=int(len(deleted_rows)),
                affected_shards=sorted(int(s) for s in affected),
                retrain_report=retrain_report,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                wall_time_sec=time.time() - start,
                delta=delta,
                prediction_disagreement=disagreement,
            ),
            df_train_new, X_train_new, y_train_new,
        )
