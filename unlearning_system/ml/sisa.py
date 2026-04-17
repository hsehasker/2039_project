"""SISA ensemble: Sharded, Isolated, Sliced, Aggregated learners.

Reference
---------
Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C.A., Jia, H., Travers,
A., Zhang, B., Lie, D., Papernot, N. (2021).
*Machine Unlearning.* IEEE Symposium on Security and Privacy.

Design choices in this implementation
-------------------------------------
* **Sharding**: disjoint, hash-based on ``user_id``. Route-to-shard for a
  single deleted user is O(1).
* **Slicing**: we do *not* implement intra-shard slices (the "S" in SISA
  improves forget cost further by checkpointing, but multiplies disk
  usage). The architecture is easy to extend if needed.
* **Aggregation**: soft voting — mean of ``P(y=1|x)`` across shard models.
* **Heterogeneous models**: every shard uses the same learner *class*
  (LogReg or RF) with the same config; mixing types per shard is
  out-of-scope for the prototype.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .base import BaseLearner
from .shards import ShardPlan


LearnerFactory = Callable[[int], BaseLearner]  # (shard_id) -> BaseLearner


@dataclass
class SISAReport:
    """Aggregated stats for a full training or retraining pass."""
    n_shards: int
    shard_sizes: list[int]
    per_shard_time_sec: list[float]
    per_shard_loss: list[float | None]
    total_wall_time_sec: float
    affected_shards: list[int] = field(default_factory=list)


class SISAEnsemble:
    """Ensemble container with ``fit`` / ``predict`` / ``retrain_shards``."""

    def __init__(self, factory: LearnerFactory, plan: ShardPlan) -> None:
        self.factory = factory
        self.plan = plan
        self.learners: list[BaseLearner | None] = [None] * plan.n_shards
        self.last_report: SISAReport | None = None

    # ---- training -----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SISAEnsemble":
        """Train every shard from scratch."""
        self.last_report = self._train_shards(range(self.plan.n_shards), X, y)
        return self

    def retrain_shards(self, shard_ids, X: np.ndarray, y: np.ndarray) -> SISAReport:
        """Re-train only the specified shards — used after unlearning."""
        report = self._train_shards(shard_ids, X, y)
        self.last_report = report
        return report

    def _train_shards(self, shard_ids, X: np.ndarray, y: np.ndarray) -> SISAReport:
        shard_ids = list(shard_ids)
        per_shard_time = [0.0] * self.plan.n_shards
        per_shard_loss: list[float | None] = [None] * self.plan.n_shards
        start_total = time.time()
        for s in shard_ids:
            idx = self.plan.indices[s]
            if len(idx) == 0:
                # Empty shard (e.g. after aggressive deletion) -> clear learner
                self.learners[s] = None
                continue
            learner = self.factory(s)
            learner.fit(X[idx], y[idx])
            self.learners[s] = learner
            st = getattr(learner, "stats_", None)
            if st is not None:
                per_shard_time[s] = st.wall_time_sec
                per_shard_loss[s] = st.final_loss
        return SISAReport(
            n_shards=self.plan.n_shards,
            shard_sizes=self.plan.sizes(),
            per_shard_time_sec=per_shard_time,
            per_shard_loss=per_shard_loss,
            total_wall_time_sec=time.time() - start_total,
            affected_shards=list(shard_ids),
        )

    # ---- inference ----------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft-vote: mean P(y=1) across non-empty shards."""
        active: list[np.ndarray] = []
        for learner in self.learners:
            if learner is None:
                continue
            active.append(learner.predict_proba(X))
        if not active:
            # no trained shards — return prior probability 0.5
            return np.full(shape=(len(X),), fill_value=0.5)
        return np.mean(np.stack(active, axis=0), axis=0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ---- plan maintenance ---------------------------------------------------

    def replace_plan(self, new_plan: ShardPlan) -> None:
        """Swap the shard plan (after unlearning re-indexed rows)."""
        if new_plan.n_shards != self.plan.n_shards:
            raise ValueError("Shard count must remain constant")
        self.plan = new_plan

    # ---- serialisation ------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "n_shards": self.plan.n_shards,
            "shard_of_row": self.plan.shard_of_row.tolist(),
            "indices": [idx.tolist() for idx in self.plan.indices],
            "learners": [
                None if lr is None else {"class": lr.__class__.__name__, "state": lr.state_dict()}
                for lr in self.learners
            ],
        }
