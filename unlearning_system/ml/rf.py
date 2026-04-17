"""RandomForest wrapper conforming to :class:`BaseLearner`.

RF is a natural second model to pair with the PyTorch LogReg:
  * non-linear decision boundary,
  * robust to scaling differences (so a sanity check on preprocessing),
  * completely different bias-variance profile, which helps interpret the
    delta-metrics after unlearning.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseLearner, TrainingStats


@dataclass
class RFConfig:
    n_estimators: int = 100
    max_depth: int | None = 12
    min_samples_leaf: int = 2
    n_jobs: int = 1
    class_weight: str | None = "balanced"
    seed: int = 0


class RFLearner(BaseLearner):
    name = "random_forest"

    def __init__(self, config: RFConfig | None = None) -> None:
        self.cfg = config or RFConfig()
        self._reset()
        self.stats_: TrainingStats | None = None

    def _reset(self) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            n_jobs=self.cfg.n_jobs,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFLearner":
        start = time.time()
        # sklearn refuses to fit when only one class is present in a shard;
        # in that degenerate case fall back to a constant predictor.
        unique = np.unique(y)
        if len(unique) < 2:
            self._only_class = int(unique[0])
            self.clf = None  # mark constant mode
            self.stats_ = TrainingStats(n_samples=len(y), wall_time_sec=time.time() - start)
            return self
        self._only_class = None
        self._reset()
        self.clf.fit(X, y)
        self.stats_ = TrainingStats(
            n_samples=len(y),
            wall_time_sec=time.time() - start,
            n_iterations=self.cfg.n_estimators,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_only_class", None) is not None:
            return np.full(shape=(len(X),), fill_value=float(self._only_class))
        if self.clf is None:
            raise RuntimeError("Call fit() before predict_proba().")
        proba = self.clf.predict_proba(X)
        # return P(class=1) column; guard for single-column edge case
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]

    def state_dict(self) -> dict:
        import joblib
        import io
        buf = io.BytesIO()
        joblib.dump({"clf": self.clf, "only_class": getattr(self, "_only_class", None)}, buf)
        return {"config": asdict(self.cfg), "blob": buf.getvalue()}

    def load_state_dict(self, state: dict) -> None:
        import joblib
        import io
        self.cfg = RFConfig(**state["config"])
        payload = joblib.load(io.BytesIO(state["blob"]))
        self.clf = payload["clf"]
        self._only_class = payload["only_class"]
