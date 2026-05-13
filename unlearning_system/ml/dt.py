"""DecisionTreeClassifier wrapper conforming to :class:`BaseLearner`.

Implements Decision Tree as required by TЗ.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .base import BaseLearner, TrainingStats


@dataclass
class DTConfig:
    max_depth: int | None = None
    min_samples_leaf: int = 1
    class_weight: str | None = None
    seed: int = 0


class DTLearner(BaseLearner):
    name = "decision_tree"

    def __init__(self, config: DTConfig | None = None) -> None:
        self.cfg = config or DTConfig()
        self._reset()
        self.stats_: TrainingStats | None = None

    def _reset(self) -> None:
        self.clf = DecisionTreeClassifier(
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DTLearner":
        start = time.time()
        unique = np.unique(y)
        if len(unique) < 2:
            self._only_class = int(unique[0])
            self.clf = None
            self.stats_ = TrainingStats(n_samples=len(y), wall_time_sec=time.time() - start)
            return self
        self._only_class = None
        self._reset()
        self.clf.fit(X, y)
        self.stats_ = TrainingStats(
            n_samples=len(y),
            wall_time_sec=time.time() - start,
            n_iterations=1,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_only_class", None) is not None:
            return np.full(shape=(len(X),), fill_value=float(self._only_class))
        if self.clf is None:
            raise RuntimeError("Call fit() before predict_proba().")
        proba = self.clf.predict_proba(X)
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
        self.cfg = DTConfig(**state["config"])
        payload = joblib.load(io.BytesIO(state["blob"]))
        self.clf = payload["clf"]
        self._only_class = payload["only_class"]
