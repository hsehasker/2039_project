"""Abstract interface shared by every shard-level learner.

Every model used in a SISA ensemble must expose a uniform contract:

  * ``fit(X, y)``            — train from scratch on the shard's data
  * ``predict_proba(X)``     — return P(y=1 | x) for binary classification
  * ``predict(X)``           — hard labels (threshold 0.5 by default)
  * ``state_dict()``         — serialisable dict for persistence
  * ``load_state_dict(sd)``  — restore from dict

The contract is kept intentionally small so that heterogeneous backends
(PyTorch, scikit-learn, possibly a gradient booster later) slot in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class BaseLearner(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseLearner": ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None: ...


@dataclass
class TrainingStats:
    """Bookkeeping returned by ``fit`` for reporting / metrics aggregation."""
    n_samples: int
    wall_time_sec: float
    final_loss: float | None = None
    n_iterations: int | None = None
