"""PyTorch-based binary logistic regression.

Why PyTorch and not ``sklearn.linear_model.LogisticRegression``?
  * The ТЗ requires PyTorch as one of the ML frameworks.
  * We need explicit access to the training loop so that the unlearning
    module can, in the future, swap in a *certified* unlearning algorithm
    (e.g. Newton-step influence removal, gradient surgery) without
    replacing the model class.

The model is a single linear layer + sigmoid, optimised with Adam on the
binary cross-entropy loss. Training is done in full-batch mode for small
shards (SISA shards are typically < 10 k rows), which gives us a smooth,
deterministic loss curve — important for reproducibility.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .base import BaseLearner, TrainingStats


@dataclass
class LogRegConfig:
    lr: float = 0.05
    weight_decay: float = 1e-4     # L2 regularisation — also a convexifier
    epochs: int = 200
    batch_size: int = 0            # 0 == full-batch
    device: str = "cpu"
    seed: int = 0
    patience: int = 20             # early stopping on train-loss plateau
    tol: float = 1e-5
    class_weight: str | None = "balanced"   # None | "balanced"


class TorchLogReg(BaseLearner):
    """Binary logistic regression implemented in PyTorch."""
    name = "torch_logreg"

    def __init__(self, config: LogRegConfig | None = None, n_features: int | None = None) -> None:
        self.cfg = config or LogRegConfig()
        self.n_features = n_features
        self._build_model()
        self.stats_: TrainingStats | None = None

    # ---- building blocks ----------------------------------------------------

    def _build_model(self) -> None:
        torch.manual_seed(self.cfg.seed)
        self.model: nn.Module | None = None
        if self.n_features is not None:
            self.model = nn.Linear(self.n_features, 1).to(self.cfg.device)

    def _ensure_built(self, n_features: int) -> None:
        if self.model is None or self.n_features != n_features:
            self.n_features = n_features
            self._build_model()

    # ---- BaseLearner API ----------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchLogReg":
        self._ensure_built(X.shape[1])
        device = torch.device(self.cfg.device)

        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)

        pos_weight = None
        if self.cfg.class_weight == "balanced":
            n_pos = float((y_t == 1).sum().item())
            n_neg = float((y_t == 0).sum().item())
            if n_pos > 0 and n_neg > 0:
                pos_weight = torch.tensor([n_neg / n_pos], device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        n = X_t.shape[0]
        batch_size = self.cfg.batch_size if self.cfg.batch_size > 0 else n

        best_loss = float("inf")
        best_state: dict | None = None
        stale_epochs = 0
        final_loss: float | None = None
        epoch = 0
        start = time.time()

        rng = np.random.default_rng(self.cfg.seed)
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            perm = rng.permutation(n)
            epoch_losses: list[float] = []
            for i in range(0, n, batch_size):
                batch_idx = perm[i : i + batch_size]
                xb = X_t[batch_idx]
                yb = y_t[batch_idx]
                optim.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
                epoch_losses.append(float(loss.item()))
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            final_loss = avg_loss

            if avg_loss < best_loss - self.cfg.tol:
                best_loss = avg_loss
                stale_epochs = 0
                # snapshot best weights so early stopping doesn't leave us
                # with a worse iterate
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                stale_epochs += 1
                if stale_epochs >= self.cfg.patience:
                    break

        # Restore best observed weights — important for unlearning where we
        # compare metrics across re-fits with tight tolerances.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.stats_ = TrainingStats(
            n_samples=n,
            wall_time_sec=time.time() - start,
            final_loss=final_loss,
            n_iterations=epoch,
        )
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        device = torch.device(self.cfg.device)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        self.model.eval()
        logits = self.model(X_t).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()

    # ---- serialisation ------------------------------------------------------

    def state_dict(self) -> dict:
        if self.model is None:
            raise RuntimeError("Cannot serialise: model not built.")
        return {
            "config": self.cfg.__dict__,
            "n_features": self.n_features,
            "weights": {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.cfg = LogRegConfig(**state["config"])
        self.n_features = int(state["n_features"])
        self._build_model()
        torch_state = {k: torch.tensor(v) for k, v in state["weights"].items()}
        self.model.load_state_dict(torch_state)
