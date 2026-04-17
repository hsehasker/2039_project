"""Feature preprocessing pipeline for the synthetic dataset.

The generator produces a mixture of:
  * numeric columns (age, purchase_amount, purchase_count, avg_check, ...)
  * low-cardinality categoricals (region, device_type, preferred_category)
  * boolean flags (is_premium, is_subscribed, consent_given, deletion_requested)
  * PII columns (user_id, first/last/middle name, email, phone, dates)
  * the target ``label``

For learning we drop PII (otherwise the model would trivially memorise users),
one-hot-encode categoricals and z-score numerics. Booleans are cast to int.

Fitted state is stored on the :class:`FeaturePipeline` object so we can
transform new rows — and, crucially, *re-transform the same rows after an
unlearning request* — using exactly the same encoding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# Columns that uniquely identify a user and MUST be excluded from training
# inputs to preserve meaningful unlearning semantics. They stay in the
# DataFrame for bookkeeping (e.g. delete-by-ID).
PII_COLUMNS: tuple[str, ...] = (
    "user_id", "first_name", "last_name", "middle_name",
    "email", "phone", "birth_date", "registration_date",
)

DEFAULT_TARGET = "label"
DEFAULT_CATEGORICAL = ("region", "device_type", "preferred_category")
DEFAULT_BOOLEAN = ("is_premium", "is_subscribed", "consent_given")
DEFAULT_NUMERIC = ("age", "purchase_amount", "purchase_count",
                   "avg_check", "days_since_last_purchase")


@dataclass
class FeaturePipeline:
    """Fits z-score + one-hot encoding and keeps state for reuse.

    The ``deletion_requested`` flag is *not* used as a feature by default,
    because it is highly correlated with the ``label`` (rule-based labeler
    uses it directly) — that would make the classification task trivial.
    """
    numeric: tuple[str, ...] = DEFAULT_NUMERIC
    categorical: tuple[str, ...] = DEFAULT_CATEGORICAL
    boolean: tuple[str, ...] = DEFAULT_BOOLEAN
    target: str = DEFAULT_TARGET

    # Which columns were actually present at ``fit`` time. Transform uses
    # exactly the same set to guarantee stable feature dimensionality —
    # regardless of whether the test DataFrame happens to drop / gain a
    # column (critical for ensuring matrix shapes match across calls).
    fitted_numeric_: list[str] = field(default_factory=list)
    fitted_boolean_: list[str] = field(default_factory=list)
    fitted_categorical_: list[str] = field(default_factory=list)

    numeric_mean_: dict[str, float] = field(default_factory=dict)
    numeric_std_: dict[str, float] = field(default_factory=dict)
    category_levels_: dict[str, list[str]] = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        self.fitted_numeric_ = []
        for col in self.numeric:
            if col not in df.columns:
                continue
            values = df[col].astype(float).values
            self.numeric_mean_[col] = float(values.mean())
            std = float(values.std())
            self.numeric_std_[col] = std if std > 1e-12 else 1.0
            self.fitted_numeric_.append(col)

        self.fitted_boolean_ = [c for c in self.boolean if c in df.columns]

        self.fitted_categorical_ = []
        for col in self.categorical:
            if col not in df.columns:
                continue
            # sorted for determinism across runs
            self.category_levels_[col] = sorted(df[col].astype(str).unique().tolist())
            self.fitted_categorical_.append(col)

        self.feature_names_ = self._compose_feature_names()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # Always iterate over columns that were *fitted*, not whatever the
        # caller happens to pass. Missing columns yield zero-filled blocks,
        # which keeps the output matrix shape stable and prevents silent
        # dimensionality shifts between train and test / before and after
        # unlearning.
        parts: list[np.ndarray] = []
        n = len(df)
        for col in self.fitted_numeric_:
            if col in df.columns:
                values = df[col].astype(float).values
                mu = self.numeric_mean_[col]
                sigma = self.numeric_std_[col]
                parts.append(((values - mu) / sigma).reshape(-1, 1))
            else:
                parts.append(np.zeros((n, 1), dtype=float))

        for col in self.fitted_boolean_:
            if col in df.columns:
                parts.append(df[col].astype(int).values.reshape(-1, 1).astype(float))
            else:
                parts.append(np.zeros((n, 1), dtype=float))

        for col in self.fitted_categorical_:
            levels = self.category_levels_[col]
            onehot = np.zeros((n, len(levels)), dtype=float)
            if col in df.columns:
                lvl_to_idx = {lvl: i for i, lvl in enumerate(levels)}
                idx = df[col].astype(str).map(lvl_to_idx).values
                known_mask = ~pd.isna(idx)
                if known_mask.any():
                    onehot[np.arange(n)[known_mask], idx[known_mask].astype(int)] = 1.0
            parts.append(onehot)

        if not parts:
            return np.empty((n, 0), dtype=float)
        return np.hstack(parts)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def extract_target(self, df: pd.DataFrame) -> np.ndarray:
        if self.target not in df.columns:
            raise KeyError(f"Target column '{self.target}' missing from DataFrame")
        return df[self.target].astype(int).values

    def _compose_feature_names(self) -> list[str]:
        names: list[str] = []
        names.extend([f"num__{c}" for c in self.fitted_numeric_])
        names.extend([f"bool__{c}" for c in self.fitted_boolean_])
        for col in self.fitted_categorical_:
            names.extend([f"cat__{col}={lvl}" for lvl in self.category_levels_[col]])
        return names

    # --- serialisation helpers -------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "boolean": list(self.boolean),
            "target": self.target,
            "numeric_mean": self.numeric_mean_,
            "numeric_std": self.numeric_std_,
            "category_levels": self.category_levels_,
            "feature_names": self.feature_names_,
            "fitted_numeric": self.fitted_numeric_,
            "fitted_boolean": self.fitted_boolean_,
            "fitted_categorical": self.fitted_categorical_,
        }

    @classmethod
    def from_state(cls, state: dict) -> "FeaturePipeline":
        obj = cls(
            numeric=tuple(state["numeric"]),
            categorical=tuple(state["categorical"]),
            boolean=tuple(state["boolean"]),
            target=state["target"],
        )
        obj.numeric_mean_ = dict(state["numeric_mean"])
        obj.numeric_std_ = dict(state["numeric_std"])
        obj.category_levels_ = {k: list(v) for k, v in state["category_levels"].items()}
        obj.feature_names_ = list(state["feature_names"])
        # Backwards-compatible defaults for older manifests that predate
        # the fitted_* fields.
        obj.fitted_numeric_ = list(state.get("fitted_numeric", list(obj.numeric_mean_.keys())))
        obj.fitted_boolean_ = list(state.get("fitted_boolean", list(obj.boolean)))
        obj.fitted_categorical_ = list(state.get("fitted_categorical",
                                                 list(obj.category_levels_.keys())))
        return obj
