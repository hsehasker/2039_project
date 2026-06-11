"""Labeling module for synthetic data."""

from typing import Any

import numpy as np
import pandas as pd


def apply_labels(df, labeling_config):
    mode = labeling_config.get("mode", "rule")
    target_col = labeling_config.get("target_column", "label")

    if mode == "rule":
        labels = _rule_based_labeling(df)
    elif mode == "probabilistic":
        labels = _probabilistic_labeling(df)
    else:
        raise ValueError(f"Unknown labeling mode: '{mode}'")

    df = df.copy()
    df[target_col] = labels.astype(int)
    return df


def _rule_based_labeling(df):
    deletion = df["deletion_requested"].values.astype(bool)
    no_consent = ~df["consent_given"].values.astype(bool)
    return deletion | no_consent


def _probabilistic_labeling(df):
    deletion = df["deletion_requested"].values.astype(float)
    no_consent = 1.0 - df["consent_given"].values.astype(float)
    inactive = (df["days_since_last_purchase"].values > 180).astype(float)
    zero_purchases = (df["purchase_count"].values == 0).astype(float)

    score = (
        2.0 * deletion
        + 1.5 * no_consent
        + 0.5 * inactive
        + 0.3 * zero_purchases
    )
    prob = 1.0 / (1.0 + np.exp(-(score - 2.0)))
    rng = np.random.default_rng()
    return rng.random(size=len(df)) < prob
