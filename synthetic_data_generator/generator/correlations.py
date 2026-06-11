"""Correlation structure injection for synthetic data."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def apply_correlations(df, corr_config, rng):
    df = df.copy()

    numeric_pairs = _get_numeric_pairs(corr_config)
    if numeric_pairs:
        df = _apply_cholesky_correlations(df, numeric_pairs)

    if "age_mobile" in corr_config:
        df = _apply_age_mobile_correlation(df, corr_config["age_mobile"], rng)

    if "consent_purchase" in corr_config:
        df = _apply_consent_purchase_correlation(df, corr_config["consent_purchase"], rng)

    if "premium_purchase" in corr_config:
        df = _apply_premium_purchase_correlation(df, corr_config["premium_purchase"], rng)

    return df


def _get_numeric_pairs(corr_config):
    mapping = {
        "purchase_amount_count": ("purchase_amount", "purchase_count"),
        "purchase_amount_avg_check": ("purchase_amount", "avg_check"),
        "days_since_purchase_count": ("days_since_last_purchase", "purchase_count"),
    }
    pairs = []
    for key, (col_a, col_b) in mapping.items():
        if key in corr_config:
            pairs.append((col_a, col_b, corr_config[key]))
    return pairs


def _apply_cholesky_correlations(df, pairs):
    for col_a, col_b, target_r in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        original_a = df[col_a].values.copy()
        original_b = df[col_b].values.copy()
        z_a = _rank_to_normal(original_a)
        z_b = _rank_to_normal(original_b)
        corr_matrix = np.array([[1.0, target_r], [target_r, 1.0]])
        L = np.linalg.cholesky(corr_matrix)
        z_independent = np.column_stack([z_a, z_b])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            z_correlated = z_independent @ L.T
        z_correlated = np.nan_to_num(z_correlated, nan=0.0, posinf=6.0, neginf=-6.0)
        df[col_a] = _match_ranks(original_a, z_correlated[:, 0])
        df[col_b] = _match_ranks(original_b, z_correlated[:, 1])
    return df


def _rank_to_normal(values):
    n = len(values)
    ranks = rankdata(values)
    quantiles = np.clip((ranks - 0.5) / n, 1e-8, 1 - 1e-8)
    z = norm.ppf(quantiles)
    return np.nan_to_num(z, nan=0.0, posinf=6.0, neginf=-6.0)


def _match_ranks(original, z_scores):
    sorted_original = np.sort(original)
    target_order = np.argsort(np.argsort(z_scores))
    return sorted_original[target_order]


def _apply_age_mobile_correlation(df, strength, rng):
    if "age" not in df.columns or "device_type" not in df.columns:
        return df
    n = len(df)
    age_quantiles = rankdata(df["age"].values) / n
    device_counts = df["device_type"].value_counts(normalize=True)
    p_mobile_base = device_counts.get("mobile", 0.55)
    p_desktop_base = device_counts.get("desktop", 0.35)
    abs_strength = abs(strength)
    new_devices = []
    for q in age_quantiles:
        age_factor = (0.5 - q) * abs_strength * 2
        p_mobile = np.clip(p_mobile_base + age_factor, 0.05, 0.95)
        p_desktop = np.clip(p_desktop_base - age_factor * 0.7, 0.05, 0.95)
        p_tablet = max(1.0 - p_mobile - p_desktop, 0.01)
        probs = np.array([p_mobile, p_desktop, p_tablet])
        probs /= probs.sum()
        new_devices.append(rng.choice(["mobile", "desktop", "tablet"], p=probs))
    df["device_type"] = new_devices
    return df


def _apply_consent_purchase_correlation(df, strength, rng):
    if "consent_given" not in df.columns or "purchase_amount" not in df.columns:
        return df
    mask = df["consent_given"].values.astype(bool)
    boost = 1.0 + strength * 2.0
    df.loc[mask, "purchase_amount"] = df.loc[mask, "purchase_amount"] * boost
    return df


def _apply_premium_purchase_correlation(df, strength, rng):
    if "is_premium" not in df.columns or "purchase_amount" not in df.columns:
        return df
    mask = df["is_premium"].values.astype(bool)
    boost = 1.0 + strength * 2.0
    df.loc[mask, "purchase_amount"] = df.loc[mask, "purchase_amount"] * boost
    return df
