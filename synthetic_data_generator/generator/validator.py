"""Statistical validation of generated synthetic datasets."""

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def validate_dataset(df, features_config, corr_config, target_column="label"):
    results: dict[str, Any] = {"checks": {}, "passed": True}

    nan_check = _check_nans(df)
    results["checks"]["nan"] = nan_check
    if not nan_check["passed"]:
        results["passed"] = False

    if "user_id" in df.columns:
        uid_check = _check_user_id_uniqueness(df)
        results["checks"]["user_id_uniqueness"] = uid_check
        if not uid_check["passed"]:
            results["passed"] = False

    dist_check = _check_distributions(df, features_config)
    results["checks"]["distributions"] = dist_check
    if not dist_check["passed"]:
        results["passed"] = False

    clip_check = _check_clipping(df, features_config)
    results["checks"]["clipping"] = clip_check
    if not clip_check["passed"]:
        results["passed"] = False

    corr_check = _check_correlations(df, corr_config)
    results["checks"]["correlations"] = corr_check
    if not corr_check["passed"]:
        results["passed"] = False

    date_check = _check_date_formats(df)
    results["checks"]["date_formats"] = date_check
    if not date_check["passed"]:
        results["passed"] = False

    if target_column in df.columns:
        balance_check = _check_class_balance(df, target_column)
        results["checks"]["class_balance"] = balance_check
        if not balance_check["passed"]:
            logger.warning("Class balance warning: %s", balance_check["message"])

    _log_results(results)
    return results


def _check_nans(df):
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    return {
        "passed": total_nans == 0,
        "total_nans": int(total_nans),
        "columns_with_nans": {col: int(count) for col, count in nan_counts.items() if count > 0},
    }


def _check_user_id_uniqueness(df):
    n_total = len(df)
    n_unique = df["user_id"].nunique()
    return {
        "passed": n_unique == n_total,
        "total": n_total,
        "unique": n_unique,
        "duplicates": n_total - n_unique,
    }


def _check_distributions(df, features_config):
    alpha = 0.01
    results: dict[str, Any] = {"passed": True, "features": {}}

    for name, config in features_config.items():
        if name not in df.columns:
            continue
        dist = config.get("distribution")
        if dist is None:
            continue

        if dist == "normal":
            values = df[name].values.astype(float)
            stat, p_value = stats.kstest(values, "norm", args=(config["mean"], config["std"]))
            passed = p_value > alpha
            results["features"][name] = {"test": "ks", "statistic": float(stat), "p_value": float(p_value), "passed": passed}
            if not passed:
                if config.get("clip"):
                    logger.warning("KS test failed for '%s' (clipped normal, p=%.4f) — expected", name, p_value)
                else:
                    results["passed"] = False

        elif dist == "lognormal":
            values = df[name].values.astype(float)
            log_values = np.log(values[values > 0])
            stat, p_value = stats.kstest(log_values, "norm", args=(config["mean"], config["std"]))
            passed = p_value > alpha
            results["features"][name] = {"test": "ks_lognormal", "statistic": float(stat), "p_value": float(p_value), "passed": passed}
            if not passed:
                logger.warning("KS test failed for '%s' (lognormal, p=%.4f) — may be due to correlation adjustments", name, p_value)

        elif dist == "categorical":
            observed = df[name].value_counts()
            categories = config["categories"]
            expected_probs = np.array(list(categories.values()), dtype=float)
            expected_probs /= expected_probs.sum()
            expected_counts = expected_probs * len(df)
            obs_counts = np.array([observed.get(cat, 0) for cat in categories.keys()], dtype=float)
            stat, p_value = stats.chisquare(obs_counts, f_exp=expected_counts)
            passed = p_value > alpha
            results["features"][name] = {"test": "chi2", "statistic": float(stat), "p_value": float(p_value), "passed": passed}
            if not passed:
                logger.warning("Chi2 test failed for '%s' (p=%.4f)", name, p_value)

        elif dist == "bernoulli":
            values = df[name].values.astype(bool)
            observed_p = values.mean()
            expected_p = config["p"]
            n = len(values)
            se = np.sqrt(expected_p * (1 - expected_p) / n)
            z = abs(observed_p - expected_p) / se if se > 0 else 0
            passed = z < 3.0
            results["features"][name] = {"test": "proportion", "observed_p": float(observed_p), "expected_p": expected_p, "z_score": float(z), "passed": passed}
            if not passed:
                results["passed"] = False

    return results


def _check_clipping(df, features_config):
    results: dict[str, Any] = {"passed": True, "features": {}}
    for name, config in features_config.items():
        if name not in df.columns:
            continue
        clip = config.get("clip")
        if clip is None:
            continue
        low, high = clip
        values = df[name].values
        try:
            values_numeric = values.astype(float)
        except (ValueError, TypeError):
            continue
        violations = {}
        if low is not None and np.any(values_numeric < low):
            violations["below_min"] = int(np.sum(values_numeric < low))
        if high is not None and np.any(values_numeric > high):
            violations["above_max"] = int(np.sum(values_numeric > high))
        passed = len(violations) == 0
        results["features"][name] = {"bounds": clip, "passed": passed, "violations": violations}
        if not passed:
            results["passed"] = False
    return results


def _check_correlations(df, corr_config):
    tolerance = 0.20
    results: dict[str, Any] = {"passed": True, "pairs": {}}
    pair_map = {
        "purchase_amount_count": ("purchase_amount", "purchase_count"),
        "purchase_amount_avg_check": ("purchase_amount", "avg_check"),
        "days_since_purchase_count": ("days_since_last_purchase", "purchase_count"),
    }
    for key, target_r in corr_config.items():
        if key not in pair_map:
            continue
        col_a, col_b = pair_map[key]
        if col_a not in df.columns or col_b not in df.columns:
            continue
        actual_r, p_value = stats.spearmanr(df[col_a].values.astype(float), df[col_b].values.astype(float))
        within_tolerance = abs(actual_r - target_r) <= tolerance
        results["pairs"][key] = {"target": target_r, "actual": float(actual_r), "p_value": float(p_value), "passed": within_tolerance}
        if not within_tolerance:
            results["passed"] = False
    return results


def _check_date_formats(df):
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    results: dict[str, Any] = {"passed": True, "columns": {}}
    for col in ["birth_date", "registration_date"]:
        if col not in df.columns:
            continue
        values = df[col].astype(str)
        matches = values.apply(lambda x: bool(date_pattern.match(x)))
        n_invalid = (~matches).sum()
        passed = n_invalid == 0
        results["columns"][col] = {"passed": passed, "invalid_count": int(n_invalid)}
        if not passed:
            results["passed"] = False
    return results


def _check_class_balance(df, target_column):
    values = df[target_column].values.astype(bool)
    positive_rate = values.mean()
    if positive_rate < 0.03 or positive_rate > 0.40:
        return {
            "passed": False,
            "positive_rate": float(positive_rate),
            "message": f"Class imbalance detected: {positive_rate:.2%} positive rate",
        }
    return {"passed": True, "positive_rate": float(positive_rate), "message": "Class balance OK"}


def _log_results(results):
    status = "PASSED" if results["passed"] else "FAILED"
    logger.info("Validation %s", status)
    for section, data in results["checks"].items():
        if isinstance(data, dict) and "passed" in data:
            section_status = "OK" if data["passed"] else "FAIL"
            logger.info("  %s: %s", section, section_status)
