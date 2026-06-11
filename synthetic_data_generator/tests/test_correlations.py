"""Tests for the correlations module (v2 schema)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from generator.correlations import apply_correlations


@pytest.fixture
def rng():
  return np.random.default_rng(42)


@pytest.fixture
def base_df(rng):
  """Create a base DataFrame with independently generated v2 features."""
  n = 10000
  return pd.DataFrame({
    "user_id": [f"USR-{i:06d}" for i in range(1, n + 1)],
    "age": rng.normal(35, 10, n).clip(18, 75),
    "purchase_amount": rng.lognormal(4.5, 1.0, n),
    "purchase_count": rng.normal(12, 5, n).clip(0, 100).astype(int),
    "avg_check": rng.normal(450, 150, n).clip(0, None),
    "days_since_last_purchase": rng.exponential(20.0, n).clip(0, 365),
    "device_type": rng.choice(
      ["mobile", "desktop", "tablet"], n, p=[0.55, 0.35, 0.10]
    ),
    "region": rng.choice(
      ["Moscow", "SPb", "Novosibirsk", "Ekaterinburg", "Other"],
      n,
      p=[0.30, 0.20, 0.10, 0.10, 0.30],
    ),
    "is_premium": rng.random(n) < 0.15,
    "is_subscribed": rng.random(n) < 0.40,
    "consent_given": rng.random(n) < 0.85,
    "deletion_requested": rng.random(n) < 0.05,
  })


class TestCholesky:
  def test_purchase_amount_count_correlation(self, base_df, rng):
    corr_config = {"purchase_amount_count": 0.5}
    result = apply_correlations(base_df, corr_config, rng)

    r, _ = spearmanr(
      result["purchase_amount"], result["purchase_count"]
    )
    assert abs(r - 0.5) < 0.15, f"Expected ~0.5, got {r:.3f}"

  def test_purchase_amount_avg_check_correlation(self, base_df, rng):
    corr_config = {"purchase_amount_avg_check": 0.6}
    result = apply_correlations(base_df, corr_config, rng)

    r, _ = spearmanr(
      result["purchase_amount"], result["avg_check"]
    )
    assert abs(r - 0.6) < 0.15, f"Expected ~0.6, got {r:.3f}"

  def test_days_purchase_count_correlation(self, base_df, rng):
    corr_config = {"days_since_purchase_count": -0.4}
    result = apply_correlations(base_df, corr_config, rng)

    r, _ = spearmanr(
      result["days_since_last_purchase"], result["purchase_count"]
    )
    assert abs(r - (-0.4)) < 0.15, f"Expected ~-0.4, got {r:.3f}"

  def test_preserves_marginals(self, base_df, rng):
    """After correlation injection, marginal distributions should be preserved."""
    corr_config = {"purchase_amount_count": 0.5}
    original_sorted = np.sort(base_df["purchase_amount"].values)
    result = apply_correlations(base_df, corr_config, rng)
    result_sorted = np.sort(result["purchase_amount"].values)
    np.testing.assert_array_almost_equal(original_sorted, result_sorted)


class TestAgeMobile:
  def test_young_more_mobile(self, base_df, rng):
    corr_config = {"age_mobile": -0.3}
    result = apply_correlations(base_df, corr_config, rng)

    young = result[result["age"] < 25]
    old = result[result["age"] > 50]

    young_mobile = (young["device_type"] == "mobile").mean()
    old_mobile = (old["device_type"] == "mobile").mean()

    assert young_mobile > old_mobile, (
      f"Young mobile rate ({young_mobile:.2f}) should exceed "
      f"old mobile rate ({old_mobile:.2f})"
    )


class TestConsentPurchase:
  def test_consent_users_spend_more(self, base_df, rng):
    corr_config = {"consent_purchase": 0.3}
    result = apply_correlations(base_df, corr_config, rng)

    consent_mean = result[result["consent_given"]]["purchase_amount"].mean()
    no_consent_mean = result[~result["consent_given"]]["purchase_amount"].mean()

    assert consent_mean > no_consent_mean, (
      f"Consent mean ({consent_mean:.2f}) should exceed "
      f"no-consent mean ({no_consent_mean:.2f})"
    )


class TestEmptyConfig:
  def test_no_correlations(self, base_df, rng):
    result = apply_correlations(base_df, {}, rng)
    pd.testing.assert_frame_equal(result, base_df)
