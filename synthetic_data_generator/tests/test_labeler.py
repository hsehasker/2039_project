"""Tests for the labeler module (v2 schema)."""

import numpy as np
import pandas as pd
import pytest

from generator.labeler import apply_labels


@pytest.fixture
def sample_df():
  """Create a small deterministic DataFrame for rule testing."""
  return pd.DataFrame({
    "user_id": ["USR-000001", "USR-000002", "USR-000003", "USR-000004", "USR-000005"],
    "age": [25, 40, 30, 55, 22],
    "purchase_amount": [100.0, 2000.0, 300.0, 100.0, 50.0],
    "purchase_count": [5, 15, 3, 8, 0],
    "avg_check": [200, 500, 100, 300, 50],
    "days_since_last_purchase": [50, 5, 200, 40, 100],
    "device_type": ["mobile", "desktop", "mobile", "tablet", "mobile"],
    "region": ["Moscow", "SPb", "Novosibirsk", "Moscow", "Other"],
    "is_premium": [False, True, False, True, False],
    "is_subscribed": [True, True, False, True, False],
    "consent_given": [True, True, False, True, False],
    "deletion_requested": [False, False, False, True, False],
  })


class TestRuleBasedLabeling:
  def test_consent_no_deletion_label_0(self, sample_df):
    """User 1: consent=True, deletion=False -> label=0."""
    config = {"mode": "rule", "target_column": "label"}
    result = apply_labels(sample_df, config)
    assert result.loc[0, "label"] == 0

  def test_consent_no_deletion_label_0_second(self, sample_df):
    """User 2: consent=True, deletion=False -> label=0."""
    config = {"mode": "rule", "target_column": "label"}
    result = apply_labels(sample_df, config)
    assert result.loc[1, "label"] == 0

  def test_no_consent_label_1(self, sample_df):
    """User 3: consent=False -> label=1 (regardless of deletion)."""
    config = {"mode": "rule", "target_column": "label"}
    result = apply_labels(sample_df, config)
    assert result.loc[2, "label"] == 1

  def test_deletion_requested_label_1(self, sample_df):
    """User 4: deletion=True -> label=1 (regardless of consent)."""
    config = {"mode": "rule", "target_column": "label"}
    result = apply_labels(sample_df, config)
    assert result.loc[3, "label"] == 1

  def test_no_consent_no_deletion_label_1(self, sample_df):
    """User 5: consent=False, deletion=False -> label=1."""
    config = {"mode": "rule", "target_column": "label"}
    result = apply_labels(sample_df, config)
    assert result.loc[4, "label"] == 1

  def test_target_column_exists(self, sample_df):
    config = {"mode": "rule", "target_column": "label"}
    result = apply_labels(sample_df, config)
    assert "label" in result.columns

  def test_custom_target_name(self, sample_df):
    config = {"mode": "rule", "target_column": "delete_flag"}
    result = apply_labels(sample_df, config)
    assert "delete_flag" in result.columns

  def test_label_is_int(self, sample_df):
    config = {"mode": "rule", "target_column": "label"}
    result = apply_labels(sample_df, config)
    assert result["label"].dtype in [np.int64, np.int32, int]


class TestProbabilisticLabeling:
  def test_probabilistic_mode(self, sample_df):
    config = {
      "mode": "probabilistic",
      "target_column": "label",
    }
    result = apply_labels(sample_df, config)
    assert "label" in result.columns

  def test_probabilistic_large_dataset(self):
    """On a large dataset, probabilistic labeling should produce both classes."""
    rng = np.random.default_rng(42)
    n = 5000
    df = pd.DataFrame({
      "deletion_requested": rng.random(n) < 0.05,
      "consent_given": rng.random(n) < 0.85,
      "days_since_last_purchase": rng.exponential(20.0, n),
      "purchase_count": rng.normal(12, 5, n).clip(0, 100).astype(int),
    })
    config = {
      "mode": "probabilistic",
      "target_column": "label",
    }
    result = apply_labels(df, config)
    true_rate = result["label"].astype(bool).mean()
    # Should have a mix of both classes
    assert 0.01 < true_rate < 0.99


class TestInvalidMode:
  def test_unknown_mode_raises(self, sample_df):
    config = {"mode": "unknown", "target_column": "label"}
    with pytest.raises(ValueError, match="Unknown labeling mode"):
      apply_labels(sample_df, config)
