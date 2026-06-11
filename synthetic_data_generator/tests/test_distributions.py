"""Tests for the distributions module (v2 schema)."""

import re

import numpy as np
import pytest

from generator.distributions import (
    generate_feature,
    LAST_NAMES_MALE, LAST_NAMES_FEMALE,
    FIRST_NAMES_MALE, FIRST_NAMES_FEMALE,
    MIDDLE_NAMES_MALE, MIDDLE_NAMES_FEMALE
)

@pytest.fixture
def rng():
  return np.random.default_rng(42)


class TestSequentialId:
  def test_sequential_ids_format(self, rng):
    config = {"type": "sequential_id", "prefix": "USR"}
    result = generate_feature("user_id", config, 100, rng)
    assert len(result) == 100
    assert result[0] == "USR-000001"
    assert result[-1] == "USR-000100"

  def test_sequential_ids_unique(self, rng):
    config = {"type": "sequential_id", "prefix": "USR"}
    result = generate_feature("user_id", config, 1000, rng)
    assert len(set(result)) == 1000


class TestFIO:
  def test_last_names_from_list(self, rng):
    config = {"type": "categorical"}
    result = generate_feature("last_name", config, 1000, rng)
    assert len(result) == 1000
    all_last = set(LAST_NAMES_MALE) | set(LAST_NAMES_FEMALE)
    assert set(result).issubset(all_last)

  def test_first_names_from_list(self, rng):
    config = {"type": "categorical"}
    result = generate_feature("first_name", config, 1000, rng)
    assert len(result) == 1000
    all_first = set(FIRST_NAMES_MALE) | set(FIRST_NAMES_FEMALE)
    assert set(result).issubset(all_first)

  def test_middle_names_from_list(self, rng):
    config = {"type": "categorical"}
    result = generate_feature("middle_name", config, 1000, rng)
    assert len(result) == 1000
    all_middle = set(MIDDLE_NAMES_MALE) | set(MIDDLE_NAMES_FEMALE)
    assert set(result).issubset(all_middle)


class TestNormal:
  def test_normal_shape(self, rng):
    config = {"distribution": "normal", "mean": 35, "std": 10, "clip": [18, 75]}
    result = generate_feature("age", config, 10000, rng)
    assert len(result) == 10000

  def test_normal_clipping(self, rng):
    config = {"distribution": "normal", "mean": 35, "std": 10, "clip": [18, 75]}
    result = generate_feature("age", config, 10000, rng)
    assert result.min() >= 18
    assert result.max() <= 75

  def test_normal_mean_approximate(self, rng):
    config = {"distribution": "normal", "mean": 12, "std": 5, "clip": [0, 100]}
    result = generate_feature("purchase_count", config, 50000, rng)
    assert abs(result.mean() - 12) < 2

  def test_normal_no_clip(self, rng):
    config = {"distribution": "normal", "mean": 0, "std": 1}
    result = generate_feature("test", config, 1000, rng)
    assert len(result) == 1000


class TestLognormal:
  def test_lognormal_positive(self, rng):
    config = {"distribution": "lognormal", "mean": 4.5, "std": 1.0}
    result = generate_feature("purchase_amount", config, 10000, rng)
    assert (result > 0).all()

  def test_lognormal_rounded(self, rng):
    config = {"distribution": "lognormal", "mean": 4.5, "std": 1.0}
    result = generate_feature("purchase_amount", config, 100, rng)
    # Check values are rounded to 2 decimals
    np.testing.assert_array_equal(result, np.round(result, 2))


class TestExponential:
  def test_exponential_positive(self, rng):
    config = {"distribution": "exponential", "lambda": 0.05, "clip": [0, 365]}
    result = generate_feature("days_since_last_purchase", config, 10000, rng)
    assert result.min() >= 0

  def test_exponential_clipping(self, rng):
    config = {"distribution": "exponential", "lambda": 0.05, "clip": [0, 365]}
    result = generate_feature("days_since_last_purchase", config, 10000, rng)
    assert result.max() <= 365


class TestCategorical:
  def test_categorical_values(self, rng):
    config = {
      "distribution": "categorical",
      "categories": {"mobile": 0.55, "desktop": 0.35, "tablet": 0.10},
    }
    result = generate_feature("device_type", config, 10000, rng)
    unique = set(result)
    assert unique <= {"mobile", "desktop", "tablet"}

  def test_categorical_proportions(self, rng):
    config = {
      "distribution": "categorical",
      "categories": {"mobile": 0.55, "desktop": 0.35, "tablet": 0.10},
    }
    result = generate_feature("device_type", config, 50000, rng)
    mobile_frac = (result == "mobile").mean()
    assert abs(mobile_frac - 0.55) < 0.02

  def test_region_values(self, rng):
    config = {
      "distribution": "categorical",
      "categories": {"Moscow": 0.30, "SPb": 0.20, "Novosibirsk": 0.10, "Ekaterinburg": 0.10, "Other": 0.30},
    }
    result = generate_feature("region", config, 10000, rng)
    assert set(result) <= {"Moscow", "SPb", "Novosibirsk", "Ekaterinburg", "Other"}


class TestBernoulli:
  def test_bernoulli_boolean(self, rng):
    config = {"distribution": "bernoulli", "p": 0.85}
    result = generate_feature("consent_given", config, 1000, rng)
    assert result.dtype == bool

  def test_bernoulli_proportion(self, rng):
    config = {"distribution": "bernoulli", "p": 0.85}
    result = generate_feature("consent_given", config, 50000, rng)
    assert abs(result.mean() - 0.85) < 0.02

  def test_deletion_requested_low_p(self, rng):
    config = {"distribution": "bernoulli", "p": 0.05}
    result = generate_feature("deletion_requested", config, 50000, rng)
    assert abs(result.mean() - 0.05) < 0.02


class TestTemplateEmail:
  def test_email_format(self, rng):
    config = {
      "type": "template_email",
      "domains": {"mail.ru": 0.4, "gmail.com": 0.3, "yandex.ru": 0.3},
    }
    ids = np.arange(1, 101)
    result = generate_feature("email", config, 100, rng, id_array=ids)
    assert len(result) == 100
    for email in result:
        assert re.match(r"^[\w.]{6,15}@[\w.]+$", email)

  def test_email_domains(self, rng):
    config = {
      "type": "template_email",
      "domains": {"mail.ru": 0.4, "gmail.com": 0.3, "yandex.ru": 0.3},
    }
    ids = np.arange(1, 10001)
    result = generate_feature("email", config, 10000, rng, id_array=ids)
    domains = [e.split("@")[1] for e in result]
    assert set(domains) <= {"mail.ru", "gmail.com", "yandex.ru"}


class TestTemplatePhone:
  def test_phone_format(self, rng):
    config = {"type": "template_phone"}
    result = generate_feature("phone", config, 100, rng)
    assert len(result) == 100
    for phone in result:
      assert re.match(r"^\+7-9\d{2}-\d{3}-\d{2}-\d{2}$", phone)


class TestDateFromAge:
  def test_birth_date_format(self, rng):
    ages = np.array([25, 30, 40, 50])
    config = {"type": "derived_from_age"}
    result = generate_feature("birth_date", config, 4, rng, reference_date="2025-01-01", age_array=ages)
    for d in result:
      assert re.match(r"^\d{4}-\d{2}-\d{2}$", d)

  def test_birth_date_requires_age(self, rng):
    config = {"type": "derived_from_age"}
    with pytest.raises(ValueError, match="age_array is required"):
      generate_feature("birth_date", config, 10, rng)


class TestRandomDateRange:
  def test_date_in_range(self, rng):
    config = {"type": "random_date_range", "start": "2020-01-01", "end": "2025-01-01"}
    result = generate_feature("registration_date", config, 1000, rng)
    for d in result:
      assert re.match(r"^\d{4}-\d{2}-\d{2}$", d)
      year = int(d[:4])
      assert 2020 <= year <= 2025


class TestUnknownDistribution:
  def test_raises_on_unknown(self, rng):
    with pytest.raises(ValueError, match="Unknown distribution"):
      generate_feature("x", {"distribution": "gamma"}, 100, rng)

  def test_raises_on_unknown_type(self, rng):
    with pytest.raises(ValueError, match="Unknown type"):
      generate_feature("x", {"type": "magic"}, 100, rng)
