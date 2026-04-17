"""End-to-end smoke test for the Machine Unlearning prototype.

Builds a tiny synthetic dataset in-memory, trains a 3-shard SISA ensemble
on PyTorch logistic regression, runs a delete-by-ID unlearning request,
and checks that:
  * only shards that contained the deleted users were retrained;
  * the remaining model still produces reasonable (above-chance) metrics;
  * the reported delta matches the (after − before) values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.metrics import compute_classification_metrics
from ml.pipeline import ExperimentPipeline
from ml.shards import ShardPlan, assign_shards
from ml.unlearning import UnlearnRequest


def _make_tiny_dataset(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(35, 10, size=n).clip(18, 75)
    purchase_amount = rng.lognormal(4.5, 1.0, size=n)
    purchase_count = rng.normal(12, 5, size=n).clip(0, 100)
    avg_check = rng.normal(450, 150, size=n).clip(0, None)
    days = rng.exponential(20, size=n).clip(0, 365)
    region = rng.choice(["Moscow", "SPb", "Other"], size=n, p=[0.4, 0.3, 0.3])
    device = rng.choice(["mobile", "desktop"], size=n, p=[0.7, 0.3])
    category = rng.choice(["A", "B", "C"], size=n)
    is_premium = rng.random(n) < 0.15
    is_subscribed = rng.random(n) < 0.4
    consent = rng.random(n) < 0.85
    deletion = rng.random(n) < 0.05
    label = (deletion | ~consent).astype(int)
    df = pd.DataFrame({
        "user_id": [f"U-{i:06d}" for i in range(n)],
        "age": age,
        "purchase_amount": purchase_amount,
        "purchase_count": purchase_count,
        "avg_check": avg_check,
        "days_since_last_purchase": days,
        "region": region,
        "device_type": device,
        "preferred_category": category,
        "is_premium": is_premium,
        "is_subscribed": is_subscribed,
        "consent_given": consent,
        "deletion_requested": deletion,
        "label": label,
    })
    return df


def _config() -> dict:
    return {
        "experiment": {"run_root": "runs_tmp", "random_seed": 1, "test_size": 0.25},
        "sharding": {"n_shards": 3, "shard_seed": 1},
        "model": {
            "backend": "logreg",
            "logreg": {"lr": 0.1, "weight_decay": 1e-4, "epochs": 50,
                       "batch_size": 0, "device": "cpu", "patience": 10,
                       "tol": 1e-4, "class_weight": "balanced"},
            "random_forest": {"n_estimators": 20, "max_depth": 6,
                              "min_samples_leaf": 2, "n_jobs": 1,
                              "class_weight": "balanced"},
        },
        "preprocessing": {
            "target_column": "label",
            "numeric_features": ["age", "purchase_amount", "purchase_count",
                                 "avg_check", "days_since_last_purchase"],
            "boolean_features": ["is_premium", "is_subscribed", "consent_given"],
            "categorical_features": ["region", "device_type", "preferred_category"],
        },
        "logging": {"level": "WARNING", "save_to_file": False},
    }


def test_assign_shards_is_deterministic_and_balanced():
    ids = [f"U-{i:06d}" for i in range(1000)]
    a = assign_shards(ids, n_shards=5, seed=42)
    b = assign_shards(ids, n_shards=5, seed=42)
    assert np.array_equal(a, b)
    counts = np.bincount(a, minlength=5)
    # allow 25% imbalance for N=1000 / 5
    assert counts.min() >= 150
    assert counts.max() <= 300


def test_shard_plan_rebuild_after_deletion_marks_affected():
    ids = [f"U-{i}" for i in range(20)]
    plan = ShardPlan.build(ids, n_shards=4, seed=0)
    deleted = [3, 4, 5]
    new_plan, affected = plan.rebuild_after_deletion(deleted)
    # Affected shards contain the originals' shard ids of those rows
    expected = {int(plan.shard_of_row[r]) for r in deleted}
    assert affected == expected
    assert new_plan.shard_of_row.shape[0] == len(ids) - len(deleted)


def test_end_to_end_train_and_unlearn(tmp_path):
    df = _make_tiny_dataset()
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    cfg = _config()
    cfg["experiment"]["run_root"] = str(tmp_path / "runs")
    pipe = ExperimentPipeline(config=cfg).load_dataset(data_path).fit()
    baseline = pipe.evaluate()
    # The task is quasi-deterministic (label = deletion | ~consent), so a
    # reasonable ensemble should do *much* better than chance.
    assert baseline.auc > 0.80

    # Choose 10 user ids that exist in the TRAINING split (not test)
    victims = pipe.df_train_["user_id"].sample(n=10, random_state=0).tolist()
    result = pipe.unlearn(UnlearnRequest(user_ids=victims))

    assert result.deleted_rows == 10
    assert 1 <= len(result.affected_shards) <= 3
    # delta values must equal the explicit after-before diff
    assert abs(
        result.delta["accuracy"]
        - (result.metrics_after.accuracy - result.metrics_before.accuracy)
    ) < 1e-9


def test_metrics_match_independent_computation():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2])
    rep = compute_classification_metrics(y_true, y_score)
    # hand-computed: threshold 0.5 -> preds [0,0,0,1,1,0]
    # tp=2, tn=3, fp=0, fn=1
    assert rep.accuracy == pytest.approx(5 / 6)
    assert rep.precision == pytest.approx(1.0)
    assert rep.recall == pytest.approx(2 / 3)
    assert rep.f1 == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))
    # AUC = P(pos > neg) for this data: pairs where pos>neg: (.35,.1) (.35,.2)
    # (.8,.1)(.8,.4)(.8,.2) (.9,.1)(.9,.4)(.9,.2) = 8 of 9 pairs -> 8/9
    assert rep.auc == pytest.approx(8 / 9)
