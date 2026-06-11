"""Core synthetic data generator orchestrating all submodules."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .correlations import apply_correlations
from .distributions import FeatureGenerator, generate_feature
from .labeler import apply_labels
from .validator import validate_dataset

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    def __init__(self, config_path, n_samples=None, seed=None, labeling_mode=None):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        gen_config = self.config["generator"]
        self.n_samples = n_samples or gen_config["n_samples"]
        self.seed = seed if seed is not None else gen_config["random_seed"]
        self.rng = np.random.default_rng(self.seed)

        self.reference_date = gen_config.get("reference_date", "2025-01-01")
        self.output_format = gen_config.get("output_format", "csv")
        self.output_path = Path(gen_config.get("output_path", "data/synthetic_dataset.csv"))

        if labeling_mode:
            self.config.setdefault("labeling", {})["mode"] = labeling_mode

        self._df: pd.DataFrame | None = None

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def generate(self):
        start_time = time.time()
        logger.info("Starting generation: n_samples=%d, seed=%d", self.n_samples, self.seed)
        logger.info("Config: %s", self.config_path)

        features_config = self.config["features"]
        data: dict[str, np.ndarray] = {}
        
        # Instantiate FeatureGenerator once to share generated state (like genders)
        self.feature_generator = FeatureGenerator(self.rng)

        distribution_features = {name: cfg for name, cfg in features_config.items() if "distribution" in cfg}
        for name, feat_config in distribution_features.items():
            data[name] = generate_feature(name, feat_config, self.n_samples, self.rng, reference_date=self.reference_date, gen=self.feature_generator)
            logger.debug("Generated feature: %s", name)

        type_features = {name: cfg for name, cfg in features_config.items() if "type" in cfg}
        id_numbers = np.arange(1, self.n_samples + 1)

        for name, feat_config in type_features.items():
            data[name] = generate_feature(
                name, feat_config, self.n_samples, self.rng,
                reference_date=self.reference_date,
                age_array=data.get("age"),
                id_array=id_numbers,
                gen=self.feature_generator
            )
            logger.debug("Generated feature: %s", name)

        column_order = list(features_config.keys())
        df = pd.DataFrame({col: data[col] for col in column_order if col in data})

        corr_config = self.config.get("correlations", {})
        if corr_config:
            df = apply_correlations(df, corr_config, self.rng)
            logger.info("Applied correlation structures")

        labeling_config = self.config.get("labeling", {"mode": "rule"})
        df = apply_labels(df, labeling_config)
        logger.info("Applied labeling (mode=%s)", labeling_config.get("mode"))

        if "age" in df.columns:
            df["age"] = df["age"].round().astype(int)
        if "purchase_count" in df.columns:
            df["purchase_count"] = df["purchase_count"].round().astype(int)
        if "days_since_last_purchase" in df.columns:
            df["days_since_last_purchase"] = df["days_since_last_purchase"].round().astype(int)
        if "avg_check" in df.columns:
            df["avg_check"] = df["avg_check"].round(2)
        if "purchase_amount" in df.columns:
            df["purchase_amount"] = df["purchase_amount"].round(2)

        self._df = df

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info("Generation completed in %.1f ms", elapsed_ms)

        target_col = labeling_config.get("target_column", "label")
        if target_col in df.columns:
            positive_rate = df[target_col].astype(bool).mean()
            logger.info("Class balance: %.2f%% label=1", positive_rate * 100)

        return df

    def validate(self, df=None):
        if df is None:
            df = self._df
        if df is None:
            raise ValueError("No dataset to validate. Run generate() first.")
        features_config = self.config["features"]
        corr_config = self.config.get("correlations", {})
        target_col = self.config.get("labeling", {}).get("target_column", "label")
        return validate_dataset(df, features_config, corr_config, target_col)

    def save(self, df=None):
        if df is None:
            df = self._df
        if df is None:
            raise ValueError("No dataset to save. Run generate() first.")

        output_path = (
            Path.cwd() / self.output_path if not self.output_path.is_absolute() else self.output_path
        )
        
        # Append timestamp to the filename to prevent overwriting
        timestamp = int(time.time())
        output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_format == "csv":
            df.to_csv(output_path, index=False)
        elif self.output_format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown output format: '{self.output_format}'")

        logger.info("Saved dataset to %s (%d rows)", output_path, len(df))
        return output_path

    def get_info(self, df=None):
        if df is None:
            df = self._df
        if df is None:
            raise ValueError("No dataset available. Run generate() first.")

        target_col = self.config.get("labeling", {}).get("target_column", "label")
        info: dict[str, Any] = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
            "numeric_summary": {},
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            info["numeric_summary"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        if target_col in df.columns:
            positive_rate = df[target_col].astype(bool).mean()
            info["class_balance"] = {
                "target_column": target_col,
                "positive_rate": float(positive_rate),
                "count_true": int(df[target_col].astype(bool).sum()),
                "count_false": int((~df[target_col].astype(bool)).sum()),
            }
        return info
