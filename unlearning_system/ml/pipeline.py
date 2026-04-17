"""High-level façade that stitches preprocessing + sharding + SISA together.

Exists so the CLI (and tests) have a one-call entry point::

    pipe = ExperimentPipeline.from_config("configs/default.yaml")
    pipe.load_dataset("data/synthetic_dataset.csv")
    pipe.fit()
    metrics = pipe.evaluate()
    result  = pipe.unlearn(UnlearnRequest(user_ids=["USR-000001"]))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .base import BaseLearner
from .logreg import LogRegConfig, TorchLogReg
from .metrics import ClassificationReport, compute_classification_metrics
from .preprocess import FeaturePipeline
from .rf import RFConfig, RFLearner
from .shards import ShardPlan
from .sisa import SISAEnsemble
from .unlearning import UnlearnRequest, UnlearnResult, Unlearner

logger = logging.getLogger(__name__)


@dataclass
class ExperimentPipeline:
    config: dict = field(default_factory=dict)
    # runtime state
    df_train_: pd.DataFrame | None = None
    df_test_: pd.DataFrame | None = None
    X_train_: np.ndarray | None = None
    y_train_: np.ndarray | None = None
    X_test_: np.ndarray | None = None
    y_test_: np.ndarray | None = None
    pipeline_: FeaturePipeline | None = None
    plan_: ShardPlan | None = None
    ensemble_: SISAEnsemble | None = None

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_config(cls, path: str | Path) -> "ExperimentPipeline":
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config=config)

    # ---- data ---------------------------------------------------------------

    def load_dataset(self, path: str | Path) -> "ExperimentPipeline":
        path = Path(path)
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        seed = int(self.config["experiment"]["random_seed"])
        test_size = float(self.config["experiment"]["test_size"])
        target = self.config["preprocessing"]["target_column"]

        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=seed,
            stratify=df[target] if target in df.columns else None,
        )
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        pp_cfg = self.config["preprocessing"]
        self.pipeline_ = FeaturePipeline(
            numeric=tuple(pp_cfg["numeric_features"]),
            categorical=tuple(pp_cfg["categorical_features"]),
            boolean=tuple(pp_cfg["boolean_features"]),
            target=target,
        )
        self.X_train_ = self.pipeline_.fit_transform(df_train)
        self.X_test_ = self.pipeline_.transform(df_test)
        self.y_train_ = self.pipeline_.extract_target(df_train)
        self.y_test_ = self.pipeline_.extract_target(df_test)
        self.df_train_ = df_train
        self.df_test_ = df_test

        n_shards = int(self.config["sharding"]["n_shards"])
        shard_seed = int(self.config["sharding"]["shard_seed"])
        if "user_id" in df_train.columns:
            keys = df_train["user_id"].astype(str).tolist()
        else:
            keys = [str(i) for i in range(len(df_train))]
        self.plan_ = ShardPlan.build(keys, n_shards=n_shards, seed=shard_seed)

        logger.info(
            "Loaded %d train / %d test rows, %d features, %d shards (sizes=%s)",
            len(df_train), len(df_test), self.X_train_.shape[1],
            self.plan_.n_shards, self.plan_.sizes(),
        )
        return self

    # ---- model factory ------------------------------------------------------

    def _learner_factory(self):
        backend = self.config["model"]["backend"]
        seed = int(self.config["experiment"]["random_seed"])
        if backend == "logreg":
            cfg_dict = dict(self.config["model"]["logreg"])
            def make(shard_id: int) -> BaseLearner:
                cfg = LogRegConfig(**cfg_dict, seed=seed + shard_id)
                return TorchLogReg(cfg, n_features=self.X_train_.shape[1])
            return make
        if backend == "random_forest":
            cfg_dict = dict(self.config["model"]["random_forest"])
            def make(shard_id: int) -> BaseLearner:
                cfg = RFConfig(**cfg_dict, seed=seed + shard_id)
                return RFLearner(cfg)
            return make
        raise ValueError(f"Unknown backend: {backend}")

    # ---- training / evaluation / unlearning --------------------------------

    def fit(self) -> "ExperimentPipeline":
        if self.X_train_ is None:
            raise RuntimeError("Call load_dataset() first.")
        self.ensemble_ = SISAEnsemble(self._learner_factory(), self.plan_)
        self.ensemble_.fit(self.X_train_, self.y_train_)
        return self

    def evaluate(self) -> ClassificationReport:
        if self.ensemble_ is None:
            raise RuntimeError("Call fit() first.")
        y_score = self.ensemble_.predict_proba(self.X_test_)
        return compute_classification_metrics(self.y_test_, y_score)

    def unlearn(self, request: UnlearnRequest) -> UnlearnResult:
        if self.ensemble_ is None:
            raise RuntimeError("Call fit() first.")
        unlearner = Unlearner(self.ensemble_)
        result, df_new, X_new, y_new = unlearner.unlearn(
            request, self.df_train_, self.X_train_, self.y_train_,
            self.X_test_, self.y_test_,
        )
        self.df_train_ = df_new
        self.X_train_ = X_new
        self.y_train_ = y_new
        # Keep the top-level reference in sync with the ensemble's updated
        # plan — otherwise downstream code that reads ``self.plan_`` (e.g.
        # the CLI's registry save) would persist the stale pre-deletion
        # indices.
        self.plan_ = self.ensemble_.plan
        return result
