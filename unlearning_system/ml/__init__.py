"""Machine Unlearning toolkit for the HSE MIEM Digital Forgetting project.

This package implements a SISA-style sharded ensemble that can quickly
"forget" individual training records in compliance with 152-ФЗ / GDPR.

Submodules
----------
preprocess : numeric/categorical feature pipeline (pandas -> numpy / tensors)
shards     : deterministic dataset sharding and index bookkeeping
base       : abstract ``BaseLearner`` interface shared by every model
logreg     : PyTorch logistic regression (gradient-based)
rf         : scikit-learn RandomForest wrapper
sisa       : ensemble container that aggregates shard predictions
unlearning : delete-by-ID / delete-by-filter + affected-shard retraining
metrics    : classification / regression / unlearning-specific metrics
registry   : serialisable run manifest (seeds, configs, artefacts)
cli        : command-line entry point
"""

from .base import BaseLearner
from .sisa import SISAEnsemble
from .unlearning import Unlearner

__all__ = ["BaseLearner", "SISAEnsemble", "Unlearner"]
