"""Synthetic data generator for e-commerce domain.

Part of the "Digital Oblivion Mechanisms" research project (HSE MIEM).
Generates tabular datasets with configurable distributions, correlations,
and labeling for machine unlearning experiments.
"""

from .core import SyntheticDataGenerator

__all__ = ["SyntheticDataGenerator"]
