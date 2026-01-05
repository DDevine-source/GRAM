"""
GRAM: Geometric Regularized Associative Memory

Core package exports for GRAM models, variants, and recursive wrappers.
Analysis and benchmarking scripts are intentionally not imported here.
"""

# Core model
from .gram import GRAM

# Variants
from .gram_variants import (
    GRAM_plain,
    GRAM_phiJ,
    GRAM_phiFull,
)

# Recursive / hierarchical wrappers
from .gram_recursive import (
    GRAMFeatureSelector,
    GRAMCoarseFine,
)

__all__ = [
    "GRAM",
    "GRAM_plain",
    "GRAM_phiJ",
    "GRAM_phiFull",
    "GRAMFeatureSelector",
    "GRAMCoarseFine",
]
