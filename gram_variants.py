%%writefile gram_variants.py
"""
gram_variants.py

Convenience constructors for GRAM (Gravitationally Regularized Associative Memory)
built on top of the flagship HOSAM implementation in hosam.py.

Variants:
  - GRAM_plain   : baseline GRAM (uniform multi-index, no phi-specific tweaks)
  - GRAM_phiJ    : GRAM with jammed-PMF neighbors & phi-guided dither, but linear tau'
  - GRAM_phiFull : full phi-structured variant (jammed PMF + phi tau' + phi thresholds)

These are thin wrappers around hosam.HOSAM, imported as GRAM.
"""

from typing import Any, Dict
import numpy as np

from hosam import HOSAM as GRAM


def GRAM_plain(
    patterns_bits01: np.ndarray,
    **kwargs: Dict[str, Any],
) -> GRAM:
    """
    Baseline GRAM variant.

    Features:
      - pmf_mode="uniform"     : original multi-index 1-bit perturbations
      - guided_mode="plain"    : fixed-K guided bitflips (no phi schedule)
      - tau_mode="linear"      : tau'(d) = alpha * delta_bar * d
      - use_phi_thresholds=False: thresholds from quantiles

    All other arguments are passed through to HOSAM/GRAM.
    """
    return GRAM(
        patterns_bits01,
        pmf_mode=kwargs.pop("pmf_mode", "uniform"),
        guided_mode=kwargs.pop("guided_mode", "plain"),
        tau_mode=kwargs.pop("tau_mode", "linear"),
        use_phi_thresholds=kwargs.pop("use_phi_thresholds", False),
        **kwargs,
    )


def GRAM_phiJ(
    patterns_bits01: np.ndarray,
    **kwargs: Dict[str, Any],
) -> GRAM:
    """
    GRAM_phiJ : "phi+jammed" variant.

    Features:
      - pmf_mode="jammed"      : jammed-PMF multi-index sampling
      - guided_mode="phi"      : phi-annealed guided dither (bit flips)
      - tau_mode="linear"      : tau'(d) still linear in d
      - use_phi_thresholds=False

    This is the variant that incorporates jammed PMF + phi scheduling in the
    dynamics, but keeps the original tau' and thresholds for easier comparison.
    """
    return GRAM(
        patterns_bits01,
        pmf_mode=kwargs.pop("pmf_mode", "jammed"),
        guided_mode=kwargs.pop("guided_mode", "phi"),
        tau_mode=kwargs.pop("tau_mode", "linear"),
        use_phi_thresholds=kwargs.pop("use_phi_thresholds", False),
        **kwargs,
    )


def GRAM_phiFull(
    patterns_bits01: np.ndarray,
    **kwargs: Dict[str, Any],
) -> GRAM:
    """
    GRAM_phiFull : fully structured phi-based variant.

    Features:
      - pmf_mode="jammed"       : jammed-PMF multi-index sampling
      - guided_mode="phi"       : phi-annealed guided dither in NEAR/pre-denoise
      - tau_mode="phi"          : phi-annealed tau'(d) schedule
      - use_phi_thresholds=True : phi-structured (theta_accept, theta_near) window

    This matches the "GRAM_phi_full" variant described in my benchmark notes:
    gravitationally regularized, phi-annealed, and jammed-PMF neighborhood search.
    """
    return GRAM(
        patterns_bits01,
        pmf_mode=kwargs.pop("pmf_mode", "jammed"),
        guided_mode=kwargs.pop("guided_mode", "phi"),
        tau_mode=kwargs.pop("tau_mode", "phi"),
        use_phi_thresholds=kwargs.pop("use_phi_thresholds", True),
        **kwargs,
    )


def GRAM_from_mode(
    mode: str,
    patterns_bits01: np.ndarray,
    **kwargs: Dict[str, Any],
) -> GRAM:
    """
    Helper to get a GRAM variant from a simple string name.

    mode in {"plain", "phiJ", "phiFull"}.
    """
    mode = mode.lower()
    if mode == "plain":
        return GRAM_plain(patterns_bits01, **kwargs)
    elif mode in ("phij", "phi_j", "phi-j"):
        return GRAM_phiJ(patterns_bits01, **kwargs)
    elif mode in ("phifull", "phi_full", "phi-full"):
        return GRAM_phiFull(patterns_bits01, **kwargs)
    else:
        raise ValueError(f"Unknown GRAM mode: {mode}")
