%%writefile gram_recursive.py
"""
gram_recursive.py

Recursive wrappers and utilities built on top of the base GRAM class.

Includes:
  - GRAMTime:        temporal smoothing / GRAM-RNN style recursion
  - GRAMCoarseFine:  hierarchical coarse→fine GRAM
  - GRAMFeatureSelector: masked / subspace GRAM
  - GRAMFrontierEstimator: frontier estimation + frontier-aware selection
"""

from typing import Callable, List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# Adjust this import to my actual filename / class name
try:
    # if kept HOSAM as the class name:
    from hosam import HOSAM as GRAM
except ImportError:
    # if renamed to GRAM directly:
    from hosam import GRAM  # type: ignore


# ----------------------------------------------------------------------
# GRAMTime: temporal recursion (GRAM-RNN style)
# ----------------------------------------------------------------------

class GRAMTime:
    """
    Temporal wrapper around a base GRAM instance.

    Idea:
      - For a time series of observations (rows in df_seq),
        I recursively use previous GRAM matches as context
        when encoding the next query.

      - encode_with_context(row, context) should produce bits that
        reflect both the current observation and some summary of
        the past GRAM state (previous matched pattern index
        or features derived from it).

    This is a skeleton; I will tailor encode_with_context and
    how context is updated to the specific application (P-series).
    """

    def __init__(
        self,
        gram: GRAM,
        encode_row: Callable[[pd.Series], np.ndarray],
        encode_with_context: Optional[
            Callable[[pd.Series, Optional[Any]], np.ndarray]
        ] = None,
    ):
        self.gram = gram
        self.encode_row = encode_row
        self.encode_with_context = encode_with_context or self._default_encode_with_ctx

    def _default_encode_with_ctx(
        self, row: pd.Series, context: Optional[Any]
    ) -> np.ndarray:
        """
        Fallback: ignore context and just encode the current row.
        can override this with a context-aware encoding.
        """
        return self.encode_row(row)

    def smooth_sequence(
        self,
        df_seq: pd.DataFrame,
        initial_context: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run GRAM across a temporal sequence, feeding context forward.

        Returns a list of dicts:
        [{'decision':..., 'index':..., 'meta':..., 'context':...}, ...]
        """
        context = initial_context
        outputs: List[Dict[str, Any]] = []

        for _, row in df_seq.iterrows():
            q_bits = self.encode_with_context(row, context)
            dec, idx, meta = self.gram.query(q_bits)

            # can choose how to update context; here uses idx directly
            if idx is not None:
                context = idx

            outputs.append({
                "decision": dec,
                "index": idx,
                "meta": meta,
                "context": context,
            })

        return outputs


# ----------------------------------------------------------------------
# GRAMCoarseFine: coarse→fine hierarchical GRAM
# ----------------------------------------------------------------------

class GRAMCoarseFine:
    r"""
    GRAMCoarseFine
    ==============

    Two-level GRAM hierarchy for multi-scale manifolds.

    Motivation
    ----------
    Many systems exhibit multi-scale structure:

        M_coarse ⊂ M_fine,

    where coarse representations capture the large-scale geometry and
    fine representations capture detailed variability.

    GRAMCoarseFine factors retrieval into:

      1. A coarse GRAM operating on a compressed / aggregated representation
      2. A fine GRAM operating on the full-resolution patterns

    Attributes
    ----------
    coarse_gram : GRAM or None
    fine_gram   : GRAM or None
    gram_ctor   : callable
        Constructor for GRAM/HOSAM (GRAM_phiFull).
    gram_kwargs_coarse : dict
        Extra kwargs passed to gram_ctor for the coarse GRAM.
    gram_kwargs_fine   : dict
        Extra kwargs passed to gram_ctor for the fine GRAM.
    """

    def __init__(
        self,
        coarse_gram: Optional[GRAM] = None,
        fine_gram: Optional[GRAM] = None,
        gram_ctor: Callable[..., GRAM] = GRAM,
        gram_kwargs_coarse: Optional[Dict[str, Any]] = None,
        gram_kwargs_fine: Optional[Dict[str, Any]] = None,
    ):
        self.coarse_gram = coarse_gram
        self.fine_gram = fine_gram
        self.gram_ctor = gram_ctor
        self.gram_kwargs_coarse = gram_kwargs_coarse or {}
        self.gram_kwargs_fine = gram_kwargs_fine or {}

    def encode_coarse(self, X_bits: np.ndarray) -> np.ndarray:
        """
        Map full-resolution patterns [P,N] to coarse patterns [P,Nc].

        Default: identity. Override or monkey-patch this to aggregate features,
        drop bits, or project onto a lower-dimensional code.

        In my script it patches this to use the first half of bits as "coarse".
        """
        return X_bits.copy()

    def fit(self, patterns_bits: np.ndarray) -> None:
        """
        Train coarse and fine GRAMs from the same full-resolution design bank.

        patterns_bits : array [P,N] of {0,1}
        """
        X_coarse = self.encode_coarse(patterns_bits)
        self.coarse_gram = self.gram_ctor(X_coarse, **self.gram_kwargs_coarse)
        self.fine_gram   = self.gram_ctor(patterns_bits, **self.gram_kwargs_fine)

    def query(self, q_bits: np.ndarray) -> Tuple[str, Optional[int], Dict]:
        """
        Two-stage query:
          1. coarse GRAM on encoded coarse query
          2. fine GRAM on original query

        Returns (decision, index, meta) using the fine GRAM decision, and
        exposes both coarse and fine meta for analysis.
        """
        if self.coarse_gram is None or self.fine_gram is None:
            return "UNKNOWN", None, {"reason": "coarse_or_fine_not_fit"}

        # Stage 1: coarse
        q_coarse = self.encode_coarse(q_bits[None, :])[0]
        dec_c, idx_c, meta_c = self.coarse_gram.query(q_coarse)

        # Stage 2: fine
        dec_f, idx_f, meta_f = self.fine_gram.query(q_bits)

        meta = {
            "coarse": meta_c,
            "fine": meta_f,
            "coarse_decision": dec_c,
            "fine_decision": dec_f,
            "coarse_idx": idx_c,
            "fine_idx": idx_f,
        }
        # Right now I trust the fine GRAM decision.
        return dec_f, idx_f, meta


# ----------------------------------------------------------------------
# GRAMFeatureSelector: masked / subspace GRAM
# ----------------------------------------------------------------------

class GRAMFeatureSelector:
    """
    GRAMFeatureSelector: use GRAM over a selected subset of bits/features.

    Useful when:
      - Only some features are relevant to a given task.
      - I want to compare multiple subspace GRAMs.
      - I want to perform “attention over features” in a discrete,
        associative way.

    Parameters
    ----------
    mask : np.ndarray or None
        Boolean mask of length N specifying which bits to keep.
    gram : GRAM or None
        An existing GRAM instance; if None, call fit(...) to train.
    gram_ctor : callable
        Constructor for GRAM (e.g. GRAM_plain / GRAM_phiFull).
    gram_kwargs : dict
        Extra kwargs passed to gram_ctor on fit().
    """

    def __init__(
        self,
        mask: Optional[np.ndarray] = None,
        gram: Optional[GRAM] = None,
        gram_ctor: Callable[..., GRAM] = GRAM,
        gram_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.mask = None
        if mask is not None:
            self.set_mask(mask)
        else:
            self.mask = None

        self.gram = gram
        self.gram_ctor = gram_ctor
        self.gram_kwargs = gram_kwargs or {}

    def set_mask(self, mask: np.ndarray) -> None:
        """Set or update the feature mask."""
        self.mask = mask.astype(bool)

    def _apply_mask(self, X_bits: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return X_bits.copy()
        return X_bits[:, self.mask]

    def fit(self, patterns_bits: np.ndarray, **gram_kwargs) -> None:
        """
        Train GRAM on masked patterns.

        patterns_bits: [P, N] {0,1} bank
        """
        X_masked = self._apply_mask(patterns_bits)
        kwargs = {**self.gram_kwargs, **gram_kwargs}
        self.gram = self.gram_ctor(X_masked, **kwargs)

    def query(self, q_bits: np.ndarray) -> Tuple[str, Optional[int], Dict]:
        """
        Query GRAM on masked version of q_bits.

        Returns (decision, index, meta) as in GRAM.query.
        """
        if self.gram is None:
            return "UNKNOWN", None, {"reason": "gram_not_fit"}

        if self.mask is None:
            q_masked = q_bits.copy()
        else:
            q_masked = q_bits[self.mask]

        return self.gram.query(q_masked)


# ----------------------------------------------------------------------
# GRAMFrontierEstimator: frontier estimation + frontier-aware selection
# ----------------------------------------------------------------------

class GRAMFrontierEstimator:
    r"""
    GRAMFrontierEstimator
    ======================

    Combine a fitted frontier model (such as a polynomial Cost–SSR) with a GRAM memory model.

    Parameters
    ----------
    gram : GRAM or None
        A GRAM/HOSAM instance. Attach later via attach_gram() if needed.
    cost_col : str
        Column name for cost in the DataFrame (like "lcc.y").
    ssr_col : str
        Column name for SSR in the DataFrame (like "ssr.y").

    Frontier Model
    --------------
    Suppose we have a Cost–SSR relationship:

        SSR ≈ f(Cost).

    We fit a polynomial in standardized Cost:

        z = (Cost - μ_cost) / σ_cost,
        f(z) = Σ a_k z^k.

    For each design i, define the frontier distance:

        d_front(i) = |SSR_i - f(Cost_i)|.
    """

    def __init__(
        self,
        gram: Optional[GRAM] = None,
        cost_col: str = "Cost",
        ssr_col: str = "SSR",
    ):
        self.gram = gram
        self.cost_col = cost_col
        self.ssr_col = ssr_col

        # frontier coefficients
        self.frontier_coeffs: Optional[np.ndarray] = None
        self.cost_mean: Optional[float] = None
        self.cost_std: Optional[float] = None

    # -----------------------------
    # Frontier fitting
    # -----------------------------

    def fit_frontier_poly(
        self,
        df: pd.DataFrame,
        degree: int = 3,
        mask_feasible: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit a polynomial frontier SSR ≈ f(Cost) from sweep data.
        """
        if mask_feasible is None:
            mask_feasible = np.ones(len(df), dtype=bool)
        sub = df.loc[mask_feasible, [self.cost_col, self.ssr_col]].dropna()
        x = sub[self.cost_col].values.astype(float)
        y = sub[self.ssr_col].values.astype(float)
        self.cost_mean = float(x.mean())
        self.cost_std = float(x.std() if x.std() > 0 else 1.0)
        z = (x - self.cost_mean) / self.cost_std
        self.frontier_coeffs = np.polyfit(z, y, degree)

    def frontier_predict(self, cost_vals: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitted frontier SSR = f(Cost).
        """
        if self.frontier_coeffs is None or self.cost_mean is None or self.cost_std is None:
            raise RuntimeError("Frontier model not fitted yet.")
        z = (np.asarray(cost_vals) - self.cost_mean) / self.cost_std
        return np.polyval(self.frontier_coeffs, z)

    def frontier_distance_for_bank(
        self,
        df: pd.DataFrame,
        cost_col: Optional[str] = None,
        ssr_col: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute |SSR - f(Cost)| for each design in df.
        """
        cc = cost_col or self.cost_col
        sc = ssr_col or self.ssr_col
        x = df[cc].values.astype(float)
        y = df[sc].values.astype(float)
        y_hat = self.frontier_predict(x)
        return np.abs(y - y_hat)

    def attach_gram(self, gram: GRAM) -> None:
        """
        Attach a GRAM instance to be used for frontier-aware selection.
        """
        self.gram = gram

    def frontier_aware_query(
        self,
        q_bits: np.ndarray,
        bank_df: pd.DataFrame,
        top_k: int = 5,
    ) -> pd.DataFrame:
        r"""
        Frontier-aware GRAM retrieval.

        For query q:

          1. Call GRAM(q) to identify manifold-consistent candidates.
          2. Rank candidate designs by distance to the fitted frontier.
          3. Return the top-k nearest-to-frontier designs.

        Currently this skeleton uses the full bank_df as the candidate set.
        I can refine this by restricting to GRAM's candidate list if desired.
        """
        if self.gram is None:
            raise RuntimeError("No GRAM instance attached.")

        # Call GRAM, but I don't yet use meta['candidates'] here.
        decision, idx, meta = self.gram.query(q_bits)

        cand_indices = list(range(len(bank_df)))
        sub = bank_df.iloc[cand_indices].copy()
        d_front = self.frontier_distance_for_bank(sub)
        sub["d_front"] = d_front
        sub = sub.sort_values(by="d_front", ascending=True)
        return sub.head(top_k)
