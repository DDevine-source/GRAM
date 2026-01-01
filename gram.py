%%writefile hosam.py
# GRAM.py
#
# Hybrid One-Shot Associative Memory (HOSAM) / GRAM core
# ------------------------------------------------------
# Features:
#   - Spectral diagnostics (SPR, gamma)
#   - gamma_factor scaling for annealing strength
#   - Feigenbaum/golden-ratio K schedule for guided bit flips
#   - Adaptive NEAR-zone annealing via adaptive_annealing_params()
#   - Optional phi-based tau' (tau_mode="phi") and thresholds (use_phi_thresholds=True)
#   - Jammed-PMF multi-index candidate generation
#   - FAR / NEAR / IN zones with multichannel certification
#
# Author: Derek Devine (foundational idea + equations)
# This file integrates the GID / SR + Feigenbaum refinements.

import math
import time
import numpy as np
from typing import Tuple, List, Dict, Optional

# ============================================================
# Feigenbaum / golden-ratio schedule + phi-annealed tau'
# ============================================================

PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI

# Feigenbaum delta (δ) for logistic-map period-doubling
FEIGENBAUM_DELTA = 4.66920160910299


def feigenbaum_cooling(t: int, T0: float = 1.0) -> float:
    """
    Nonlinear Feigenbaum-inspired cooling schedule.

    T(t) = T0 / (1 + δ * (t - 1))

    - t >= 1
    - δ ~ 4.669 is Feigenbaum's constant.
    - Using this as a gentle shrink factor on K_t in the NEAR-zone guided dither.
    """
    t = max(1, int(t))
    return T0 / (1.0 + FEIGENBAUM_DELTA * float(t - 1))


def phi_K_schedule(K0: int, t: int) -> int:
    """
    Feigenbaum/golden-ratio K schedule.
    K_t = max(1, round(K0 * phi^{-(t-1)})).
    """
    if t <= 0:
        return K0
    val = K0 * (PHI_INV ** (t - 1))
    return max(1, int(round(val)))


def tau_prime_phi_schedule(t: int, delta_bar: float, alpha: float = 1.0) -> float:
    """
    Phi-annealed tau'(t): grows with t via geometric sum of phi^{-k}, k = 1..t.

    tau'(t) = alpha * delta_bar * sum_{k=1}^t phi^{-k}
    """
    t = max(t, 0)
    if t == 0:
        return 0.0
    # geometric sum of phi^{-k}, k=1..t
    geom = PHI_INV * (1.0 - (PHI_INV ** t)) / (1.0 - PHI_INV)
    return alpha * delta_bar * geom


# ============================================================
# Jammed PMF for multi-index perturbations (CA saturation)
# ============================================================

JAMMED_PMF_BASE = np.array([
    0.1137521810,  # k=0
    0.3354531030,  # k=1
    0.1677266052,  # k=2
    0.1115381940,  # k=3
    0.0747061270,  # k=4
    0.0596268081,  # k=5
    0.0496965545,  # k=6
    0.0479211684,  # k=7
    0.0419316513,  # k=8
], dtype=np.float64)
JAMMED_PMF_BASE /= JAMMED_PMF_BASE.sum()


def sample_jammed_k(block_len: int, rng: np.random.Generator) -> int:
    """
    Sample a number of flips k from the jammed PMF, truncated at block_len.
    Returns k in [0, block_len].
    """
    max_k = min(block_len, JAMMED_PMF_BASE.size - 1)
    if max_k <= 0:
        return 0
    pmf = JAMMED_PMF_BASE[:max_k + 1].copy()
    pmf /= pmf.sum()
    ks = np.arange(max_k + 1)
    return int(rng.choice(ks, p=pmf))


def _perturb_jammed(win: np.ndarray, multiprobe: int,
                    rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """
    Generate up to `multiprobe` jammed-PMF-based perturbations of window bits.
    For each perturbation:
      - sample k from jammed PMF
      - flip k random positions in the block
    """
    if rng is None:
        rng = np.random.default_rng()
    out: List[np.ndarray] = []
    L = win.size
    for _ in range(multiprobe):
        k = sample_jammed_k(L, rng)
        if k <= 0:
            continue
        idx = rng.choice(L, size=k, replace=False)
        w = win.copy()
        w[idx] ^= 1
        out.append(w)
    return out


# ============================================================
# Adaptive annealing parameters for NEAR-zone
# ============================================================

def adaptive_annealing_params(
    p_noise: float,
    margin_neg_frac: float,
    gamma_factor: float,
    K_max: int = 8,
    steps_max: int = 8,
) -> Tuple[int, int]:
    """
    Decide (K0, steps) for NEAR-zone annealing based on:

      - p_noise: effective noise level (0..1), e.g. d1 / N
      - margin_neg_frac: fraction of negative margins at entry
      - gamma_factor: under/over-constrained factor from SPR/gamma
                       (>1 underconstrained / chaotic, <1 over-steep)

    Theory mapping:
      - BAM GID/SR: margin_neg_frac is your local "instability" mi < 0 fraction.
      - SOM: optimal annealing occurs near 1/ϕ, so I bias toward moderate steps
             unless chaos is high.
      - UCEG: gamma_factor encodes how far the system is from the Feigenbaum/golden
              equilibrium γ_opt ≈ 2.58.
    """
    # clamp inputs
    p_noise = float(np.clip(p_noise, 0.0, 1.0))
    margin_neg_frac = float(np.clip(margin_neg_frac, 0.0, 1.0))

    # Effective "chaos" measure: blend distance noise + margin instability
    #   - if either d1/N or margin_neg_frac is high, chaos ~ 1
    chaos = 0.5 * p_noise + 0.5 * margin_neg_frac

    # Under/over-constrained weight:
    #   gamma_factor > 1  => underconstrained (BAM N>>P), allow more steps
    #   gamma_factor < 1  => over-steep basins, fewer steps
    gamma_weight = float(np.clip(gamma_factor, 0.5, 2.0))

    # Base "SOM-like" annealing level around 1/ϕ ≈ 0.618:
    #   when chaos ~0, I want ~few steps and small K0.
    #   when chaos ~1, I let chaos/gamma push us toward the caps.
    base_steps = 2.0  # minimum nontrivial annealing loop
    base_K0 = 2.0

    # Scale with chaos & gamma:
    #   - steps grows with chaos * gamma_weight
    #   - K0 grows with chaos but slightly less sensitive than steps
    steps = base_steps + 4.0 * chaos * gamma_weight
    K0 = base_K0 + 3.0 * chaos

    # Clamp to reasonable maxima
    steps = int(max(1, min(steps_max, round(steps))))
    K0 = int(max(1, min(K_max, round(K0))))

    return K0, steps


# ============================================================
# Optional accelerators (Numba)
# ============================================================

USE_NUMBA = True

try:
    if USE_NUMBA:
        from numba import njit
    else:
        def njit(*args, **kwargs):
            def wrap(f):
                return f
            return wrap
except Exception:
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap
    USE_NUMBA = False


# ============================================================
# Utils: bit-parallel ops, Hamming, packing
# ============================================================

def to_pm1(bits01: np.ndarray) -> np.ndarray:
    """{0,1} -> {-1,+1}"""
    return np.where(bits01 > 0, 1, -1).astype(np.int8)


def to_bits01(pm1: np.ndarray) -> np.ndarray:
    """{-1,+1} -> {0,1}"""
    return (pm1 > 0).astype(np.uint8)


def pack_uint64_rows(bits01: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Pack {0,1} matrix [P,N] into uint64 blocks [P,nblocks].
    Returns (blocks, nblocks, padbits).
    """
    P, N = bits01.shape
    nblocks = (N + 63) // 64
    padbits = nblocks * 64 - N
    out = np.zeros((P, nblocks), dtype=np.uint64)
    for p in range(P):
        for b in range(nblocks):
            start = b * 64
            end = min((b + 1) * 64, N)
            word = 0
            for k, bit in enumerate(bits01[p, start:end]):
                word |= (int(bit) & 1) << k
            out[p, b] = np.uint64(word)
    return out, nblocks, padbits


@njit(fastmath=True)
def _popcount_u64(x: np.uint64) -> int:
    c = 0
    v = x
    while v:
        v &= v - np.uint64(1)
        c += 1
    return c


@njit(fastmath=True)
def hamming_distance_blocks(query: np.ndarray, X_blocks: np.ndarray) -> np.ndarray:
    """query: (nblocks,), X_blocks: (P,nblocks) -> dists: (P,)"""
    P, nb = X_blocks.shape
    out = np.empty(P, dtype=np.int32)
    for i in range(P):
        s = 0
        for b in range(nb):
            s += _popcount_u64(query[b] ^ X_blocks[i, b])
        out[i] = s
    return out


def hamming01(a_bits: np.ndarray, B_bits: np.ndarray) -> np.ndarray:
    """Slow reference: Hamming distance of a vs. rows in B_bits."""
    return np.sum(np.abs(B_bits - a_bits), axis=1).astype(np.int32)


# ============================================================
# Hebbian training and S' scoring
# ============================================================

def train_hebbian_W(patterns_pm1: np.ndarray) -> np.ndarray:
    """
    patterns_pm1 in {-1,+1}^{P x N}; return W in R^{N x N} with diag=0.
    """
    W = patterns_pm1.T @ patterns_pm1
    np.fill_diagonal(W, 0)
    return W.astype(np.float64)


@njit(fastmath=True)
def signed_score_Sprime(v_pm1: np.ndarray, W: np.ndarray) -> float:
    """Compute 0.5 * v^T W v, W symmetric, diag 0."""
    N = v_pm1.shape[0]
    z = 0.0
    for i in range(N):
        s = 0.0
        vi = v_pm1[i]
        row = W[i]
        for j in range(N):
            s += row[j] * v_pm1[j]
        z += vi * s
    return 0.5 * z


def batch_Sprime(V_pm1: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Vectorized S' for batches: returns [B]."""
    return 0.5 * np.einsum('bi,ij,bj->b', V_pm1, W, V_pm1, optimize=True)


# ============================================================
# Multi-index Hamming search (candidate generation)
# ============================================================

def multi_index_build(patterns_bits: np.ndarray, B: int = 8):
    """
    Build B block-wise hash tables for fast candidate retrieval.
    Returns (tables, (B, block_len, N)).
    """
    P, N = patterns_bits.shape
    block_len = (N + B - 1) // B
    tables = []
    for b in range(B):
        start = b * block_len
        end = min(b * block_len + block_len, N)
        keys: Dict[bytes, List[int]] = {}
        for idx in range(P):
            key = patterns_bits[idx, start:end].tobytes()
            if key in keys:
                keys[key].append(idx)
            else:
                keys[key] = [idx]
        tables.append((start, end, keys))
    return tables, (B, block_len, N)


def _perturb_1bit(win: np.ndarray) -> List[np.ndarray]:
    """Generate all 1-bit flips within window."""
    out: List[np.ndarray] = []
    for i in range(win.size):
        w = win.copy()
        w[i] ^= 1
        out.append(w)
    return out


def multi_index_candidates(
    query_bits: np.ndarray,
    tables,
    blocks_info,
    multiprobe: int = 2,
    K: int = 8,
    budget: int = 100,
    pmf_mode: str = "uniform",
) -> List[int]:
    """
    Probe block hashes around the query (exact + perturbations per block).

    pmf_mode:
      - "uniform": original behaviour (all 1-bit flips)
      - "jammed" : jammed-PMF-based k-bit flips per block

    Merge candidates up to budget and return unique candidate indices.
    """
    B, block_len, N = blocks_info
    cands: List[int] = []
    seen = set()
    rng = np.random.default_rng()

    for (start, end, keys) in tables:
        # exact probe
        ek = query_bits[start:end].tobytes()
        if ek in keys:
            for idx in keys[ek]:
                if idx not in seen:
                    cands.append(idx)
                    seen.add(idx)
                    if len(cands) >= budget:
                        return cands

        if multiprobe > 0:
            win = query_bits[start:end]
            if pmf_mode == "jammed":
                perturbs = _perturb_jammed(win, multiprobe, rng=rng)
            else:
                perturbs = _perturb_1bit(win)[:multiprobe]
            for p in perturbs:
                pk = p.tobytes()
                if pk in keys:
                    for idx in keys[pk]:
                        if idx not in seen:
                            cands.append(idx)
                            seen.add(idx)
                            if len(cands) >= budget:
                                return cands
    return cands[:budget]


# ============================================================
# Multi-channel signatures & checks
# ============================================================

def ones_count(bits01: np.ndarray) -> int:
    return int(np.sum(bits01))


def pair_signature(bits01: np.ndarray, which: str = "11") -> np.ndarray:
    """
    Return a boolean upper-triangular signature vector for (1,1) or (0,0) pairs.
    Encoded as a compact boolean vector of length N*(N-1)/2.
    """
    N = bits01.size
    sig = []
    if which == "11":
        for i in range(N):
            bi = bits01[i]
            for j in range(i + 1, N):
                sig.append((bi & bits01[j]) == 1)
    elif which == "00":
        for i in range(N):
            bi = bits01[i]
            for j in range(i + 1, N):
                sig.append((bi | bits01[j]) == 0)
    else:
        raise ValueError("which must be '11' or '00'")
    return np.array(sig, dtype=np.bool_)


def minhash_pairs(bits01: np.ndarray, seeds: List[int]) -> np.ndarray:
    """
    Simple SimHash/MinHash-like sketch over bit-pairs:
    hash indices of equal pairs. Deterministic lightweight sketch.
    """
    N = bits01.size
    L = len(seeds)
    sketch = np.zeros(L, dtype=np.int32)
    for i in range(N):
        bi = bits01[i]
        for j in range(i + 1, N):
            bj = bits01[j]
            if bi == bj:
                idx = i * N + j
                h = 0
                for s in seeds:
                    x = (idx ^ s) & 0xFFFFFFFF
                    x ^= (x << 13) & 0xFFFFFFFF
                    x ^= (x >> 17)
                    x ^= (x << 5) & 0xFFFFFFFF
                    h ^= x
                sketch[h % L] += 1
    return sketch


def majority_of_channels(pass_flags: List[bool], need: int) -> bool:
    return sum(1 for f in pass_flags if f) >= need


# ============================================================
# Pre-denoisers: attention jump / bit-flip polish
# ============================================================

def attention_jump(
    v_pm1: np.ndarray,
    patterns_pm1: np.ndarray,
    beta: float = 10.0,
    topK: int = 5,
) -> np.ndarray:
    """
    One-step "continuous Hopfield / attention" jump; round back to {-1,+1}.
    """
    sims = (patterns_pm1 @ v_pm1) / v_pm1.size
    k = min(topK, sims.size)
    idx = np.argpartition(-sims, k - 1)[0:k]
    sims_top = sims[idx]
    w = np.exp(beta * sims_top - np.max(beta * sims_top))
    w = w / (np.sum(w) + 1e-12)
    v_new = np.sign((w[:, None] * patterns_pm1[idx]).sum(axis=0))
    v_new[v_new == 0] = 1
    return v_new.astype(np.int8)


def bitflip_polish(v_pm1: np.ndarray, W: np.ndarray, K: int = 2) -> np.ndarray:
    """
    One small gradient-like polish:
      flip K bits with smallest stability margin m_i = v_i * (Wv)_i.
    """
    g = W @ v_pm1
    margins = v_pm1 * g
    idx = np.argsort(margins)[:K]
    out = v_pm1.copy()
    out[idx] *= -1
    return out


# ============================================================
# kNN voting proposal
# ============================================================

def knn_vote(query_bits: np.ndarray, patterns_bits: np.ndarray, k: int = 3) -> np.ndarray:
    d = hamming01(query_bits, patterns_bits)
    idx = np.argpartition(d, min(k, d.size) - 1)[0:k]
    vote = np.sign(np.sum(to_pm1(patterns_bits[idx]), axis=0))
    vote[vote == 0] = 1
    return vote.astype(np.int8)


# ============================================================
# Adaptive tau'(t) estimation
# ============================================================

def estimate_delta_Sprime_per_flip(
    W: np.ndarray,
    patterns_pm1: np.ndarray,
    samples: int = 256,
) -> float:
    """
    Empirical average drop in S' per single random bit flip (over random patterns).
    """
    P, N = patterns_pm1.shape
    ss = []
    for _ in range(min(samples, P)):
        i = np.random.randint(P)
        x = patterns_pm1[i].copy()
        s0 = signed_score_Sprime(x, W)
        j = np.random.randint(N)
        x[j] *= -1
        s1 = signed_score_Sprime(x, W)
        ss.append(abs(s0 - s1))
    return float(np.mean(ss)) if ss else 1.0


def tau_prime_from_noise(t: int, delta_bar: float, alpha: float = 1.0) -> float:
    return alpha * delta_bar * max(t, 0)


# ============================================================
# Tiny LDPC-like heavy-noise wrapper (optional)
# ============================================================

def generate_sparse_parity_check(N: int, rows: int = 2, w: int = 3, seed: int = 0) -> np.ndarray:
    """
    Tiny sparse H for demo (not production ECC).
    Returns H in {0,1}^{M x N}.
    """
    rng = np.random.default_rng(seed)
    H = np.zeros((rows, N), dtype=np.uint8)
    for r in range(rows):
        cols = rng.choice(N, size=w, replace=False)
        H[r, cols] = 1
    return H


def ldpc_bitflip_decode(v_bits: np.ndarray, H: np.ndarray, iters: int = 2) -> np.ndarray:
    """
    Very small demo: parity-check H, K-step bitflip on worst offenders.
    """
    x = v_bits.copy()
    for _ in range(max(0, iters)):
        synd = (H @ x) & 1
        if np.all(synd == 0):
            break
        scores = (H.T @ synd)
        k = max(1, x.size // 100)
        idx = np.argpartition(-scores, k - 1)[0:k]
        x[idx] ^= 1
    return x


# ============================================================
# Three-zone decision, margins, threshold calibration
# ============================================================

def compute_margins(
    d_top1: int,
    d_top2: int,
    Spr1: float,
    Spr2: float,
) -> Tuple[int, float]:
    return (d_top2 - d_top1, Spr1 - Spr2)


def calibrate_thresholds(
    Sprime_self: np.ndarray,
    d_intra: np.ndarray,
    target_fpr: float = 0.001,
    use_phi_thresholds: bool = False,
) -> Dict[str, float]:
    """
    Threshold calibration.
    If use_phi_thresholds:
       theta_accept = median(d_intra)
       theta_near   = phi * theta_accept
    else:
       theta_accept = 95% quantile
       theta_near   = 99% quantile
    """
    out = {}
    if use_phi_thresholds:
        med_d = float(np.median(d_intra))
        theta_accept = med_d
        theta_near = med_d * PHI
    else:
        theta_accept = float(np.quantile(d_intra, 0.95))
        theta_near = float(np.quantile(d_intra, 0.99))

    out["theta_accept"] = theta_accept
    out["theta_near"] = theta_near
    out["tau_alpha"] = 1.0
    out["gamma_d"] = max(1.0, 0.5 * np.median(d_intra))
    out["gamma_S"] = 0.01 * np.median(Sprime_self) if np.median(Sprime_self) > 0 else 1.0
    return out


# ============================================================
# HOSAM core (original base code was named HOSAM before GRAM branding)
# ============================================================

class HOSAM:
    def __init__(
        self,
        patterns_bits01: np.ndarray,
        use_multi_index: bool = True,
        B_blocks: int = 8,
        multiprobe: int = 2,
        cand_budget: int = 64,
        pmf_mode: str = "uniform",         # "uniform" or "jammed"

        # verification configs
        use_pre_denoise: bool = True,
        pre_denoise_mode: str = "bitflip",  # "bitflip" or "attention"
        pre_denoise_K: int = 2,
        pre_denoise_beta: float = 10.0,
        guided_mode: str = "plain",         # "plain" or "phi"
        tau_mode: str = "linear",           # "linear" or "phi"
        use_phi_thresholds: bool = False,

        use_knn_vote: bool = True,
        knn_k: int = 3,

        # thresholds
        theta_accept: Optional[float] = None,
        theta_near: Optional[float] = None,
        gamma_d: Optional[float] = None,
        gamma_S: Optional[float] = None,
        tau_alpha: float = 1.0,

        # signatures
        use_multichannel: bool = True,
        need_channels: int = 3,
        seeds: List[int] = [0xA5A5A5A5, 0x5A5A5A5A],

        # ldpc heavy-noise
        use_ldpc: bool = False,
        ldpc_it: int = 2,
    ):
        """
        Initialize memory and verification structures.
        """
        self.patterns_bits = patterns_bits01.astype(np.uint8)
        self.P, self.N = self.patterns_bits.shape
        self.patterns_pm1 = to_pm1(self.patterns_bits)

        # Hebbian weight matrix / S'(self)
        self.W = train_hebbian_W(self.patterns_pm1)
        self.Sprime_self = batch_Sprime(self.patterns_pm1, self.W)

        # ---- Spectral diagnostics: singular values, SPR, gamma ----
        try:
            s = np.linalg.svd(self.W, compute_uv=False)
            self.singular_values = s
            P_eff = min(self.P, s.size - 1)
            if P_eff >= 1:
                num = float((s.sum()) ** 2)
                den = float((s ** 2).sum() + 1e-12)
                self.SPR = num / den
                self.gamma = float(s[P_eff - 1] / (s[P_eff] + 1e-12))
            else:
                self.SPR = 0.0
                self.gamma = 0.0
        except Exception:
            self.singular_values = None
            self.SPR = 0.0
            self.gamma = 0.0

        # Feigenbaum–golden equilibrium for gamma (value I calculated analytically)
        self.gamma_opt = 2.58

        # gamma_factor: >1 when underconstrained (gamma < gamma_opt),
        #               <1 when over-steep (gamma >> gamma_opt), in [0.5, 2.0]
        if self.gamma > 0:
            raw_factor = self.gamma_opt / self.gamma
            self.gamma_factor = float(np.clip(raw_factor, 0.5, 2.0))
        else:
            self.gamma_factor = 1.0

        # Defaults for NEAR-zone guided dither (base values, still subject to adaptive_annealing)
        self.base_K0_near = 5
        self.base_near_steps = 5

        # Multichannel signatures
        self.use_multichannel = use_multichannel
        if use_multichannel:
            self.sig11 = np.array(
                [pair_signature(self.patterns_bits[i], "11") for i in range(self.P)],
                dtype=np.bool_,
            )
            self.sig00 = np.array(
                [pair_signature(self.patterns_bits[i], "00") for i in range(self.P)],
                dtype=np.bool_,
            )
            self.sketch = np.array(
                [minhash_pairs(self.patterns_bits[i], seeds) for i in range(self.P)],
                dtype=np.int32,
            )
        else:
            self.sig11 = self.sig00 = self.sketch = None

        # Multi-index
        self.use_multi_index = use_multi_index
        self.pmf_mode = pmf_mode
        if self.use_multi_index:
            self.tables, self.blocks_info = multi_index_build(self.patterns_bits, B=B_blocks)
            self.multiprobe = multiprobe
            self.cand_budget = cand_budget
        else:
            self.tables = self.blocks_info = None

        # Packed for brute-force Hamming if needed
        self.blocks, self.nblocks, self.padbits = pack_uint64_rows(self.patterns_bits)

        # Intra distances for calibration
        d_intra = []
        for i in range(min(self.P, 256)):
            q = self.patterns_bits[i]
            if self.use_multi_index:
                cand = multi_index_candidates(
                    q, self.tables, self.blocks_info,
                    multiprobe=self.multiprobe,
                    K=8, budget=self.cand_budget,
                    pmf_mode=self.pmf_mode,
                )
                if i in cand:
                    cand.remove(i)
                if not cand:
                    cand = list(range(self.P))
                    cand.remove(i)
                d = np.min(hamming01(q, self.patterns_bits[cand]))
            else:
                d = np.min(hamming01(q, np.delete(self.patterns_bits, i, axis=0)))
            d_intra.append(d)
        d_intra = np.array(d_intra) if d_intra else np.array([0])

        # Calibration
        calib = calibrate_thresholds(
            self.Sprime_self,
            d_intra,
            use_phi_thresholds=use_phi_thresholds,
        )
        self.theta_accept = float(theta_accept if theta_accept is not None else calib["theta_accept"])
        self.theta_near = float(theta_near if theta_near is not None else calib["theta_near"])
        self.gamma_d = float(gamma_d if gamma_d is not None else calib["gamma_d"])
        self.gamma_S = float(gamma_S if (gamma_S is not None and not math.isnan(gamma_S))
                             else calib["gamma_S"])
        self.tau_alpha = float(tau_alpha)

        # Average drop per flip
        self.delta_bar = estimate_delta_Sprime_per_flip(
            self.W,
            self.patterns_pm1,
            samples=min(256, self.P),
        )

        # knobs
        self.use_pre_denoise = use_pre_denoise
        self.pre_denoise_mode = pre_denoise_mode
        self.pre_denoise_K = pre_denoise_K
        self.pre_denoise_beta = pre_denoise_beta
        self.guided_mode = guided_mode
        self.tau_mode = tau_mode

        self.use_knn_vote = use_knn_vote
        self.knn_k = knn_k
        self.need_channels = need_channels
        self.seeds = seeds
        self.use_ldpc = use_ldpc
        self.ldpc_it = ldpc_it
        self.H_ldpc = generate_sparse_parity_check(self.N, rows=2, w=3) if use_ldpc else None

        # RNG for golden jitter / NEAR-zone randomized kernel
        self.rng = np.random.default_rng()

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _nearest_candidates(self, q_bits: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Return candidate indices and their distances (if computed).
        """
        if self.use_multi_index:
            cand = multi_index_candidates(
                q_bits,
                self.tables,
                self.blocks_info,
                multiprobe=self.multiprobe,
                K=8,
                budget=self.cand_budget,
                pmf_mode=self.pmf_mode,
            )
            if not cand:
                cand = list(range(self.P))
            dists = hamming01(q_bits, self.patterns_bits[cand])
            return cand, dists
        else:
            q_blocks, _, _ = pack_uint64_rows(q_bits[None, :])
            dists = hamming_distance_blocks(q_blocks[0], self.blocks)
            cand = list(range(self.P))
            return cand, dists

    def _pre_denoise_bitflip(self, v_pm1: np.ndarray, steps: int = 2) -> np.ndarray:
        """
        Guided bit-flip denoise in {-1,+1} space.
        """
        v = v_pm1.copy()
        for t in range(1, steps + 1):
            if self.guided_mode == "phi":
                K_t = phi_K_schedule(self.pre_denoise_K, t)
            else:
                K_t = self.pre_denoise_K
            g = self.W @ v
            margins = v * g
            idx = np.argsort(margins)[:K_t]
            v[idx] *= -1
        return v

    def _guided_dither_bitflip(
        self,
        v_pm1: np.ndarray,
        d1: Optional[int] = None,
        steps: Optional[int] = None,
        K0: Optional[int] = None,
    ) -> np.ndarray:
        """
        NEAR-zone guided dither in {-1,+1} space (full φ patch).

        Additions vs the baseline:
          - φ-weighted chaos scaling of (K0, steps)
          - Feigenbaum cooling shrinking K_t across steps
          - golden-randomized perturbation kernel:
                * core: lowest-margin bits (deterministic)
                * plus: a φ-fraction of bits drawn from the next-worst band
        """
        v = v_pm1.copy()

        # --- 1. local instability / chaos proxies ---
        h0 = self.W @ v
        margins0 = v * h0
        num_neg = np.sum(margins0 < 0)
        margin_neg_frac = float(num_neg) / float(margins0.size + 1e-12)

        # effective noise from Hamming distance if provided
        if d1 is not None and self.N > 0:
            p_noise = float(d1) / float(self.N)
        else:
            p_noise = margin_neg_frac

        # local chaos (0..1): blend Hamming noise + negative margins
        chaos = 0.5 * p_noise + 0.5 * margin_neg_frac

        # --- 2. adaptive (K0, steps) from BAM/SOM/UCEG law ---
        K0_adapt, steps_adapt = adaptive_annealing_params(
            p_noise=p_noise,
            margin_neg_frac=margin_neg_frac,
            gamma_factor=self.gamma_factor,
            K_max=self.base_K0_near,
            steps_max=self.base_near_steps,
        )

        # Optional explicit overrides
        if steps is not None:
            steps_adapt = int(max(1, steps))
        if K0 is not None:
            K0_adapt = int(max(1, K0))

        # --- φ-weight scaling of (K0, steps) ---
        # φ-scale: when chaos ~ 0, factor ~ 1/φ; when chaos ~ 1, factor ~ 1
        phi_scale = PHI_INV + chaos * (1.0 - PHI_INV)
        steps_adapt = max(1, int(round(steps_adapt * phi_scale)))
        K0_adapt = max(1, int(round(K0_adapt * phi_scale)))

        # --- 3. multi-step annealing with φ-schedule + Feigenbaum cooling ---
        for t in range(1, steps_adapt + 1):
            # Base K_t: φ-schedule or constant, as before
            if self.guided_mode == "phi":
                K_t_raw = phi_K_schedule(K0_adapt, t)
            else:
                K_t_raw = K0_adapt

            # Feigenbaum cooling: T(t) decreases with t
            T_t = feigenbaum_cooling(t, T0=1.0)

            # Shrink K_t by T_t^{1/φ} to give a graceful cooling
            K_t = K_t_raw * (T_t ** PHI_INV)
            K_t = int(max(1, round(K_t)))
            K_t = min(K_t, self.N)  # safety clamp

            # --- golden-randomized perturbation kernel ---
            h = self.W @ v
            margins = v * h
            order = np.argsort(margins)  # most negative first

            if K_t >= margins.size:
                idx = order
            else:
                # deterministic "core" flips
                K_core = max(1, int(round(K_t * PHI_INV)))  # ~ 0.618 K_t
                K_core = min(K_core, K_t)
                core_idx = order[:K_core]

                # jitter pool: next-worst band
                band_start = K_core
                band_end = min(margins.size, K_core + max(1, K_t - K_core) * 4)
                band = order[band_start:band_end]

                if band.size > 0:
                    K_jitter = K_t - K_core
                    K_jitter = max(0, K_jitter)
                    if K_jitter > 0:
                        if K_jitter >= band.size:
                            jitter_idx = band
                        else:
                            jitter_idx = self.rng.choice(
                                band,
                                size=K_jitter,
                                replace=False,
                            )
                        idx = np.concatenate([core_idx, jitter_idx])
                    else:
                        idx = core_idx
                else:
                    # fallback: purely deterministic lowest margins
                    idx = order[:K_t]

            v[idx] *= -1

        return v

    def _tau_prime(self, d1: int) -> float:
        """
        Compute tau'(d1) based on tau_mode:
          - "linear": tau' = alpha * delta_bar * d1
          - "phi"   : phi-annealed schedule
        """
        if self.tau_mode == "phi":
            return tau_prime_phi_schedule(d1, self.delta_bar, alpha=self.tau_alpha)
        else:
            return tau_prime_from_noise(d1, self.delta_bar, alpha=self.tau_alpha)

    def _pre_denoise(self, v_pm1: np.ndarray, q_bits: np.ndarray, topK: int = 5) -> np.ndarray:
        """
        Wrapper for pre-denoising the proposal vector v_pm1.
        """
        if not self.use_pre_denoise:
            return v_pm1

        if self.pre_denoise_mode == "bitflip":
            base_steps = 2
            steps = int(np.clip(round(base_steps * self.gamma_factor), 1, 4))
            return self._pre_denoise_bitflip(v_pm1, steps=steps)
        elif self.pre_denoise_mode == "attention":
            return attention_jump(
                v_pm1,
                self.patterns_pm1,
                beta=self.pre_denoise_beta,
                topK=topK,
            )
        return v_pm1

    def _certification(
        self,
        v_pm1: np.ndarray,
        xk_pm1: np.ndarray,
        v_bits: np.ndarray,
        xk_bits: np.ndarray,
        tau_prime: float,
        k_idx: int,
    ) -> bool:
        """
        Multichannel certification gate:
          Require S'(v) >= S'(x_k) - tau' AND majority passes over channels.
        """
        Spr_v = signed_score_Sprime(v_pm1, self.W)
        Spr_x = signed_score_Sprime(xk_pm1, self.W)
        pass_score = (Spr_v >= Spr_x - tau_prime)

        if not self.use_multichannel:
            return bool(pass_score)

        oc_v = ones_count(v_bits)
        oc_x = ones_count(xk_bits)
        pass_ones = (abs(oc_v - oc_x) <= max(1, int(0.02 * self.N)))

        sig11_v = pair_signature(v_bits, "11")
        sig00_v = pair_signature(v_bits, "00")
        if self.sig11 is not None and self.sig00 is not None:
            sig11_x = self.sig11[k_idx]
            sig00_x = self.sig00[k_idx]
        else:
            sig11_x = pair_signature(xk_bits, "11")
            sig00_x = pair_signature(xk_bits, "00")
        pass_sig11 = (np.mean(sig11_v == sig11_x) >= 0.98)
        pass_sig00 = (np.mean(sig00_v == sig00_x) >= 0.98)

        sk_v = minhash_pairs(v_bits, self.seeds)
        if self.sketch is not None:
            sk_x = self.sketch[k_idx]
        else:
            sk_x = minhash_pairs(xk_bits, self.seeds)
        pass_sketch = (
            np.linalg.norm(sk_v - sk_x, ord=1)
            <= max(2, 0.05 * np.linalg.norm(sk_x, ord=1))
        )

        passes = [
            bool(pass_score),
            bool(pass_ones),
            bool(pass_sig11),
            bool(pass_sig00),
            bool(pass_sketch),
        ]
        return majority_of_channels(passes, self.need_channels)

    # -----------------------------
    # Public API: query
    # -----------------------------

    def query(self, v_bits01: np.ndarray) -> Tuple[str, Optional[int], Dict]:
        """
        Returns (decision, index, meta)
          decision in {"KNOWN","NEAR-UNKNOWN","UNKNOWN"}
          index == stored pattern idx if KNOWN else None
          meta contains diagnostic fields
        """
        meta: Dict = {}

        v_bits = v_bits01.astype(np.uint8)
        v_pm1 = to_pm1(v_bits)

        # optional LDPC
        if self.use_ldpc and self.H_ldpc is not None:
            v_bits = ldpc_bitflip_decode(v_bits, self.H_ldpc, iters=self.ldpc_it)
            v_pm1 = to_pm1(v_bits)

        # candidate search
        cand, dists = self._nearest_candidates(v_bits)
        k1 = cand[int(np.argmin(dists))]
        d1 = int(np.min(dists))
        meta["k1"] = k1
        meta["d1"] = d1

        # three-zone decision
        if d1 > self.theta_near:
            zone = "FAR"
        elif d1 > self.theta_accept:
            zone = "NEAR"
        else:
            zone = "IN"
        meta["zone_initial"] = zone

        # optional k-NN vote
        if self.use_knn_vote and (self.theta_accept < d1 <= self.theta_near):
            v_pm1_prop = knn_vote(v_bits, self.patterns_bits, k=self.knn_k)
        else:
            v_pm1_prop = v_pm1.copy()

        # pre-denoise for NEAR
        if self.use_pre_denoise and zone == "NEAR":
            v_pm1_prop = self._pre_denoise(v_pm1_prop, v_bits, topK=self.knn_k)

        # NEAR-zone guided dither
        if zone == "NEAR":
            v_pm1_prop = self._guided_dither_bitflip(v_pm1_prop, d1=d1)

        v_bits_prop = to_bits01(v_pm1_prop)

        # tau'
        tau_prime = self._tau_prime(d1)
        meta["tau_prime"] = tau_prime

        # candidate pattern
        xk_bits = self.patterns_bits[k1]
        xk_pm1 = self.patterns_pm1[k1]

        # certification
        ok = self._certification(
            v_pm1_prop,
            xk_pm1,
            v_bits_prop,
            xk_bits,
            tau_prime,
            k1,
        )

        # FAR zone
        if zone == "FAR":
            meta["zone"] = "FAR"
            return "UNKNOWN", None, meta

        # IN zone
        if zone == "IN":
            meta["zone"] = "IN"
            if ok:
                return "KNOWN", int(k1), meta
            else:
                meta["zone"] = "REJECT_CERT"
                return "UNKNOWN", None, meta

        # NEAR zone: compute margins vs next best
        dists2 = dists.copy()
        dists2[np.argmin(dists)] = dists2.max() + 1
        k2 = cand[int(np.argmin(dists2))]
        d2 = int(np.min(dists2))

        Spr_v = signed_score_Sprime(v_pm1_prop, self.W)
        Spr1 = signed_score_Sprime(xk_pm1, self.W)
        Spr2 = signed_score_Sprime(self.patterns_pm1[k2], self.W)

        dd, dS = compute_margins(d1, d2, Spr1, Spr2)

        meta["zone"] = "NEAR"
        meta["k2"] = k2
        meta["d2"] = d2
        meta["margin_d"] = dd
        meta["margin_S"] = dS
        meta["Spr_v"] = Spr_v
        meta["Spr_k1"] = Spr1
        meta["Spr_k2"] = Spr2

        if ok and (dd >= self.gamma_d or dS >= self.gamma_S):
            return "KNOWN", int(k1), meta
        else:
            return "NEAR-UNKNOWN", None, meta


# ============================================================
# Benchmark harness (optional)
# ============================================================

def corrupt_bits(bits01: np.ndarray, flips: int) -> np.ndarray:
    out = bits01.copy()
    N = out.shape[0]
    if flips <= 0:
        return out
    idx = np.random.choice(N, size=min(flips, N), replace=False)
    out[idx] ^= 1
    return out


def evaluate(
    hosam: HOSAM,
    patterns_bits: np.ndarray,
    noise_levels: List[float],
    n_unknown: int = 1000,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate recall and FPR across noise levels.

    Returns: dict[nl] -> {
        'TP', 'FN', 'FP', 'TN', 'TPR', 'FPR', 'time_ms'
    }
    """
    P, N = patterns_bits.shape
    results: Dict[float, Dict[str, float]] = {}

    for nl in noise_levels:
        flips = int(round(nl * N))
        TP = FN = FP = TN = 0
        t0 = time.time()

        # known queries
        for i in range(P):
            q = corrupt_bits(patterns_bits[i], flips)
            dec, idx, _ = hosam.query(q)
            if dec == "KNOWN" and idx == i:
                TP += 1
            else:
                FN += 1

        # unknown queries
        for _ in range(n_unknown):
            q = np.random.randint(0, 2, size=N, dtype=np.uint8)
            dec, idx, _ = hosam.query(q)
            if dec == "KNOWN":
                FP += 1
            else:
                TN += 1

        t1 = time.time()
        total_queries = P + n_unknown
        time_ms = 1000.0 * (t1 - t0) / max(1, total_queries)

        results[nl] = {
            "TP": float(TP),
            "FN": float(FN),
            "FP": float(FP),
            "TN": float(TN),
            "TPR": float(TP) / max(1.0, P),
            "FPR": float(FP) / max(1.0, n_unknown),
            "time_ms": float(time_ms),
        }

    return results
