# %%writefile manifold_minimal_surfaces.py
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Optional: Colab drive mount
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

from hosam import HOSAM
from gram_variants import GRAM_plain, GRAM_phiFull
from gram_recursive import GRAMFeatureSelector
from baselines_am import (
    DenseAssociativeMemory,
    GriponBerrouAM,
    ModernHopfieldAttention,
)

# ----------------- CONFIG -----------------

CSV_PATH = "/content/drive/My Drive/Sweep_final.csv"

PV_COL = "nPV"
BESS_COL = "EBatMax [kW.h]"
COST_COL = "lcc.y"
SSR_COL = "ssr.y"
SCR_COL = "scr.y"

PV_MAX = 18500
BESS_MAX = 36400
BESS_LEVELS = 64

SSR_MIN = 0.40  # adjust as needed

# Target numbers of unique designs for minimal-manifold fits
TARGET_UNIQUES = [150, 230]   # approx 130-160 and 210-230; adjust if needed
MAX_UNIQUE = 800              # global cap on unique designs

BATCH_SIZE = 50               # queries per iteration
MAX_STEPS = 200               # safety cap
RANDOM_SEED = 123

OUT_DIR = "manifold_minimal_out"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "minimal_surfaces_summary.csv")

# MHN betas
MHN_BETAS = [5.0, 10.0, 20.0]

# ----------------- BASIC UTILITIES -----------------


def nbits_for_levels(levels: int) -> int:
    return max(1, int(np.ceil(np.log2(max(2, levels)))))


def quantize_levels(arr, vmin, vmax, levels):
    arr = np.clip(arr, vmin, vmax)
    step = (vmax - vmin) / (levels - 1) if levels > 1 else 1.0
    idx = np.round((arr - vmin) / (step + 1e-12)).astype(int)
    idx = np.clip(idx, 0, levels - 1)
    return idx, step


def build_feasible_bank(df_full: pd.DataFrame, ssr_min: float) -> pd.DataFrame:
    df = df_full.copy()
    df[PV_COL]   = pd.to_numeric(df[PV_COL],   errors="coerce")
    df[BESS_COL] = pd.to_numeric(df[BESS_COL], errors="coerce")
    df[COST_COL] = pd.to_numeric(df[COST_COL], errors="coerce")
    df[SSR_COL]  = pd.to_numeric(df[SSR_COL],  errors="coerce")
    df[SCR_COL]  = pd.to_numeric(df[SCR_COL],  errors="coerce")
    df[SCR_COL]  = df[SCR_COL].clip(0.0, 1.0)

    df = df.dropna(subset=[PV_COL, BESS_COL, COST_COL, SSR_COL]).reset_index(drop=True)

    cmin, cmax = df[COST_COL].min(), df[COST_COL].max()
    df["LCC_norm"] = (df[COST_COL] - cmin) / (cmax - cmin + 1e-12)
    df["J"] = df["LCC_norm"] + (1.0 - df[SSR_COL]) + (1.0 - df[SCR_COL])

    feas_mask = (
        (df[SSR_COL] >= ssr_min)
        & (df[PV_COL]  <= PV_MAX)
        & (df[BESS_COL] <= BESS_MAX)
    )
    bank = df.loc[feas_mask].copy().reset_index(drop=True)
    print(f"Feasible rows at SSR_MIN={ssr_min}: {len(bank)}/{len(df)}")
    return bank


def encode_bank(bank: pd.DataFrame, df_full: pd.DataFrame) -> Tuple[np.ndarray, callable, int, int]:
    """Return X_bits, encode_row, pv_bits, bess_bits."""
    pv_min, pv_max   = df_full[PV_COL].min(),   df_full[PV_COL].max()
    bess_min, bess_max = df_full[BESS_COL].min(), df_full[BESS_COL].max()

    pv_levels = 1 << int(np.ceil(np.log2(max(2, df_full[PV_COL].nunique()))))
    pv_bits   = nbits_for_levels(pv_levels)
    bess_bits = nbits_for_levels(BESS_LEVELS)

    pv_idx, pv_step   = quantize_levels(bank[PV_COL].to_numpy(),   pv_min,  pv_max,  pv_levels)
    bess_idx, bess_step = quantize_levels(bank[BESS_COL].to_numpy(), bess_min, bess_max, BESS_LEVELS)

    P = len(bank)
    N_bits = pv_bits + bess_bits
    X_bits = np.zeros((P, N_bits), dtype=np.uint8)

    for i in range(P):
        for b in range(pv_bits):
            X_bits[i, b] = (pv_idx[i]   >> b) & 1
        for b in range(bess_bits):
            X_bits[i, pv_bits + b] = (bess_idx[i] >> b) & 1

    def encode_row(row: pd.Series) -> np.ndarray:
        pvi = int(np.round((row[PV_COL]   - pv_min)   / (pv_step   + 1e-12)))
        bsi = int(np.round((row[BESS_COL] - bess_min) / (bess_step + 1e-12)))
        pvi = np.clip(pvi, 0, pv_levels   - 1)
        bsi = np.clip(bsi, 0, BESS_LEVELS - 1)
        bits = np.zeros(N_bits, dtype=np.uint8)
        for b in range(pv_bits):
            bits[b] = (pvi >> b) & 1
        for b in range(bess_bits):
            bits[pv_bits + b] = (bsi >> b) & 1
        return bits

    return X_bits, encode_row, pv_bits, bess_bits


def fit_surface(front: pd.DataFrame) -> Tuple[np.ndarray, float, int]:
    """Fit Cost ~ quadratic(SSR, SCR)."""
    S = front[SSR_COL].to_numpy(dtype=float)
    H = front[SCR_COL].to_numpy(dtype=float)
    C = front[COST_COL].to_numpy(dtype=float)
    n = len(S)
    if n < 6:
        return np.zeros(6, float), np.nan, n
    X = np.stack([np.ones_like(S), S, H, S**2, H**2, S * H], axis=1)
    beta, *_ = np.linalg.lstsq(X, C, rcond=None)
    C_hat = X @ beta
    ss_tot = np.sum((C - C.mean())**2)
    ss_res = np.sum((C - C_hat)**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta.astype(float), float(R2), n

# ------------------ Build selectors ------------------


def build_selectors(
    df_full: pd.DataFrame,
    bank: pd.DataFrame,
    X_bits: np.ndarray,
    pv_bits: int,
    bess_bits: int
) -> Dict[str, callable]:
    """
    Build selectors for each method and return a dict mapping
    method names to selector functions.
    """
    P, N_bits = X_bits.shape

    # Base GRAM models
    gram_plain   = GRAM_plain(X_bits)
    gram_phiFull = GRAM_phiFull(X_bits)

    # FS mask: try to load learned mask, else first half
    mask_fs = np.zeros(N_bits, dtype=bool)
    mask_path = "feature_importance_out/learned_fs_mask.npy"
    if os.path.exists(mask_path):
        m = np.load(mask_path)
        if m.shape[0] == N_bits:
            mask_fs = m.astype(bool)
            print(f"Using learned FS mask from {mask_path} (true bits: {mask_fs.sum()}/{N_bits})")
        else:
            print("WARNING: learned mask length mismatch; using first-half mask.")
            mask_fs[: N_bits // 2] = True
    else:
        print("No learned FS mask, using first-half mask.")
        mask_fs[: N_bits // 2] = True

    fs_phiFull = GRAMFeatureSelector(mask=mask_fs, gram_ctor=GRAM_phiFull)
    fs_phiFull.fit(X_bits)

    # Hopfield built from GRAM_plain weights
    W_hop = gram_plain.W

    def hopfield_minimize(y0_bits, W, max_iters=50):
        y = (y0_bits.astype(np.int8) * 2 - 1).copy()
        n = y.shape[0]
        for _ in range(max_iters):
            changed = 0
            for i in range(n):
                h = int(W[i] @ y)
                new = 1 if h >= 0 else -1
                if new != y[i]:
                    y[i] = new
                    changed += 1
            if changed == 0:
                break
        return (y > 0).astype(np.uint8)

    # Common J-based fallback used by GRAM variants
    def gram_fallback_select(gram, q_bits):
        dec, idx, meta = gram.query(q_bits)
        if dec == "KNOWN" and idx is not None:
            return int(idx)

        # Hamming distance shortlist
        d_all = np.sum(np.abs(X_bits - q_bits), axis=1)
        M = min(16, len(X_bits))
        cand_idx = np.argpartition(d_all, M - 1)[:M]
        cand = bank.iloc[cand_idx].copy()

        feas = cand[
            (cand[SSR_COL]  >= SSR_MIN)
            & (cand[PV_COL]  <= PV_MAX)
            & (cand[BESS_COL] <= BESS_MAX)
        ]
        if len(feas):
            return int(feas["J"].idxmin())

        # penalty fallback if nothing feasible in shortlist
        def pen(row):
            p = 0.0
            if row[SSR_COL] < SSR_MIN:
                p += (SSR_MIN - row[SSR_COL]) * 1_000.0
            if row[PV_COL] > PV_MAX:
                p += (row[PV_COL] - PV_MAX) * 1_000.0
            if row[BESS_COL] > BESS_MAX:
                p += (row[BESS_COL] - BESS_MAX) * 1_000.0
            return p

        cand["J_pen"] = cand["J"] + cand.apply(pen, axis=1)
        return int(cand["J_pen"].idxmin())

    # ---- Selector wrappers ----

    def sel_gram_phi(q_bits):
        return gram_fallback_select(gram_phiFull, q_bits)

    def sel_gram_fs(q_bits):
        dec, idx, meta = fs_phiFull.query(q_bits)
        if dec == "KNOWN" and idx is not None:
            return int(idx)
        return gram_fallback_select(gram_phiFull, q_bits)

    def sel_hopfield(q_bits):
        y_bits = hopfield_minimize(q_bits, W_hop)
        d_all = np.sum(np.abs(X_bits - y_bits), axis=1)
        return int(np.argmin(d_all))

    def sel_dam(q_bits):
        bestJ = np.inf
        bestK = 0
        for n_order in [3, 5]:
            for steps in [1, 3, 5]:
                dam = DenseAssociativeMemory(X_bits, n_order=n_order, steps=steps)
                k = dam.retrieve(q_bits)
                sc = float(bank.iloc[k]["J"])
                if sc < bestJ:
                    bestJ, bestK = sc, k
        return int(bestK)

    def sel_gba(q_bits):
        C = 8 if N_bits >= 32 else 4
        gba = GriponBerrouAM(X_bits, C=C)
        return int(gba.retrieve(q_bits))

    def sel_mhn(q_bits):
        bestJ = np.inf
        bestK = 0
        for beta in MHN_BETAS:
            m = ModernHopfieldAttention(X_bits, beta=beta)
            k = m.retrieve(q_bits)
            sc = float(bank.iloc[k]["J"])
            if sc < bestJ:
                bestJ, bestK = sc, k
        return int(bestK)

    def sel_gram_feig_cooling(q_bits):
        """
        Selector using Feigenbaum-cooled NEAR dither implemented inside HOSAM.

        We assume HOSAM supports:
            mem = HOSAM(X_bits)
            dec, idx, meta = mem.query(q_bits)
        and returns an index idx into X_bits / bank when dec == "KNOWN".

        If the HOSAM API differs, adjust this function accordingly.
        """
        # Build HOSAM once per selector (per process)
        if not hasattr(sel_gram_feig_cooling, "_mem"):
            sel_gram_feig_cooling._mem = HOSAM(X_bits)
        mem = sel_gram_feig_cooling._mem

        # Try HOSAM's query interface
        try:
            dec, idx, meta = mem.query(q_bits)
            if dec == "KNOWN" and idx is not None:
                return int(idx)
        except Exception:
            # If HOSAM has a different API, this is where we'd adapt it.
            pass

        # Fallback: Hamming NN in the bank
        d_all = np.sum(np.abs(X_bits - q_bits), axis=1)
        return int(np.argmin(d_all))

    selectors: Dict[str, callable] = {}
    selectors["GRAM_phiFull"]     = sel_gram_phi
    selectors["GRAM_FS_phiFull"]  = sel_gram_fs
    selectors["Hopfield"]         = sel_hopfield
    selectors["DAM"]              = sel_dam
    selectors["GB-AM"]            = sel_gba
    selectors["MHN"]              = sel_mhn
    selectors["GRAM_FeigCooling"] = sel_gram_feig_cooling

    return selectors

# ------------------ Collect unique selections ------------------


def collect_unique_indices(
    df_all: pd.DataFrame,
    encode_row,
    selector_fn,
    target_unique: int,
    max_unique: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    step = 0

    while True:
        if len(set(chosen)) >= min(target_unique, max_unique):
            break
        step += 1
        if step > MAX_STEPS:
            print(
                f"  Reached MAX_STEPS={MAX_STEPS} for target {target_unique}, "
                f"stopping with {len(set(chosen))} uniques."
            )
            break

        # sample a batch of queries
        batch_n = min(batch_size, len(df_all))
        df_sample = df_all.sample(
            n=batch_n, random_state=rng.integers(1_000_000)
        )

        for _, row in df_sample.iterrows():
            q_bits = encode_row(row)
            idx = selector_fn(q_bits)
            chosen.append(idx)

    uniq = np.unique(np.array(chosen))
    if len(uniq) > target_unique:
        uniq = uniq[:target_unique]
    print(f"  Collected {len(uniq)} unique designs for target {target_unique}.")
    return uniq

# ------------------ MAIN ------------------


def main():
    # Load full data and build feasible bank at SSR_MIN
    df_full = pd.read_csv(CSV_PATH)
    bank = build_feasible_bank(df_full, SSR_MIN)

    # Encode bank
    X_bits, encode_row, pv_bits, bess_bits = encode_bank(bank, df_full)

    # Build selectors
    selectors = build_selectors(df_full, bank, X_bits, pv_bits, bess_bits)

    # Methods to analyze
    methods = [
        "GRAM_phiFull",
        "GRAM_FS_phiFull",
        "Hopfield",
        "DAM",
        "GB-AM",
        "MHN",
        "GRAM_FeigCooling",
    ]

    # For reproducibility
    base_rng = np.random.default_rng(RANDOM_SEED)

    results = []

    # Fit TRUE surface on full bank
    beta_true, R2_true, n_true = fit_surface(bank)
    print("\nTRUE surface fit:")
    print("  n =", n_true, "R2 =", R2_true)
    print("  beta_true:", beta_true)

    # For each method and target unique count
    for method in methods:
        for target in TARGET_UNIQUES:
            print(f"\n=== Method {method}, target_unique={target} ===")
            seed = int(base_rng.integers(1_000_000))
            uniq_idx = collect_unique_indices(
                df_all=bank,
                encode_row=encode_row,
                selector_fn=selectors[method],
                target_unique=target,
                max_unique=MAX_UNIQUE,
                batch_size=BATCH_SIZE,
                seed=seed,
            )
            front = bank.iloc[uniq_idx].copy()
            beta, R2, n = fit_surface(front)
            print(f"  Used n={n} unique designs, R2={R2:.5f}")
            print(f"  beta: {beta}")

            results.append(
                {
                    "method": method,
                    "target_unique": target,
                    "used_unique": int(n),
                    "b0": float(beta[0]),
                    "b1": float(beta[1]),
                    "b2": float(beta[2]),
                    "b3": float(beta[3]),
                    "b4": float(beta[4]),
                    "b5": float(beta[5]),
                    "R2": float(R2),
                }
            )

            # Save subset for plotting later
            front.to_csv(
                os.path.join(OUT_DIR, f"{method}_N{target}_subset.csv"),
                index=False,
            )

    # Save summary CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved summary to {OUT_CSV}")

    # ------------- Plotting 3D surfaces -------------

    # Build grid over full SSR/SCR range
    ssr_min, ssr_max = bank[SSR_COL].min(), bank[SSR_COL].max()
    scr_min, scr_max = bank[SCR_COL].min(), bank[SCR_COL].max()
    ssr_grid = np.linspace(ssr_min, ssr_max, 50)
    scr_grid = np.linspace(scr_min, scr_max, 50)
    SSRg, SCRg = np.meshgrid(ssr_grid, scr_grid)

    def eval_surface(params, SSR, SCR):
        b0, b1, b2, b3, b4, b5 = params
        return (
            b0
            + b1 * SSR
            + b2 * SCR
            + b3 * SSR**2
            + b4 * SCR**2
            + b5 * SSR * SCR
        )

    for target in TARGET_UNIQUES:
        fig = plt.figure(figsize=(18, 9))

        # TRUE surface
        Z_true = eval_surface(beta_true, SSRg, SCRg)
        ax = fig.add_subplot(2, 4, 1, projection="3d")
        ax.scatter(
            bank[SSR_COL], bank[SCR_COL], bank[COST_COL],
            s=5, c="gray", alpha=0.3
        )
        ax.plot_surface(SSRg, SCRg, Z_true, alpha=0.6, cmap="viridis")
        ax.set_title(f"TRUE (n={n_true}, R²={R2_true:.3f})")
        ax.set_xlabel("SSR")
        ax.set_ylabel("SCR")
        ax.set_zlabel("Cost")

        for j, method in enumerate(methods, start=2):
            sub = df_res[
                (df_res["method"] == method)
                & (df_res["target_unique"] == target)
            ]
            if sub.empty:
                continue

            beta = np.array(
                [
                    sub["b0"].item(),
                    sub["b1"].item(),
                    sub["b2"].item(),
                    sub["b3"].item(),
                    sub["b4"].item(),
                    sub["b5"].item(),
                ]
            )
            R2 = sub["R2"].item()
            n_used = int(sub["used_unique"].item())

            Z = eval_surface(beta, SSRg, SCRg)
            axm = fig.add_subplot(2, 4, j, projection="3d")
            axm.scatter(
                bank[SSR_COL], bank[SCR_COL], bank[COST_COL],
                s=5, c="gray", alpha=0.2
            )
            axm.plot_surface(SSRg, SCRg, Z, alpha=0.7)
            axm.set_title(f"{method}, N={n_used}, R²={R2:.3f}")
            axm.set_xlabel("SSR")
            axm.set_ylabel("SCR")
            axm.set_zlabel("Cost")

        plt.suptitle(
            f"Minimal-manifold fits at SSR_MIN={SSR_MIN}, target N={target}"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUT_DIR,
                f"surfaces_SSRmin{SSR_MIN}_N{target}.png",
            ),
            dpi=150,
        )
        plt.show()


if __name__ == "__main__":
    main()
