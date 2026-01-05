#%%writefile analyze_feature_importance_gram.py
import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

from hosam import HOSAM
from gram_variants import GRAM_plain, GRAM_phiFull
from gram_recursive import GRAMFeatureSelector

# ----------------- CONFIG -----------------

CSV_PATH  = "/content/drive/My Drive/Sweep_final.csv"
PV_COL    = "nPV"
BESS_COL  = "EBatMax [kW.h]"
COST_COL  = "lcc.y"
SSR_COL   = "ssr.y"
SCR_COL   = "scr.y"

PV_MAX    = 18500
BESS_MAX  = 36400
BESS_LEVELS = 64

SSR_MIN   = 0.40          # analysis focus
Q_SAMPLES = 800           # number of queries per method
M_CAND    = 16            # fallback candidate pool size
SEED      = 123           # single seed for importance scan

OUT_DIR   = "feature_importance_out"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_BITS  = os.path.join(OUT_DIR, "gram_feature_importance.csv")
OUT_SUM   = os.path.join(OUT_DIR, "gram_feature_importance_summary.csv")

# ----------------- BASIC UTILITIES -----------------

def nbits_for_levels(levels: int) -> int:
    return max(1, int(np.ceil(np.log2(max(2, levels)))))

def quantize_levels(arr, vmin, vmax, levels):
    arr = np.clip(arr, vmin, vmax)
    step = (vmax - vmin) / (levels - 1) if levels > 1 else 1.0
    idx  = np.round((arr - vmin) / (step + 1e-12)).astype(int)
    idx  = np.clip(idx, 0, levels - 1)
    return idx, step

def build_objective_and_feasible(df: pd.DataFrame, ssr_min: float):
    df = df.copy()
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
        (df[SSR_COL] >= ssr_min) &
        (df[PV_COL] <= PV_MAX) &
        (df[BESS_COL] <= BESS_MAX)
    )
    return df, feas_mask

def encode_bank(bank: pd.DataFrame, df_full: pd.DataFrame):
    """Binary-encode nPV and BESS following your GRAM scripts."""
    pv_min, pv_max   = df_full[PV_COL].min(),   df_full[PV_COL].max()
    bess_min, bess_max = df_full[BESS_COL].min(), df_full[BESS_COL].max()

    # pv_levels chosen from unique nPV count, rounded up to power of 2
    pv_levels = 1 << int(np.ceil(np.log2(max(2, bank[PV_COL].nunique()))))
    pv_bits   = nbits_for_levels(pv_levels)
    bess_bits = nbits_for_levels(BESS_LEVELS)

    pv_idx, pv_step     = quantize_levels(bank[PV_COL].to_numpy(),   pv_min, pv_max, pv_levels)
    bess_idx, bess_step = quantize_levels(bank[BESS_COL].to_numpy(), bess_min, bess_max, BESS_LEVELS)

    P = len(bank)
    N_bits = pv_bits + bess_bits
    X_bits = np.zeros((P, N_bits), dtype=np.uint8)

    for i in range(P):
        for b in range(pv_bits):
            X_bits[i, b] = (pv_idx[i] >> b) & 1
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

def fit_surface_cost_ssr_scr(front: pd.DataFrame) -> Tuple[np.ndarray, float, int]:
    S = front[SSR_COL].to_numpy(dtype=float)
    H = front[SCR_COL].to_numpy(dtype=float)
    C = front[COST_COL].to_numpy(dtype=float)
    n = len(S)
    if n < 6:
        return np.zeros(6, float), np.nan, n
    X = np.stack([
        np.ones_like(S),
        S,
        H,
        S**2,
        H**2,
        S*H
    ], axis=1)
    beta, *_ = np.linalg.lstsq(X, C, rcond=None)
    C_hat = X @ beta
    ss_tot = np.sum((C - C.mean())**2)
    ss_res = np.sum((C - C_hat)**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta.astype(float), float(R2), n

# ----------------- GRAM + FS -----------------

def build_gram_models(X_bits):
    """Build GRAM_plain, GRAM_phiFull, and FS(phiFull) using first-half mask."""
    gram_plain   = GRAM_plain(X_bits)
    gram_phiFull = GRAM_phiFull(X_bits)

    P, N_bits = X_bits.shape
    mask_first_half = np.zeros(N_bits, dtype=bool)
    mask_first_half[:N_bits // 2] = True  # your current FS design

    fs_phiFull = GRAMFeatureSelector(mask=mask_first_half, gram_ctor=GRAM_phiFull)
    fs_phiFull.fit(X_bits)
    return gram_plain, gram_phiFull, fs_phiFull, mask_first_half

def make_selectors_from_models(gram_plain, gram_phiFull, fs_phiFull, X_bits, bank):
    """Replicate your J-based feasible-first fallback logic."""
    def gram_select(gram, q_bits):
        dec, idx, meta = gram.query(q_bits)
        if dec == "KNOWN" and idx is not None:
            return int(idx)
        # fallback: nearest by Hamming, then J-based feasible re-ranking
        d_all = np.sum(np.abs(X_bits - q_bits), axis=1)
        M = min(M_CAND, len(X_bits))
        cand_idx = np.argpartition(d_all, M - 1)[:M]
        cand = bank.iloc[cand_idx].copy()
        feas = cand[
            (cand[SSR_COL]  >= SSR_MIN) &
            (cand[PV_COL]   <= PV_MAX) &
            (cand[BESS_COL] <= BESS_MAX)
        ]
        if len(feas):
            return int(feas["J"].idxmin())
        # penalty fallback if no feasible
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

    def sel_plain(q_bits):   return gram_select(gram_plain,   q_bits)
    def sel_phi(q_bits):     return gram_select(gram_phiFull, q_bits)
    def sel_fs(q_bits):
        # try FS (coarse view) first
        dec, idx, meta = fs_phiFull.query(q_bits)
        if dec == "KNOWN" and idx is not None:
            return int(idx)
        # otherwise fall back to full-resolution φFull
        return gram_select(gram_phiFull, q_bits)

    return {"GRAM_plain": sel_plain, "GRAM_phiFull": sel_phi, "GRAM_FS_phiFull": sel_fs}

def eval_method_R2(df_full, bank, encode_row, selector_fn, X_bits):
    """Evaluate 3D Cost(SSR,SCR) R² for a given selector."""
    df_all, _ = build_objective_and_feasible(df_full, SSR_MIN)
    if df_all.empty:
        return np.nan
    df_sample = df_all.sample(n=min(Q_SAMPLES, len(df_all)), random_state=SEED).reset_index(drop=True)
    costs, ssrs, scrs = [], [], []
    for _, row in df_sample.iterrows():
        q_bits = encode_row(row)
        idx = selector_fn(q_bits)
        chosen = bank.iloc[idx]
        costs.append(float(chosen[COST_COL]))
        ssrs.append(float(chosen[SSR_COL]))
        scrs.append(float(chosen[SCR_COL]))
    front = pd.DataFrame({COST_COL: costs, SSR_COL: ssrs, SCR_COL: scrs})
    beta, R2, n = fit_surface_cost_ssr_scr(front)
    return R2

# ----------------- MAIN IMPORTANCE ANALYSIS -----------------

def main():
    df_full = pd.read_csv(CSV_PATH)
    df_full[PV_COL]   = pd.to_numeric(df_full[PV_COL],   errors="coerce")
    df_full[BESS_COL] = pd.to_numeric(df_full[BESS_COL], errors="coerce")
    df_full[COST_COL] = pd.to_numeric(df_full[COST_COL], errors="coerce")
    df_full[SSR_COL]  = pd.to_numeric(df_full[SSR_COL],  errors="coerce")
    df_full[SCR_COL]  = pd.to_numeric(df_full[SCR_COL],  errors="coerce")

    df, feas_mask = build_objective_and_feasible(df_full, SSR_MIN)
    bank = df[feas_mask].copy().reset_index(drop=True)
    print(f"Feasible rows at SSR_MIN={SSR_MIN}: {len(bank)}/{len(df)}")

    X_bits, encode_row, pv_bits, bess_bits = encode_bank(bank, df)

    gram_plain, gram_phiFull, fs_phiFull, mask_fs = build_gram_models(X_bits)
    selectors = make_selectors_from_models(gram_plain, gram_phiFull, fs_phiFull, X_bits, bank)

    # baseline R^2 for each method
    base_R2 = {}
    for method in ["GRAM_plain", "GRAM_phiFull", "GRAM_FS_phiFull"]:
        R2 = eval_method_R2(df_full, bank, encode_row, selectors[method], X_bits)
        base_R2[method] = R2
        print(f"Base R2 for {method}: {R2:.4f}")

    records = []
    P, N_bits = X_bits.shape

    # bit-wise ablation
    for j in range(N_bits):
        print(f"\nAblating bit {j+1}/{N_bits}")
        X_abl = X_bits.copy()
        X_abl[:, j] = 0

        gram_plain_a, gram_phiFull_a, fs_phiFull_a, _ = build_gram_models(X_abl)
        selectors_a = make_selectors_from_models(gram_plain_a, gram_phiFull_a, fs_phiFull_a, X_abl, bank)

        for method in ["GRAM_plain", "GRAM_phiFull", "GRAM_FS_phiFull"]:
            R2_a = eval_method_R2(df_full, bank, encode_row, selectors_a[method], X_abl)
            imp = base_R2[method] - R2_a
            var = "PV" if j < pv_bits else "BESS"
            pos = j if j < pv_bits else (j - pv_bits)
            in_fs = bool(mask_fs[j])  # whether this bit is in FS mask
            records.append({
                "SSR_MIN": SSR_MIN,
                "method": method,
                "bit_index": j,
                "variable": var,
                "bit_pos": pos,
                "in_FS_mask": in_fs,
                "base_R2": base_R2[method],
                "R2_ablated": R2_a,
                "importance": imp,
            })
            print(f"  {method}: R2_ablated={R2_a:.4f}, importance={imp:.4f}, in_FS={in_fs}")

    df_bits = pd.DataFrame(records)
    df_bits.to_csv(OUT_BITS, index=False)
    print(f"\nWrote bit-level importance to {OUT_BITS}")

    # aggregate by variable + FS membership
    df_var = df_bits.groupby(["SSR_MIN", "method", "variable", "in_FS_mask"])["importance"].sum().reset_index()
    df_var.to_csv(OUT_SUM, index=False)
    print(f"Wrote variable-level importance to {OUT_SUM}")

if __name__ == "__main__":
    main()
