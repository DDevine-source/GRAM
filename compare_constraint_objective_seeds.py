# %%writefile run_compare_constraint_objective_seeds_newGRAM.py
import os, csv, time, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Colab drive mount (no-op if local)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

# ------------------------------------------------------------------
# Imports: GRAM variants + recursive wrappers + baselines
# ------------------------------------------------------------------
from hosam import HOSAM

from gram_variants import GRAM_plain, GRAM_phiJ, GRAM_phiFull
from gram_recursive import GRAMCoarseFine, GRAMFeatureSelector
from baselines_am import (
    ModernHopfieldAttention,
    DenseAssociativeMemory,
    GriponBerrouAM,
    WillshawCMM,
    SparseDistributedMemory,
)

# ------------------------------------------------------------------
# GRNN implementation
# ------------------------------------------------------------------


class GRNN:
    """
    Generalized Regression Neural Network (Specht, 1991).

    X_train: (P, D) inputs (here normalized [nPV, BESS])
    Y_train: (P, K) outputs [LCC_norm, SSR, SCR, Cost]

    predict(x_norm) returns shape (K,)
    """

    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, sigma: float = 0.15):
        self.X = np.asarray(X_train, dtype=float)
        self.Y = np.asarray(Y_train, dtype=float)
        self.P, self.D = self.X.shape
        assert self.Y.shape[0] == self.P
        self.sigma = float(sigma)
        self._eps = 1e-12

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(1, -1)  # (1,D)
        d2 = np.sum((self.X - x) ** 2, axis=1)         # (P,)
        w  = np.exp(-d2 / (2.0 * self.sigma**2))      # (P,)
        Z  = np.sum(w) + self._eps
        w_norm = w / Z
        y_hat = w_norm @ self.Y                      # (K,)
        return y_hat


# ---------------------- CONFIG ----------------------

CSV_PATH  = "/content/drive/My Drive/Sweep_final.csv"
OUT_ROOT  = "am_constraint_objective_seeds_out_newGRAM"

PV_COL    = "nPV"
BESS_COL  = "EBatMax [kW.h]"
COST_COL  = "lcc.y"
SSR_COL   = "ssr.y"
SCR_COL   = "scr.y"  # if missing, we set SCR=0

# Hard constraints
SSR_MIN   = 0.40  # adjust here to rerun other regimes
PV_MAX    = 18500
BESS_MAX  = 36400

# Encoding for AMs
BESS_LEVELS = 64
Q           = 1000  # queries per seed

# GRAM fallback: re-rank M nearest by J (feasible-first)
M_CAND      = 16

# Seeds
SEEDS       = [123, 456, 789, 101112, 131415]

# Baseline sweeps
MHN_BETAS   = [5.0, 10.0, 20.0]
DAM_ORDERS  = [3, 5]
DAM_STEPS   = [1, 3, 5]


# ---------------------------------------------------

def nbits_for_levels(levels: int) -> int:
    return max(1, int(np.ceil(np.log2(max(2, levels)))))


def quantize_levels(arr, vmin, vmax, levels):
    arr = np.clip(arr, vmin, vmax)
    step = (vmax - vmin) / (levels - 1) if levels > 1 else 1.0
    idx  = np.round((arr - vmin) / (step + 1e-12)).astype(int)
    idx  = np.clip(idx, 0, levels - 1)
    return idx, step


def build_objective_and_feasibility(df: pd.DataFrame):
    # numeric + hygiene
    df[PV_COL]   = pd.to_numeric(df[PV_COL],   errors="coerce")
    df[BESS_COL] = pd.to_numeric(df[BESS_COL], errors="coerce")
    df[COST_COL] = pd.to_numeric(df[COST_COL], errors="coerce")
    df[SSR_COL]  = pd.to_numeric(df[SSR_COL],  errors="coerce")
    if SCR_COL in df.columns:
        df[SCR_COL] = pd.to_numeric(df[SCR_COL], errors="coerce")
        df[SCR_COL] = df[SCR_COL].clip(0.0, 1.0)
    else:
        df[SCR_COL] = 0.0

    df = df.dropna(subset=[PV_COL, BESS_COL, COST_COL, SSR_COL]).reset_index(drop=True)

    # LCC_norm on whole dataset
    cmin, cmax = df[COST_COL].min(), df[COST_COL].max()
    df["LCC_norm"] = (df[COST_COL] - cmin) / (cmax - cmin + 1e-12)

    # NEW Objective: J = LCC_norm + (1 - SSR)
    # (SCR is no longer in the objective, but kept for reporting.)
    df["J"] = df["LCC_norm"] + (1.0 - df[SSR_COL])

    # Feasibility
    feas_mask = (
        (df[SSR_COL]  >= SSR_MIN) &
        (df[PV_COL]   <= PV_MAX)  &
        (df[BESS_COL] <= BESS_MAX)
    )
    return df, feas_mask


def encode_bank(bank: pd.DataFrame, df_full: pd.DataFrame):
    # Use full-dataset bounds for stable encoding
    pv_min, pv_max     = df_full[PV_COL].min(),   df_full[PV_COL].max()
    bess_min, bess_max = df_full[BESS_COL].min(), df_full[BESS_COL].max()

    pv_levels = 1 << int(np.ceil(np.log2(max(2, df_full[PV_COL].nunique()))))
    pv_bits   = nbits_for_levels(pv_levels)
    bess_bits = nbits_for_levels(BESS_LEVELS)

    pv_idx, pv_step     = quantize_levels(bank[PV_COL].to_numpy(),   pv_min,  pv_max,  pv_levels)
    bess_idx, bs_step   = quantize_levels(bank[BESS_COL].to_numpy(), bess_min, bess_max, BESS_LEVELS)

    P = len(bank)
    N_bits = pv_bits + bess_bits
    X_bits = np.zeros((P, N_bits), dtype=np.uint8)

    for i in range(P):
        for b in range(pv_bits):
            X_bits[i, b] = (pv_idx[i] >> b) & 1
        for b in range(bess_bits):
            X_bits[i, pv_bits + b] = (bess_idx[i] >> b) & 1

    def encode_row(row):
        pvi = int(np.round((row[PV_COL]   - pv_min)   / (pv_step   + 1e-12)))
        bsi = int(np.round((row[BESS_COL] - bess_min) / (bs_step   + 1e-12)))
        pvi = np.clip(pvi, 0, pv_levels   - 1)
        bsi = np.clip(bsi, 0, BESS_LEVELS - 1)

        bits = np.zeros(N_bits, dtype=np.uint8)
        for b in range(pv_bits):
            bits[b] = (pvi >> b) & 1
        for b in range(bess_bits):
            bits[pv_bits + b] = (bsi >> b) & 1
        return bits

    return X_bits, encode_row


def feasibility_penalty_values(ssr, pv, bess):
    PENALTY = 1_000.0
    pen = 0.0
    if ssr < SSR_MIN:
        pen += (SSR_MIN - ssr) * PENALTY
    if pv > PV_MAX:
        pen += (pv - PV_MAX) * PENALTY
    if bess > BESS_MAX:
        pen += (bess - BESS_MAX) * PENALTY
    return pen


def feasibility_penalty(row):
    return feasibility_penalty_values(row[SSR_COL], row[PV_COL], row[BESS_COL])

# ------------------------------------------------------------------
# FeigCooling selector (top-level helper)
# ------------------------------------------------------------------


def make_feigcooling_selector(X_bits: np.ndarray, bank: pd.DataFrame = None):
    """
    GRAM_FeigCooling selector using HOSAM with Feigenbaum-cooled NEAR dither.
    Falls back to Hamming NN if query is not KNOWN.

    X_bits: (P, N) binary patterns used by HOSAM.
    bank:   optional DataFrame; not used here, but kept for API symmetry.
    """
    mem = HOSAM(X_bits)

    def selector(q_bits: np.ndarray) -> int:
        # Primary: HOSAM query
        try:
            dec, idx, meta = mem.query(q_bits)
            if dec == "KNOWN" and idx is not None:
                return int(idx)
        except Exception:
            pass
        # Fallback: nearest by Hamming distance
        d_all = np.sum(np.abs(X_bits - q_bits), axis=1)
        return int(np.argmin(d_all))

    return selector

# ------------------------------------------------------------------
# One-seed run
# ------------------------------------------------------------------


def run_one_seed(df_full: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    outdir = os.path.join(OUT_ROOT, f"seed{seed}")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Build objective + feasibility
    df, feas_mask = build_objective_and_feasibility(df_full)

    # Feasible bank
    bank = df[feas_mask].copy().reset_index(drop=True)
    bank["feasible"] = True
    print(f"[seed {seed}] Feasible rows: {len(bank)}/{len(df)} "
          f"({100.0 * len(bank) / max(1, len(df)):.1f}%)")

    # Bounds for GRNN input normalization
    pv_min_full, pv_max_full     = df[PV_COL].min(),   df[PV_COL].max()
    bess_min_full, bess_max_full = df[BESS_COL].min(), df[BESS_COL].max()

    def normalize_pv_bess(pv, bess):
        pv_norm   = (pv   - pv_min_full)   / (pv_max_full   - pv_min_full   + 1e-12)
        bess_norm = (bess - bess_min_full) / (bess_max_full - bess_min_full + 1e-12)
        return np.array([pv_norm, bess_norm], dtype=float)

    # Encode bank for associative memories
    X_bits, encode_row = encode_bank(bank, df)

    # ------------ GRAM variants ------------
    gram_plain   = GRAM_plain(X_bits)
    gram_phiJ    = GRAM_phiJ(X_bits)
    gram_phiFull = GRAM_phiFull(X_bits)

    # ------------ Recursive wrappers: FS + CoarseFine ------------
    P, N_bits = X_bits.shape
    mask_first_half = np.zeros(N_bits, dtype=bool)
    mask_first_half[:N_bits // 2] = True

    fs_phiFull = GRAMFeatureSelector(mask=mask_first_half, gram_ctor=GRAM_phiFull)
    fs_phiFull.fit(X_bits)

    cf_phiFull = GRAMCoarseFine(
        gram_ctor=GRAM_phiFull,
        gram_kwargs_coarse={},
        gram_kwargs_fine={},
    )

    import types
    def encode_coarse_fn(self, X):
        return X[:, :X.shape[1] // 2]
    cf_phiFull.encode_coarse = types.MethodType(encode_coarse_fn, cf_phiFull)
    cf_phiFull.fit(X_bits)

    # ------------ GRAM_FeigCooling (HOSAM-based) ------------
    feig_sel = make_feigcooling_selector(X_bits, bank)

    # ------------ Baselines: Hopfield, MHN, DAM, GB-AM, Willshaw, SDM ------------
    W_hop = gram_plain.W

    def hopfield_minimize(y0_bits, W, max_iters=50):
        y = (y0_bits.astype(np.int8) * 2 - 1).copy()
        Nloc = y.shape[0]
        for it in range(max_iters):
            changed = 0
            for i in range(Nloc):
                h = int(W[i] @ y)
                new = 1 if h >= 0 else -1
                if new != y[i]:
                    y[i] = new
                    changed += 1
            if changed == 0:
                return ((y > 0).astype(np.uint8)), (it + 1)
        return ((y > 0).astype(np.uint8)), max_iters

    # GRNN training data
    X_grnn = np.column_stack([
        (bank[PV_COL].to_numpy()   - pv_min_full)   / (pv_max_full   - pv_min_full   + 1e-12),
        (bank[BESS_COL].to_numpy() - bess_min_full) / (bess_max_full - bess_min_full + 1e-12),
    ])
    Y_grnn = bank[["LCC_norm", SSR_COL, SCR_COL, COST_COL]].to_numpy()
    grnn_model = GRNN(X_train=X_grnn, Y_train=Y_grnn, sigma=0.15)

    # Query set
    df_sample = df.sample(n=min(Q, len(df)), random_state=seed).reset_index(drop=True)

    # ---------- Selector wrappers ----------

    def gram_select(gram, q_bits):
        dec, idx, meta = gram.query(q_bits)
        if dec == "KNOWN" and idx is not None:
            return int(idx)
        d_all = np.sum(np.abs(X_bits - q_bits), axis=1)
        M = min(M_CAND, len(bank))
        cand_idx = np.argpartition(d_all, M - 1)[:M]
        cand = bank.iloc[cand_idx].copy()
        feas = cand[cand["feasible"]]
        if len(feas):
            return int(feas["J"].idxmin())
        cand["J_pen"] = cand["J"] + cand.apply(feasibility_penalty, axis=1)
        return int(cand["J_pen"].idxmin())

    def gram_fs_phiFull_select(q_bits):
        dec, idx, meta = fs_phiFull.query(q_bits)
        if dec == "KNOWN" and idx is not None:
            return int(idx)
        return gram_select(gram_phiFull, q_bits)

    def gram_cf_phiFull_select(q_bits):
        dec, idx, meta = cf_phiFull.query(q_bits)
        if dec == "KNOWN" and idx is not None:
            return int(idx)
        return gram_select(gram_phiFull, q_bits)

    def hopfield_select(q_bits):
        y_bits, _ = hopfield_minimize(q_bits, W_hop, max_iters=50)
        d = np.sum(np.abs(X_bits - y_bits), axis=1)
        return int(np.argmin(d))

    def mhn_best_select(q_bits):
        bestJ = np.inf
        bestK = None
        for beta in MHN_BETAS:
            m = ModernHopfieldAttention(X_bits, beta=beta)
            k = m.retrieve(q_bits)
            sc = float(bank.iloc[k]["J"])
            if sc < bestJ:
                bestJ, bestK = sc, k
        return int(bestK)

    def dam_best_select(q_bits):
        bestJ = np.inf
        bestK = None
        for n in DAM_ORDERS:
            for s in DAM_STEPS:
                dmodel = DenseAssociativeMemory(X_bits, n_order=n, steps=s)
                k = dmodel.retrieve(q_bits)
                sc = float(bank.iloc[k]["J"])
                if sc < bestJ:
                    bestJ, bestK = sc, k
        return int(bestK)

    def gb_select(q_bits):
        C = 8 if X_bits.shape[1] >= 32 else 4
        return GriponBerrouAM(X_bits, C=C).retrieve(q_bits)

    def cmm_select(q_bits):
        return WillshawCMM(X_bits, thresh_frac=0.3).retrieve(q_bits)

    def sdm_select(q_bits):
        R = int(0.2 * X_bits.shape[1])
        return SparseDistributedMemory(X_bits, L=256, R=R, seed=seed).retrieve(q_bits)

    # ---------- Evaluation helpers ----------

    def run_selector(name, selector_fn):
        scores = []; costs = []; ssrs = []; scrs = []; t = 0.0
        for i in range(len(df_sample)):
            row = df_sample.iloc[i]
            q_bits = encode_row(row)
            t0 = time.perf_counter()
            k = selector_fn(q_bits)
            t += (time.perf_counter() - t0)
            chosen = bank.iloc[k]
            scores.append(float(chosen["J"]))
            costs.append(float(chosen[COST_COL]))
            ssrs.append(float(chosen[SSR_COL]))
            scrs.append(float(chosen.get(SCR_COL, 0.0)))
        return dict(
            name=name,
            J=np.array(scores), cost=np.array(costs),
            ssr=np.array(ssrs), scr=np.array(scrs), t=t,
        )

    def run_grnn(name, grnn_model):
        scores = []; costs = []; ssrs = []; scrs = []; t = 0.0
        for i in range(len(df_sample)):
            row = df_sample.iloc[i]
            pv   = float(row[PV_COL])
            bess = float(row[BESS_COL])
            x_norm = normalize_pv_bess(pv, bess)

            t0 = time.perf_counter()
            y_hat = grnn_model.predict(x_norm)
            t += (time.perf_counter() - t0)

            lcc_norm_pred = float(y_hat[0])
            ssr_pred      = float(y_hat[1])
            scr_pred      = float(y_hat[2])
            cost_pred     = float(y_hat[3])

            # NEW GRNN objective: J_pred = LCC_norm_pred + (1 - SSR_pred)
            J_pred = lcc_norm_pred + (1.0 - ssr_pred)

            # Apply same feasibility penalty as AMs
            pen = feasibility_penalty_values(ssr_pred, pv, bess)
            J_eff = J_pred + pen

            scores.append(J_eff)
            costs.append(cost_pred)
            ssrs.append(ssr_pred)
            scrs.append(scr_pred)

        return dict(
            name=name,
            J=np.array(scores), cost=np.array(costs),
            ssr=np.array(ssrs), scr=np.array(scrs), t=t,
        )

    # ---------- Run all methods ----------

    results = []
    results.append(run_selector("GRAM_plain",        lambda q: gram_select(gram_plain,   q)))
    results.append(run_selector("GRAM_phiJ",         lambda q: gram_select(gram_phiJ,    q)))
    results.append(run_selector("GRAM_phiFull",      lambda q: gram_select(gram_phiFull, q)))
    results.append(run_selector("GRAM_FS_phiFull",   gram_fs_phiFull_select))
    results.append(run_selector("GRAM_FeigCooling",  feig_sel))
    results.append(run_selector("GRAM_CF_phiFull",   gram_cf_phiFull_select))
    results.append(run_selector("Hopfield",          hopfield_select))
    results.append(run_selector("MHN",               mhn_best_select))
    results.append(run_selector("DAM",               dam_best_select))
    results.append(run_selector("GB-AM",             gb_select))
    results.append(run_selector("Willshaw",          cmm_select))
    results.append(run_selector("SDM",               sdm_select))
    results.append(run_grnn("GRNN",                  grnn_model))

    # Per-seed summary
    print(
        f"\n=== Seed {seed} — Constraint-aware Objective "
        f"(Q = {len(df_sample)}, feasible P = {len(bank)}, bits = {X_bits.shape[1]}) ==="
    )
    print("{:15s}  {:>8s}  {:>8s}  {:>10s}  {:>8s}  {:>8s}  {:>10s}".format(
        "Method", "J-mean", "J-med", "Cost-mean", "SSR", "SCR", "ms/query"
    ))
    for R in results:
        print("{:15s}  {:8.4f}  {:8.4f}  {:10.2f}  {:8.3f}  {:8.3f}  {:10.3f}".format(
            R["name"], np.mean(R["J"]), np.median(R["J"]),
            np.mean(R["cost"]), np.mean(R["ssr"]), np.mean(R["scr"]),
            1e3 * R["t"] / len(df_sample),
        ))

    # Save per-seed CSV
    seed_dir = Path(outdir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    with open(seed_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "J_mean", "J_median", "cost_mean",
            "ssr_mean", "scr_mean", "ms_per_query",
        ])
        for R in results:
            w.writerow([
                R["name"], np.mean(R["J"]), np.median(R["J"]),
                np.mean(R["cost"]), np.mean(R["ssr"]), np.mean(R["scr"]),
                1e3 * R["t"] / len(df_sample),
            ])

    # 2D/3D plots

    names_to_plot = [
        "GRAM_plain", "GRAM_phiJ", "GRAM_phiFull",
        "GRAM_FS_phiFull", "GRAM_FeigCooling", "GRAM_CF_phiFull",
        "Hopfield", "MHN", "DAM", "GRNN",
    ]
    color_map = {
        "GRAM_plain":        "#1f77b4",
        "GRAM_phiJ":         "#17becf",
        "GRAM_phiFull":      "#ff7f0e",
        "GRAM_FS_phiFull":   "#9467bd",
        "GRAM_FeigCooling":  "#2ca02c",
        "GRAM_CF_phiFull":   "#8c564b",
        "Hopfield":          "#d62728",
        "MHN":               "#2ca02c",
        "DAM":               "#7f7f7f",
        "GB-AM":             "#bcbd22",
        "Willshaw":          "#17becf",
        "SDM":               "#e377c2",
        "GRNN":              "#ff9896",
    }

    # Cost vs SSR
    plt.figure(figsize=(7, 5))
    plt.scatter(df[COST_COL], df[SSR_COL], s=6, c="#cccccc", label="All designs")
    for R in results:
        if R["name"] in names_to_plot:
            plt.scatter(
                R["cost"], R["ssr"], s=18,
                c=color_map.get(R["name"], "#000000"),
                marker="o", label=R["name"]
            )
    plt.axhline(SSR_MIN, color="red", linestyle="--", linewidth=0.8,
                label=f"SSR ≥ {SSR_MIN}")
    plt.xlabel("Present Cost")
    plt.ylabel("SSR")
    plt.title(f"Cost vs SSR (feasible bank + GRNN, seed={seed})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Cost vs SCR
    if SCR_COL in df.columns:
        plt.figure(figsize=(7, 5))
        plt.scatter(df[COST_COL], df[SCR_COL], s=6, c="#cccccc",
                    label="All designs")
        for R in results:
            if R["name"] in names_to_plot:
                plt.scatter(
                    R["cost"], R["scr"], s=18,
                    c=color_map.get(R["name"], "#000000"),
                    marker="o", label=R["name"]
                )
        plt.xlabel("Present Cost")
        plt.ylabel("SCR")
        plt.title(f"Cost vs SCR (feasible bank + GRNN, seed={seed})")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # SSR vs SCR
        plt.figure(figsize=(7, 5))
        plt.scatter(df[SSR_COL], df[SCR_COL], s=6, c="#cccccc",
                    label="All designs")
        for R in results:
            if R["name"] in names_to_plot:
                plt.scatter(
                    R["ssr"], R["scr"], s=18,
                    c=color_map.get(R["name"], "#000000"),
                    marker="o", label=R["name"]
                )
        plt.xlabel("SSR")
        plt.ylabel("SCR")
        plt.title(f"SSR vs SCR (feasible bank + GRNN, seed={seed})")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 3D plot
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            df[COST_COL], df[SSR_COL], df[SCR_COL],
            s=5, c="#cccccc", alpha=0.35, label="All designs"
        )
        for nm in names_to_plot:
            R = [r for r in results if r["name"] == nm][0]
            ax.scatter(R["cost"], R["ssr"], R["scr"], s=28, label=nm)
        ax.set_xlabel("Cost")
        ax.set_ylabel("SSR")
        ax.set_zlabel("SCR")
        ax.set_title(f"3D frontier (feasible bank + GRNN, seed={seed})")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return results

# ------------------------------------------------------------------
# Aggregate across seeds
# ------------------------------------------------------------------


def ci95(arr):
    arr = np.asarray(arr, float)
    n = len(arr)
    if n < 2:
        m = float(np.mean(arr)) if n > 0 else float("nan")
        return m, (float("nan"), float("nan"))
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    hw = 1.96 * s / (n ** 0.5)
    return m, (m - hw, m + hw)


def main():
    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
    df_full = pd.read_csv(CSV_PATH)

    methods = [
        "GRAM_plain", "GRAM_phiJ", "GRAM_phiFull",
        "GRAM_FS_phiFull", "GRAM_FeigCooling", "GRAM_CF_phiFull",
        "Hopfield", "MHN", "DAM", "GB-AM",
        "Willshaw", "SDM", "GRNN",
    ]

    per_method_Jmean = {m: [] for m in methods}
    per_method_Jmed  = {m: [] for m in methods}
    per_method_cost  = {m: [] for m in methods}
    per_method_ssr   = {m: [] for m in methods}
    per_method_scr   = {m: [] for m in methods}
    per_method_ms    = {m: [] for m in methods}

    for seed in SEEDS:
        results = run_one_seed(df_full.copy(), seed)
        for R in results:
            name = R["name"]
            if name not in methods:
                continue

            j_vals   = np.asarray(R["J"],    float)
            c_vals   = np.asarray(R["cost"], float)
            ssr_vals = np.asarray(R["ssr"],  float)
            scr_vals = np.asarray(R["scr"],  float)
            ms_val   = 1e3 * R["t"] / len(j_vals) if len(j_vals) > 0 else float("nan")

            per_method_Jmean[name].append(float(np.mean(j_vals)))
            per_method_Jmed[name].append(float(np.median(j_vals)))
            per_method_cost[name].append(float(np.mean(c_vals)))
            per_method_ssr[name].append(float(np.mean(ssr_vals)))
            per_method_scr[name].append(float(np.mean(scr_vals)))
            per_method_ms[name].append(float(ms_val))

    print("\n=== Aggregate across seeds (mean ± 95% CI for J, Cost, SSR, SCR, runtime) ===")
    header = (
        "Method",
        "J mean (95% CI)",
        "J median (95% CI)",
        "Cost mean (95% CI)",
        "SSR mean (95% CI)",
        "SCR mean (95% CI)",
        "ms/query (95% CI)",
    )
    print("{:15s}  {:>26s}  {:>26s}  {:>26s}  {:>26s}  {:>26s}  {:>26s}".format(*header))

    rows = []

    for name in methods:
        Jm_vals   = per_method_Jmean[name]
        Jmed_vals = per_method_Jmed[name]
        C_vals    = per_method_cost[name]
        SSR_vals  = per_method_ssr[name]
        SCR_vals  = per_method_scr[name]
        MS_vals   = per_method_ms[name]

        if not Jm_vals:
            continue

        Jm,   (Jm_lo,   Jm_hi)   = ci95(Jm_vals)
        Jmed, (Jmed_lo, Jmed_hi) = ci95(Jmed_vals)
        Cm,   (C_lo,    C_hi)    = ci95(C_vals)
        SSRm, (SSR_lo,  SSR_hi)  = ci95(SSR_vals)
        SCRm, (SCR_lo,  SCR_hi)  = ci95(SCR_vals)
        MSm,  (MS_lo,   MS_hi)   = ci95(MS_vals)

        print(
            "{:15s}  {:>26s}  {:>26s}  {:>26s}  {:>26s}  {:>26s}  {:>26s}".format(
                name,
                f"{Jm:.4f} ({Jm_lo:.4f},{Jm_hi:.4f})",
                f"{Jmed:.4f} ({Jmed_lo:.4f},{Jmed_hi:.4f})",
                f"{Cm:.2f} ({C_lo:.2f},{C_hi:.2f})",
                f"{SSRm:.3f} ({SSR_lo:.3f},{SSR_hi:.3f})",
                f"{SCRm:.3f} ({SCR_lo:.3f},{SCR_hi:.3f})",
                f"{MSm:.3f} ({MS_lo:.3f},{MS_hi:.3f})",
            )
        )

        rows.append([
            name,
            Jm,   Jm_lo,   Jm_hi,
            Jmed, Jmed_lo, Jmed_hi,
            Cm,   C_lo,    C_hi,
            SSRm, SSR_lo,  SSR_hi,
            SCRm, SCR_lo,  SCR_hi,
            MSm,  MS_lo,   MS_hi,
        ])

    out_csv = os.path.join(OUT_ROOT, "aggregate_summary_newGRAM.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method",
            "J_mean",    "J_mean_CI_lo",    "J_mean_CI_hi",
            "J_median",  "J_median_CI_lo",  "J_median_CI_hi",
            "Cost_mean", "Cost_CI_lo",      "Cost_CI_hi",
            "SSR_mean",  "SSR_CI_lo",       "SSR_CI_hi",
            "SCR_mean",  "SCR_CI_lo",       "SCR_CI_hi",
            "ms_per_query_mean", "ms_per_query_CI_lo", "ms_per_query_CI_hi",
        ])
        w.writerows(rows)

    print(f"\nAggregate summary written to: {out_csv}")


if __name__ == "__main__":
    main()
