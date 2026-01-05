# %%writefile compare_surfaces_metrics.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- CONFIG (match manifold_minimal_surfaces.py) ---------

CSV_PATH   = "/content/drive/My Drive/Sweep_final.csv"

PV_COL     = "nPV"
BESS_COL   = "EBatMax [kW.h]"
COST_COL   = "lcc.y"
SSR_COL    = "ssr.y"
SCR_COL    = "scr.y"

PV_MAX     = 18500
BESS_MAX   = 36400

SSR_MIN    = 0.40

OUT_DIR    = "manifold_minimal_out"
SUMMARY_CSV = os.path.join(OUT_DIR, "minimal_surfaces_summary.csv")

# Methods and targets we want to compare
METHODS        = ["GRAM_phiFull", "GRAM_FS_phiFull", "Hopfield", "DAM", "GB-AM", "MHN", "GRAM_FeigCooling"]
TARGET_UNIQUES = [150, 230]

# Grid resolution for comparison
GRID_N = 75  # can increase to 100 for finer metrics


# --------- Utilities copied from original script (minimal) ---------

def build_feasible_bank(df_full: pd.DataFrame, ssr_min: float) -> pd.DataFrame:
    df = df_full.copy()
    for col in [PV_COL, BESS_COL, COST_COL, SSR_COL, SCR_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[SCR_COL] = df[SCR_COL].clip(0.0, 1.0)
    df = df.dropna(subset=[PV_COL, BESS_COL, COST_COL, SSR_COL]).reset_index(drop=True)

    cmin, cmax = df[COST_COL].min(), df[COST_COL].max()
    df["LCC_norm"] = (df[COST_COL] - cmin) / (cmax - cmin + 1e-12)
    df["J"] = df["LCC_norm"] + (1.0 - df[SSR_COL]) + (1.0 - df[SCR_COL])

    feas_mask = (
        (df[SSR_COL] >= ssr_min) &
        (df[PV_COL] <= PV_MAX) &
        (df[BESS_COL] <= BESS_MAX)
    )
    bank = df.loc[feas_mask].copy().reset_index(drop=True)
    print(f"[COMPARE] Feasible rows at SSR_MIN={ssr_min}: {len(bank)}/{len(df)}")
    return bank


def eval_surface(beta, SSR, SCR):
    """
    Cost = b0 + b1*SSR + b2*SCR + b3*SSR^2 + b4*SCR^2 + b5*SSR*SCR
    """
    b0, b1, b2, b3, b4, b5 = beta
    return (b0 + b1*SSR + b2*SCR + b3*SSR**2 + b4*SCR**2 + b5*SSR*SCR)


def get_hessian_from_beta(beta):
    """
    Hessian wrt [SSR, SCR] for quadratic polynomial.
    d^2C/dSSR^2 = 2*b3
    d^2C/dSCR^2 = 2*b4
    d^2C/dSSR dSCR = b5
    """
    _, _, _, b3, b4, b5 = beta
    H = np.array([[2.0*b3,    b5],
                  [   b5, 2.0*b4]])
    return H


def principal_curvatures_from_beta(beta):
    """
    Principal curvatures = eigenvalues of Hessian matrix.
    """
    H = get_hessian_from_beta(beta)
    vals, _ = np.linalg.eig(H)
    return np.sort(vals)  # ascending order


# --------- MAIN ANALYSIS ---------

def main():
    # Load the summary of surface fits
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(
            f"Could not find {SUMMARY_CSV}. "
            "Make sure you ran manifold_minimal_surfaces.py first."
        )
    df_summary = pd.read_csv(SUMMARY_CSV)

    # Load full data and build the same feasible bank (for SSR/SCR ranges)
    df_full = pd.read_csv(CSV_PATH)
    bank = build_feasible_bank(df_full, SSR_MIN)

    ssr_min_val, ssr_max_val = bank[SSR_COL].min(), bank[SSR_COL].max()
    scr_min_val, scr_max_val = bank[SCR_COL].min(), bank[SCR_COL].max()

    print("\n[COMPARE] SSR range in bank:", ssr_min_val, ssr_max_val)
    print("[COMPARE] SCR range in bank:", scr_min_val, scr_max_val)

    # Build the global grid used to compare all surfaces
    ssr_grid = np.linspace(ssr_min_val, ssr_max_val, GRID_N)
    scr_grid = np.linspace(scr_min_val, scr_max_val, GRID_N)
    SSRg, SCRg = np.meshgrid(ssr_grid, scr_grid)

    # Extract TRUE beta (only one row for method == 'TRUE' or where we saved it)
    # In current script, TRUE surface parameters weren't stored in CSV,
    # so we re-fit here on the full bank, just like manifold_minimal_surfaces did.
    # If already saved beta_true, can load it instead.
    print("\n[COMPARE] Re-fitting TRUE surface on full usable bank...")
    S = bank[SSR_COL].to_numpy(dtype=float)
    H = bank[SCR_COL].to_numpy(dtype=float)
    C = bank[COST_COL].to_numpy(dtype=float)
    X = np.stack([np.ones_like(S), S, H, S**2, H**2, S*H], axis=1)
    beta_true, *_ = np.linalg.lstsq(X, C, rcond=None)
    C_hat_true = X @ beta_true
    ss_tot = np.sum((C - C.mean())**2)
    ss_res = np.sum((C - C_hat_true)**2)
    R2_true = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print("[COMPARE] TRUE beta:", beta_true)
    print("[COMPARE] TRUE R² :", R2_true)

    # Evaluate TRUE surface on the grid
    Z_true = eval_surface(beta_true, SSRg, SCRg)

    # Prepare a results table for metrics
    rows_metrics = []

    for target in TARGET_UNIQUES:
        print(f"\n================ Target_unique = {target} ================")

        # Loop over each method
        for method in METHODS:
            sub = df_summary[
                (df_summary["method"] == method) &
                (df_summary["target_unique"] == target)
            ]
            if sub.empty:
                print(f"[COMPARE] No summary row for {method}, target={target}, skipping.")
                continue

            # Grab the betas from manifold_minimal_surfaces script
            beta = np.array([
                sub["b0"].item(),
                sub["b1"].item(),
                sub["b2"].item(),
                sub["b3"].item(),
                sub["b4"].item(),
                sub["b5"].item(),
            ])
            R2_m = float(sub["R2"].item())
            n_used = int(sub["used_unique"].item())

            print(f"\n[COMPARE] Method={method}, target={target}")
            print("  used_unique =", n_used)
            print("  R2 (method) =", R2_m)

            # Evaluate method surface on the same grid
            Z_m = eval_surface(beta, SSRg, SCRg)

            # ---------- 1. RMSE & manifold distance metrics ----------

            diff = Z_m - Z_true
            mse = np.mean(diff**2)
            rmse = np.sqrt(mse)
            max_abs = np.max(np.abs(diff))
            l2_norm = np.sqrt(np.sum(diff**2))

            print("  RMSE(method vs TRUE)       =", rmse)
            print("  Max abs error vs TRUE      =", max_abs)
            print("  L2 norm of surface diff    =", l2_norm)

            # ---------- 2. Curvature comparison (Hessian eigenvalues) ----------

            k_true = principal_curvatures_from_beta(beta_true)
            k_m    = principal_curvatures_from_beta(beta)

            # Relative error in each principal curvature
            # (handle near-zero safely)
            rel_errs = []
            for kt, km in zip(k_true, k_m):
                denom = (np.abs(kt) + 1e-12)
                rel_errs.append(np.abs(km - kt) / denom)

            mean_rel_curv_err = float(np.mean(rel_errs))
            max_rel_curv_err  = float(np.max(rel_errs))

            print("  TRUE principal curvatures   =", k_true)
            print("  method principal curvatures =", k_m)
            print("  mean rel curvature error    =", mean_rel_curv_err)
            print("  max  rel curvature error    =", max_rel_curv_err)

            # ---------- 3. Save heatmaps of ΔCost = method - TRUE ----------

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(
                diff,
                origin="lower",
                extent=[ssr_min_val, ssr_max_val, scr_min_val, scr_max_val],
                aspect="auto"
            )
            ax.set_xlabel("SSR")
            ax.set_ylabel("SCR")
            ax.set_title(f"ΔCost = {method} - TRUE (N={target})")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Cost difference")

            fname = os.path.join(
                OUT_DIR,
                f"heatmap_dCost_{method}_N{target}_SSRmin{SSR_MIN}.png"
            )
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  Saved heatmap to {fname}")

            # Store metrics row
            rows_metrics.append({
                "method": method,
                "target_unique": target,
                "used_unique": n_used,
                "R2_true": float(R2_true),
                "R2_method": float(R2_m),
                "RMSE_vs_TRUE": float(rmse),
                "MaxAbs_vs_TRUE": float(max_abs),
                "L2norm_vs_TRUE": float(l2_norm),
                "k1_true": float(k_true[0]),
                "k2_true": float(k_true[1]),
                "k1_method": float(k_m[0]),
                "k2_method": float(k_m[1]),
                "mean_rel_curv_err": mean_rel_curv_err,
                "max_rel_curv_err": max_rel_curv_err,
            })

    # Save all metrics to CSV
    df_metrics = pd.DataFrame(rows_metrics)
    metrics_csv = os.path.join(OUT_DIR, "surface_comparison_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\n[COMPARE] Saved metrics table to {metrics_csv}")


if __name__ == "__main__":
    main()
