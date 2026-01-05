# %%writefile analyze_objective_2D_multiobj_Jmedian_no_GRNN.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

# Path to the aggregate summary produced by the multi-objective run:
# J = LCC_norm + (1 - SSR) + (1 - SCR)
AGG_CSV = "am_constraint_objective_seeds_out_newGRAM/aggregate_summary_newGRAM.csv"

# Methods to include (no GRNN)
METHODS = [
    "GRAM_phiFull",
    "GRAM_CF_phiFull",
    "GRAM_FS_phiFull",
    "GRAM_FeigCooling",
    "GRAM_plain",
    "Hopfield",
    "MHN",
    "DAM",
    "GB-AM",
]

COLOR_MAP = {
    "GRAM_phiFull":    "#ff7f0e",
    "GRAM_CF_phiFull": "#8c564b",
    "GRAM_FS_phiFull": "#9467bd",
    "GRAM_FeigCooling":"#ffbb78",
    "GRAM_plain":      "#1f77b4",
    "Hopfield":        "#d62728",
    "MHN":             "#2ca02c",
    "DAM":             "#7f7f7f",
    "GB-AM":           "#bcbd22",
}

OUT_DIR = "analyze_objective_2D_multiobj_Jmedian_no_GRNN_out"

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def normalize(x):
    x = np.asarray(x, float)
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(AGG_CSV)

    # Ensure consistent names
    rename_map = {
        "J_median": "J_median",
        "Cost_mean": "cost_mean",
        "SSR_mean":  "ssr_mean",
        "SCR_mean":  "scr_mean",
    }
    for old, new in rename_map.items():
        if old in df.columns and new != old:
            df = df.rename(columns={old: new})

    # Filter to methods of interest
    df = df[df["method"].isin(METHODS)].copy()

    print("Aggregate summary (multi-objective J_median, no GRNN):")
    print(df[["method", "J_median", "cost_mean", "ssr_mean", "scr_mean"]])

    methods = df["method"].values
    cost    = df["cost_mean"].values
    ssr     = df["ssr_mean"].values
    Jmed    = df["J_median"].values

    cost_norm = normalize(cost)

    # -----------------------------
    # 1) Cost vs SSR
    # -----------------------------
    d_cost_ssr = np.sqrt(cost_norm**2 + (1.0 - ssr)**2)
    knee_idx_1 = int(np.argmin(d_cost_ssr))
    knee_method_1 = methods[knee_idx_1]

    print("\n[Multi-objective J_median] Cost vs SSR distances to utopia (cost_norm=0, SSR=1):")
    for m, d in zip(methods, d_cost_ssr):
        print(f"{m:15s}: distance = {d:.4f}")
    print(f"--> Knee method (Cost vs SSR): {knee_method_1}")

    plt.figure(figsize=(7,5))
    for i, m in enumerate(methods):
        plt.scatter(cost[i], ssr[i], c=COLOR_MAP.get(m, "black"))
        plt.text(cost[i], ssr[i], " " + m, fontsize=8, va="center", ha="left")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(cost.min(), color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Cost_mean")
    plt.ylabel("SSR_mean")
    plt.title("Cost_mean vs SSR_mean (multi-objective J_median)")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "cost_vs_SSR.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved Cost vs SSR plot to {out_path}")

    # -----------------------------
    # 2) J_median vs Cost
    # -----------------------------
    d_J_cost = np.sqrt(Jmed**2 + cost_norm**2)
    knee_idx_2 = int(np.argmin(d_J_cost))
    knee_method_2 = methods[knee_idx_2]

    print("\n[Multi-objective J_median] J_median vs Cost distances to utopia (J_median=0, cost_norm=0):")
    for m, d in zip(methods, d_J_cost):
        print(f"{m:15s}: distance = {d:.4f}")
    print(f"--> Knee method (J_median vs Cost): {knee_method_2}")

    plt.figure(figsize=(7,5))
    for i, m in enumerate(methods):
        plt.scatter(cost[i], Jmed[i], c=COLOR_MAP.get(m, "black"))
        plt.text(cost[i], Jmed[i], " " + m, fontsize=8, va="center", ha="left")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(cost.min(), color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Cost_mean")
    plt.ylabel("J_median")
    plt.title("J_median vs Cost_mean (multi-objective J)")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "Jmedian_vs_cost.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved J_median vs Cost plot to {out_path}")

    # -----------------------------
    # 3) J_median vs SSR
    # -----------------------------
    d_J_ssr = np.sqrt(Jmed**2 + (1.0 - ssr)**2)
    knee_idx_3 = int(np.argmin(d_J_ssr))
    knee_method_3 = methods[knee_idx_3]

    print("\n[Multi-objective J_median] J_median vs SSR distances to utopia (J_median=0, SSR=1):")
    for m, d in zip(methods, d_J_ssr):
        print(f"{m:15s}: distance = {d:.4f}")
    print(f"--> Knee method (J_median vs SSR): {knee_method_3}")

    plt.figure(figsize=(7,5))
    for i, m in enumerate(methods):
        plt.scatter(ssr[i], Jmed[i], c=COLOR_MAP.get(m, "black"))
        plt.text(ssr[i], Jmed[i], " " + m, fontsize=8, va="center", ha="left")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("SSR_mean")
    plt.ylabel("J_median")
    plt.title("J_median vs SSR_mean (multi-objective J)")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "Jmedian_vs_SSR.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved J_median vs SSR plot to {out_path}")

if __name__ == "__main__":
    main()
