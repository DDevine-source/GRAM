# %%writefile plot_multiobj_Jmedian_distances_no_GRNN.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AGG_CSV = "am_constraint_objective_seeds_out_newGRAM/aggregate_summary_newGRAM.csv"

METHODS = [
    "GRAM_plain",
    "GRAM_phiFull",
    "GRAM_FS_phiFull",
    "GRAM_FeigCooling",
    "GRAM_CF_phiFull",
    "Hopfield",
    "MHN",
    "DAM",
    "GB-AM",
]

COLOR_MAP = {
    "GRAM_plain":    "#1f77b4",
    "GRAM_phiFull":  "#ff7f0e",
    "GRAM_FS_phiFull":"#9467bd",
    "GRAM_FeigCooling":"#ffbb78",
    "GRAM_CF_phiFull":"#8c564b",
    "Hopfield":      "#d62728",
    "MHN":           "#2ca02c",
    "DAM":           "#7f7f7f",
    "GB-AM":         "#bcbd22",
}

OUT_DIR = "plot_multiobj_Jmedian_distances_no_GRNN_out"

def normalize(x):
    x = np.asarray(x, float)
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(AGG_CSV)

    # Rename to consistent names if needed
    rename_map = {
        "J_median": "J_median",
        "Cost_mean":"cost_mean",
        "SSR_mean": "ssr_mean",
        "SCR_mean": "scr_mean",
    }
    for old, new in rename_map.items():
        if old in df.columns and new != old:
            df = df.rename(columns={old: new})

    df = df[df["method"].isin(METHODS)].copy()

    print("Aggregate (multi-objective, J_median):")
    print(df[["method", "J_median", "cost_mean", "ssr_mean", "scr_mean"]])

    methods = df["method"].values
    Jmed    = df["J_median"].values
    cost    = df["cost_mean"].values
    ssr     = df["ssr_mean"].values

    cost_norm = normalize(cost)

    # 1) Cost vs SSR distances
    d_cost_ssr = np.sqrt(cost_norm**2 + (1.0 - ssr)**2)
    # 2) J_median vs Cost distances
    d_J_cost   = np.sqrt(Jmed**2 + cost_norm**2)
    # 3) J_median vs SSR distances
    d_J_ssr    = np.sqrt(Jmed**2 + (1.0 - ssr)**2)

    # Helper to make a sorted bar plot
    def plot_bar(distances, title, filename, ylabel):
        sort_idx = np.argsort(distances)
        methods_sorted = methods[sort_idx]
        d_sorted = distances[sort_idx]

        plt.figure(figsize=(8,5))
        bars = plt.bar(range(len(methods_sorted)), d_sorted,
                       color=[COLOR_MAP.get(m, "gray") for m in methods_sorted])
        plt.xticks(range(len(methods_sorted)), methods_sorted, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, filename)
        plt.savefig(out_path, dpi=150)
        plt.show()
        print(f"Saved {title} bar plot to {out_path}")

    print("\nDistances (Cost vs SSR):")
    for m, d in zip(methods, d_cost_ssr):
        print(f"{m:15s}: {d:.4f}")
    plot_bar(d_cost_ssr,
             "Distance to utopia (J_median, Cost vs SSR)",
             "dist_cost_vs_SSR.png",
             "Distance to (cost_norm=0, SSR=1)")

    print("\nDistances (J_median vs Cost):")
    for m, d in zip(methods, d_J_cost):
        print(f"{m:15s}: {d:.4f}")
    plot_bar(d_J_cost,
             "Distance to utopia (J_median vs Cost)",
             "dist_Jmedian_vs_cost.png",
             "Distance to (J_median=0, cost_norm=0)")

    print("\nDistances (J_median vs SSR):")
    for m, d in zip(methods, d_J_ssr):
        print(f"{m:15s}: {d:.4f}")
    plot_bar(d_J_ssr,
             "Distance to utopia (J_median vs SSR)",
             "dist_Jmedian_vs_SSR.png",
             "Distance to (J_median=0, SSR=1)")

if __name__ == "__main__":
    main()
