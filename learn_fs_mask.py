#%%writefile learn_fs_mask.py
import numpy as np
import pandas as pd
import os

# Path to the bit-level importance file produced by analyze_feature_importance_gram.py
BITS_CSV = "feature_importance_out/gram_feature_importance.csv"

OUT_DIR  = "feature_importance_out"
OUT_PATH = os.path.join(OUT_DIR, "learned_fs_mask.npy")

# Method and SSR_MIN to base the FS mask on
METHOD   = "GRAM_phiFull"
SSR_MIN  = 0.40

# Fraction of bits to keep (0.5 = top 50% of bits by positive importance)
MASK_FRACTION = 0.5

def main():
    if not os.path.exists(BITS_CSV):
        raise FileNotFoundError(f"{BITS_CSV} not found. Run analyze_feature_importance_gram.py first.")

    df = pd.read_csv(BITS_CSV)

    df_m = df[(df["method"] == METHOD) & (df["SSR_MIN"] == SSR_MIN)].copy()
    if df_m.empty:
        raise RuntimeError(f"No rows for method={METHOD!r} and SSR_MIN={SSR_MIN} in {BITS_CSV}")

    # Extract bit indices and importances
    bit_idx = df_m["bit_index"].to_numpy()
    imp     = df_m["importance"].to_numpy()

    # Only consider positive contributions
    imp_pos = np.where(imp > 0, imp, 0.0)

    # How many bits to keep?
    n_bits = len(bit_idx)
    k = max(1, int(MASK_FRACTION * n_bits))

    # Rank bits by positive importance
    order = np.argsort(-imp_pos)  # descending
    top_indices = order[:k]

    mask = np.zeros(n_bits, dtype=bool)
    mask[top_indices] = True

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(OUT_PATH, mask)

    print(f"Saved learned FS mask with {mask.sum()} bits (out of {n_bits}) to {OUT_PATH}")
    print("Top bits (bit_index, importance):")
    for idx in top_indices[:10]:
        print(f"  bit {bit_idx[idx]}  importance={imp[idx]:.4g}")

if __name__ == "__main__":
    main()
