import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
OUTDIR = BASE / "outputs_compare"

df = pd.read_csv(OUTDIR / "all_methods_comparison_normalized.csv")

methods = {
    "Original": set(df.sort_values("original_rank").head(20)["factor"]),
    "EWCR":     set(df.sort_values("ewcr_rank").head(20)["factor"]),
    "TOPSIS":   set(df.sort_values("topsis_rank").head(20)["factor"]),
    "NWCR":     set(df.sort_values("nwcr_rank").head(20)["factor"]),
}

names = list(methods.keys())
mat = np.zeros((4,4))

for i, m1 in enumerate(names):
    for j, m2 in enumerate(names):
        mat[i, j] = len(methods[m1] & methods[m2])

plt.figure(figsize=(6,5))
sns.heatmap(mat, annot=True, fmt=".0f", cmap="Blues",
            xticklabels=names, yticklabels=names)
plt.title("Top-20 Overlap Heatmap Across 4 Methods")
plt.tight_layout()

plt.savefig(OUTDIR / "step12_top20_overlap_heatmap.png", dpi=300)
plt.close()

print("Saved:", OUTDIR / "step12_top20_overlap_heatmap.png")