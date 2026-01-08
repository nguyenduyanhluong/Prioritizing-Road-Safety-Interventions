import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")

INPUT_PATH = BASE_PATH / "outputs_step_5" / "all_methods_comparison_normalized.csv"
OUTDIR = BASE_PATH / "outputs_step_6"
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_PATH)

df["shift_orig_vs_ewcr"] = df["original_rank"] - df["ewcr_rank"]
df["shift_orig_vs_topsis"] = df["original_rank"] - df["topsis_rank"]
df["shift_orig_vs_nwcr"] = df["original_rank"] - df["nwcr_rank"]

df["shift_orig_vs_ewcr_norm"] = df["orig_norm"] - df["ewcr_norm"]
df["shift_orig_vs_topsis_norm"] = df["orig_norm"] - df["topsis_norm"]
df["shift_orig_vs_nwcr_norm"] = df["orig_norm"] - df["nwcr_norm"]

method_norm_cols = ["orig_norm", "ewcr_norm", "topsis_norm", "nwcr_norm"]
df["consensus_norm"] = df[method_norm_cols].mean(axis=1)
df["consensus_rank"] = df["consensus_norm"].rank(ascending=True)

corr_mat = df[method_norm_cols].corr(method="spearman")
corr_mat.to_csv(OUTDIR / "methods_spearman_correlation.csv")

plt.figure(figsize=(6,5))
sns.heatmap(corr_mat, annot=True, cmap="Blues", fmt=".2f")
plt.tight_layout()
plt.savefig(OUTDIR / "correlation_heatmap.png", dpi=300)
plt.close()

methods_rank_cols = ["original_rank","ewcr_rank","topsis_rank","nwcr_rank"]
stable = df[methods_rank_cols].std(axis=1)
df["stability_sd"] = stable
df["stability_rank"] = df["stability_sd"].rank(ascending=True)
df.to_csv(OUTDIR / "all_methods_comparison_extended.csv", index=False)

pairs = [
    ("orig_norm","ewcr_norm"),
    ("orig_norm","topsis_norm"),
    ("orig_norm","nwcr_norm"),
    ("ewcr_norm","topsis_norm"),
    ("ewcr_norm","nwcr_norm"),
    ("topsis_norm","nwcr_norm")
]

for x,y in pairs:
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=df[x], y=df[y], alpha=0.5, s=25)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"scatter_{x}_vs_{y}.png", dpi=300)
    plt.close()

def top20(col):
    return set(df.sort_values(col).head(20)["factor"])

top_orig = top20("original_rank")
top_ewcr = top20("ewcr_rank")
top_topsis = top20("topsis_rank")
top_nwcr = top20("nwcr_rank")

methods = {
    "Original": top_orig,
    "EWCR": top_ewcr,
    "TOPSIS": top_topsis,
    "NWCR": top_nwcr
}

names = list(methods.keys())
mat = np.zeros((4,4))
for i,m1 in enumerate(names):
    for j,m2 in enumerate(names):
        mat[i,j] = len(methods[m1] & methods[m2])

overlap_df = pd.DataFrame(mat, index=names, columns=names)
overlap_df.to_csv(OUTDIR / "top20_overlap_matrix.csv")

plt.figure(figsize=(6,5))
sns.heatmap(overlap_df, annot=True, cmap="Blues", fmt=".0f")
plt.tight_layout()
plt.savefig(OUTDIR / "top20_overlap_heatmap.png", dpi=300)
plt.close()