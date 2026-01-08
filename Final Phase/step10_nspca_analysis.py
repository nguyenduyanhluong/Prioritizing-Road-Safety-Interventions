import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib_venn import venn2

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")

RANK_PATH = BASE_PATH / "outputs_step_5" / "all_methods_comparison_normalized.csv"
NSPCA_PATH = BASE_PATH / "outputs_step_2b" / "nspca_ewcr_scores.csv"               
OUTDIR = BASE_PATH / "outputs_step_10"
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(RANK_PATH)
nspca = pd.read_csv(NSPCA_PATH)

df = df.merge(nspca, on="factor", how="left")

df["nspca_norm"] = (df["ewcr_nspca_rank"] - df["ewcr_nspca_rank"].min()) / (
    df["ewcr_nspca_rank"].max() - df["ewcr_nspca_rank"].min()
)

cols = ["ewcr_norm", "nspca_norm"]
corr = df[cols].corr(method="spearman")

plt.figure(figsize=(8, 5))
sns.heatmap(
    corr,
    annot=True,
    cmap="Blues",
    vmin=-1.0,
    vmax=1.0,
    fmt=".2f"
)
plt.title("PCA–EWCR vs. NSPCA–EWCR (Spearman Correlation)")
plt.tight_layout()
plt.savefig(OUTDIR / "nspca_correlation.png", dpi=300)
plt.close()

df["shift_ewcr_vs_nspca"] = df["ewcr_rank"] - df["ewcr_nspca_rank"]

plt.figure(figsize=(8, 5))
sns.kdeplot(df["shift_ewcr_vs_nspca"], linewidth=2)
plt.axvline(0, color="black", linestyle="--")
plt.title("Rank Shift Distribution: PCA–EWCR vs NSPCA–EWCR")
plt.xlabel("Rank Difference (PCA–EWCR minus NSPCA–EWCR)")
plt.tight_layout()
plt.savefig(OUTDIR / "nspca_rank_shift_distribution.png", dpi=300)
plt.close()

shift_summary = {
    "mean_shift": df["shift_ewcr_vs_nspca"].mean(),
    "sd_shift": df["shift_ewcr_vs_nspca"].std(),
    "min_shift": df["shift_ewcr_vs_nspca"].min(),
    "max_shift": df["shift_ewcr_vs_nspca"].max(),
}
pd.DataFrame([shift_summary]).to_csv(
    OUTDIR / "nspca_rank_shift_summary.csv", index=False
)

def topN(dataframe, col, N):
    return set(dataframe.sort_values(col).head(N)["factor"])

top_pca = topN(df, "ewcr_rank", 300)
top_nspca = topN(df, "ewcr_nspca_rank", 300)

overlap = len(top_pca & top_nspca)
overlap_pct = overlap / 300.0

pd.DataFrame({
    "N": [300],
    "overlap_count": [overlap],
    "overlap_pct": [overlap_pct]
}).to_csv(OUTDIR / "nspca_top300_overlap.csv", index=False)

plt.figure(figsize=(8, 5))
venn2([top_pca, top_nspca], set_labels=("PCA–EWCR", "NSPCA–EWCR"))
plt.title("Top-300 Overlap: PCA–EWCR vs NSPCA–EWCR")
plt.tight_layout()
plt.savefig(OUTDIR / "nspca_top300_venn.png", dpi=300)
plt.close()

df.to_csv(OUTDIR / "nspca_full_comparison.csv", index=False)