import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib_venn import venn3

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")

INPUT_PATH = BASE_PATH / "outputs_step_9a" / "final_comparison_all_methods.csv"
OUTDIR = BASE_PATH / "outputs_step_9b"
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_PATH)

rank_cols = ["original_rank", "ewcr_rank", "topsis_rank", "nwcr_rank"]

corr = df[rank_cols].corr(method="spearman")
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="Blues", vmin=0.7, vmax=1.0, fmt=".2f")
plt.title("Method Agreement — Spearman Rank Correlation")
plt.tight_layout()
plt.savefig(OUTDIR / "method_agreement_correlation.png", dpi=300)
plt.close()

sns.pairplot(df[rank_cols], corner=True, plot_kws={"alpha":0.4, "s":15})
plt.suptitle("Pairwise Rank Agreement Across Methods", y=1.02)
plt.savefig(OUTDIR / "method_agreement_pairplot.png", dpi=300)
plt.close()

df["orig_ewcr_diff"] = df["original_rank"] - df["ewcr_rank"]
df["orig_topsis_diff"] = df["original_rank"] - df["topsis_rank"]
df["orig_nwcr_diff"] = df["original_rank"] - df["nwcr_rank"]

plt.figure(figsize=(10,6))
for col in ["orig_ewcr_diff", "orig_topsis_diff", "orig_nwcr_diff"]:
    sns.kdeplot(df[col], linewidth=2)
plt.title("Rank Difference Distributions")
plt.xlabel("Rank Difference")
plt.tight_layout()
plt.savefig(OUTDIR / "rank_difference_distributions.png", dpi=300)
plt.close()

def topN(col, N):
    return set(df.sort_values(col).head(N)["factor"])

methods = ["Original", "EWCR", "TOPSIS", "NWCR"]

for N in [100, 200, 300]:

    top = {
        "Original": topN("original_rank", N),
        "EWCR": topN("ewcr_rank", N),
        "TOPSIS": topN("topsis_rank", N),
        "NWCR": topN("nwcr_rank", N)
    }

    mat = np.zeros((4,4))
    for i,a in enumerate(methods):
        for j,b in enumerate(methods):
            mat[i,j] = len(top[a] & top[b])

    plt.figure(figsize=(6,5))
    sns.heatmap(mat, annot=True, cmap="Blues", fmt=".0f",
                xticklabels=methods, yticklabels=methods)
    plt.title(f"Top-{N} Overlap Heatmap")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"top{N}_overlap_heatmap.png", dpi=300)
    plt.close()

    jac = np.zeros((4,4))
    for i,a in enumerate(methods):
        for j,b in enumerate(methods):
            A = top[a]
            B = top[b]
            jac[i,j] = len(A & B) / len(A | B)

    jac_df = pd.DataFrame(jac, index=methods, columns=methods)
    jac_df.to_csv(OUTDIR / f"jaccard_similarity_top{N}.csv")

    plt.figure(figsize=(6,5))
    sns.heatmap(jac_df, annot=True, cmap="YlGnBu", vmin=0.3, vmax=1.0, fmt=".2f")
    plt.title(f"Jaccard Similarity of Top-{N} Features")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"jaccard_similarity_heatmap_top{N}.png", dpi=300)
    plt.close()

    orig_set = top["Original"]
    ewcr_set = top["EWCR"]
    topsis_set = top["TOPSIS"]
    nwcr_set = top["NWCR"]

    plt.figure(figsize=(7,7))
    venn3([orig_set, ewcr_set, topsis_set],
          set_labels=("Original", "EWCR", "TOPSIS"))
    plt.title(f"Top-{N} Venn: Original / EWCR / TOPSIS")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"venn_top{N}_orig_ewcr_topsis.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7,7))
    venn3([ewcr_set, topsis_set, nwcr_set],
          set_labels=("EWCR", "TOPSIS", "NWCR"))
    plt.title(f"Top-{N} Venn: EWCR / TOPSIS / NWCR")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"venn_top{N}_ewcr_topsis_nwcr.png", dpi=300)
    plt.close()

print("Generated Venn and overlap analysis for N = 100, 200, 300")