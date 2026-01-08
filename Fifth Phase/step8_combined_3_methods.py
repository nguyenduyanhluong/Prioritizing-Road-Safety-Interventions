import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

summary_path = r"outputs_compare\summary_original_topsis_ewcr.csv"

df = pd.read_csv(summary_path)

corr_df = df[["original_rank", "EWCR_rank", "topsis_rank"]]

corr_matrix = corr_df.corr(method="spearman")

plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Original vs EWCR vs TOPSIS Rankings")
plt.tight_layout()

out_path = r"outputs_compare\correlation_heatmap_ranks.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"Saved:")