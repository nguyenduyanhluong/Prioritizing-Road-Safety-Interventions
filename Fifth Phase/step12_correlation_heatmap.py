import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
df = pd.read_csv(BASE / "outputs_compare" / "all_methods_comparison_normalized.csv")

corr_df = df[["orig_norm", "ewcr_norm", "topsis_norm", "nwcr_norm"]].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr_df, annot=True, cmap="viridis", fmt=".2f")
plt.title("Correlation Heatmap of Ranking Methods")
plt.tight_layout()

OUTDIR = BASE / "outputs_compare"
plt.savefig(OUTDIR / "step12_correlation_heatmap.png", dpi=300)
plt.close()

print("Saved:", OUTDIR / "step12_correlation_heatmap.png")