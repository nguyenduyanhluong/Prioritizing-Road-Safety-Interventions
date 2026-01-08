import pandas as pd
from matplotlib_venn import venn3, venn3_circles
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
OUTDIR = BASE / "outputs_compare"

df = pd.read_csv(OUTDIR / "all_methods_comparison_normalized.csv")

top20_orig   = set(df.sort_values("original_rank").head(20)["factor"])
top20_ewcr   = set(df.sort_values("ewcr_rank").head(20)["factor"])
top20_topsis = set(df.sort_values("topsis_rank").head(20)["factor"])
top20_nwcr   = set(df.sort_values("nwcr_rank").head(20)["factor"])

plt.figure(figsize=(6,6))
venn3([top20_orig, top20_ewcr, top20_topsis], 
      set_labels=("Original", "EWCR", "TOPSIS"))
plt.title("Top-20 Overlap: Original–EWCR–TOPSIS")
plt.savefig(OUTDIR / "step_12_venn_orig_ewcr_topsis.png", dpi=300)
plt.close()

plt.figure(figsize=(6,6))
venn3([top20_ewcr, top20_topsis, top20_nwcr], 
      set_labels=("EWCR", "TOPSIS", "NWCR"))
plt.title("Top-20 Overlap: EWCR–TOPSIS–NWCR")
plt.savefig(OUTDIR / "step12_venn_ewcr_topsis_nwcr.png", dpi=300)
plt.close()

print("Saved: venn diagrams")