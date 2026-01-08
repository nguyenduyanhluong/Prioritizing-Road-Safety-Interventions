import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
df = pd.read_csv(BASE / "outputs_compare" / "all_methods_comparison_normalized.csv")

shift_col = "shift_orig_vs_nwcr_norm"

top = df.sort_values(shift_col, ascending=False).head(20)

plt.figure(figsize=(8,7))
y = range(len(top))
plt.hlines(y, 0, top[shift_col], color="gray")
plt.scatter(top[shift_col], y, s=80)

plt.yticks(y, top["factor"])
plt.xlabel("Normalized Rank Shift")
plt.title("Top 20 Rank Shifts (Original vs NWCR)")
plt.gca().invert_yaxis()
plt.tight_layout()

OUTDIR = BASE / "outputs_compare"
plt.savefig(OUTDIR / "rank_shift_lollipop_rank_shift_lollipop.png", dpi=300)
plt.close()

print("Saved:", OUTDIR / "rank_shift_lollipop_rank_shift_lollipop.png")