import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
COMPARE_DIR = BASE_PATH / "outputs_compare"
OUTDIR = BASE_PATH / "outputs_plots"
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(COMPARE_DIR / "summary_original_topsis_ewcr.csv")

df["abs_shift"] = df["shift_orig_vs_topsis"].abs()

top_shift = df.sort_values("abs_shift", ascending=False).head(30)

plt.figure(figsize=(12, 14))
y = range(len(top_shift))

plt.hlines(y, xmin=0, xmax=top_shift["shift_orig_vs_topsis"], color="gray")
plt.scatter(top_shift["shift_orig_vs_topsis"], y, color="steelblue", s=100)

plt.yticks(y, top_shift["factor"])
plt.axvline(0, color="black", linewidth=1)

plt.title("Rank Shift between Original Consensus and TOPSIS\n(Top 30 by Absolute Shift)", fontsize=16)
plt.xlabel("Rank Shift (Original Rank − TOPSIS Rank)")
plt.ylabel("Factor")

plt.tight_layout()
out_path = OUTDIR / "plot_topsis_rank_shift.png"
plt.savefig(out_path, dpi=300)
plt.close()

print("Saved:", out_path)