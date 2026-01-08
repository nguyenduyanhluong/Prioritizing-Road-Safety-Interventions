import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11 and 3-12 and 10-12 and 17-12")
INPUT_PATH = BASE_PATH / "outputs_step_5" / "all_methods_comparison_normalized.csv"
OUTDIR = BASE_PATH / "outputs_step_4b"
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_PATH)

rank_cols = ["original_rank", "ewcr_rank", "topsis_rank", "nwcr_rank"]

df["mean_rank"] = df[rank_cols].mean(axis=1)
df["sd_rank"] = df[rank_cols].std(axis=1)

plt.figure(figsize=(7, 5))

plt.scatter(
    df["mean_rank"],
    df["sd_rank"],
    s=15,
    alpha=0.6
)

plt.xlabel("Mean Rank Across Models and Years")
plt.ylabel("Rank Standard Deviation")
plt.title("Baseline Importance and Stability of Factors")

plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()

plt.savefig(OUTDIR / "mean_rank_vs_rank_variability.png", dpi=300)
plt.close()