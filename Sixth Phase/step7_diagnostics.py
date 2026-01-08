import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")

INPUT_PATH = BASE_PATH / "outputs_step_6" / "all_methods_comparison_extended.csv"
OUTDIR = BASE_PATH / "outputs_step_7"
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_PATH)

shift_cols = [
    "shift_orig_vs_ewcr",
    "shift_orig_vs_topsis",
    "shift_orig_vs_nwcr",
    "shift_orig_vs_ewcr_norm",
    "shift_orig_vs_topsis_norm",
    "shift_orig_vs_nwcr_norm"
]

rank_shift_summary = df[["factor"] + shift_cols]
rank_shift_summary.to_csv(OUTDIR / "rank_shift_summary.csv", index=False)

def plot_shift(column, title, filename):
    top = df.iloc[np.argsort(df[column].abs())[::-1]].head(30)

    plt.figure(figsize=(10, 12))
    y = range(len(top))

    plt.hlines(y, 0, top[column], color="gray")
    plt.scatter(top[column], y, color="steelblue", s=80)

    plt.yticks(y, top["factor"])
    plt.axvline(0, color="black", linewidth=1)

    plt.title(title)
    plt.xlabel("Rank Shift (Original − Method Rank)")
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=300)
    plt.close()

plot_shift("shift_orig_vs_ewcr", "Top 30 Rank Shifts — Original vs EWCR", "shift_orig_vs_ewcr.png")
plot_shift("shift_orig_vs_topsis", "Top 30 Rank Shifts — Original vs TOPSIS", "shift_orig_vs_topsis.png")
plot_shift("shift_orig_vs_nwcr", "Top 30 Rank Shifts — Original vs NWCR", "shift_orig_vs_nwcr.png")

plt.figure(figsize=(10, 6))
for col in ["shift_orig_vs_ewcr", "shift_orig_vs_topsis", "shift_orig_vs_nwcr"]:
    sns.kdeplot(df[col], label=col, linewidth=2)

plt.title("Distribution of Rank Shifts")
plt.xlabel("Rank Shift Value")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "shift_distributions.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(df["stability_sd"], df["consensus_norm"], alpha=0.6)

plt.xlabel("Stability (SD of normalized ranks)")
plt.ylabel("Consensus Normalized Rank")
plt.title("Stability vs. Consensus Rank")
plt.tight_layout()
plt.savefig(OUTDIR / "stability_vs_rank.png", dpi=300)
plt.close()

df[["factor", "stability_sd", "stability_rank"]].to_csv(
    OUTDIR / "stability_table.csv", index=False
)

df["outlier_flag"] = (
    (df["shift_orig_vs_topsis"].abs() > 200) |
    (df["shift_orig_vs_ewcr"].abs() > 200) |
    (df["shift_orig_vs_nwcr"].abs() > 200)
).astype(int)

df.to_csv(OUTDIR / "all_methods_with_outliers.csv", index=False)
summary = {
    "orig_vs_ewcr_corr": df[["orig_norm","ewcr_norm"]].corr().iloc[0,1],
    "orig_vs_topsis_corr": df[["orig_norm","topsis_norm"]].corr().iloc[0,1],
    "orig_vs_nwcr_corr": df[["orig_norm","nwcr_norm"]].corr().iloc[0,1],
    "ewcr_vs_topsis_corr": df[["ewcr_norm","topsis_norm"]].corr().iloc[0,1],
    "ewcr_vs_nwcr_corr": df[["ewcr_norm","nwcr_norm"]].corr().iloc[0,1],
    "topsis_vs_nwcr_corr": df[["topsis_norm","nwcr_norm"]].corr().iloc[0,1]
}

pd.DataFrame([summary]).to_csv(
    OUTDIR / "method_agreement_summary.csv", index=False
)