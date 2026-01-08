import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11 and 3-12 and 10-12")
INPUT = BASE_PATH / "outputs_step_9a" / "final_comparison_all_methods.csv"
OUTDIR = BASE_PATH / "outputs_step_11"
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)

rank_cols = ["original_rank", "ewcr_rank", "topsis_rank", "nwcr_rank"]

df["mean_rank"] = df[rank_cols].mean(axis=1)
df["sd_rank"] = df[rank_cols].std(axis=1)

def classify(row):
    if row["mean_rank"] <= 100 and row["sd_rank"] <= 50:
        return "Highly Important (Strong Consensus)"
    elif row["mean_rank"] <= 200:
        return "Moderately Important"
    elif row["sd_rank"] >= 150:
        return "Unstable but High Impact"
    else:
        return "Low Importance"

df["importance_category"] = df.apply(classify, axis=1)

df_sorted = df.sort_values(["importance_category", "mean_rank"])

df_sorted.to_csv(OUTDIR / "key_risk_factors_summary_full.csv", index=False)

df_top50 = df_sorted.head(50)
df_top50.to_csv(OUTDIR / "key_risk_factors_summary_top50.csv", index=False)