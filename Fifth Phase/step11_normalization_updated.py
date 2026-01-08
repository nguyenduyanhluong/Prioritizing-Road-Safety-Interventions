import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
df = pd.read_csv(BASE / "outputs_compare" / "all_methods_comparison.csv")

def normalize(series):
    return 1 - (series - series.min()) / (series.max() - series.min())

df["orig_norm"]   = normalize(df["original_rank"])
df["ewcr_norm"]   = normalize(df["ewcr_rank"])
df["topsis_norm"] = normalize(df["topsis_rank"])
df["nwcr_norm"]   = normalize(df["nwcr_rank"])

df["shift_orig_vs_ewcr_norm"] = df["orig_norm"] - df["ewcr_norm"]
df["shift_orig_vs_topsis_norm"] = df["orig_norm"] - df["topsis_norm"]
df["shift_orig_vs_nwcr_norm"] = df["orig_norm"] - df["nwcr_norm"]

df["shift_ewcr_vs_nwcr_norm"] = df["ewcr_norm"] - df["nwcr_norm"]
df["shift_topsis_vs_nwcr_norm"] = df["topsis_norm"] - df["nwcr_norm"]

out = BASE / "outputs_compare" / "all_methods_comparison_normalized.csv"
df.to_csv(out, index=False)

print(df.head(10))