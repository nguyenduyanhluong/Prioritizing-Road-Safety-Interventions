import pandas as pd
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")

CONS_PATH   = BASE_PATH / "consensus_rank_overall.csv"
EWCR_PATH   = BASE_PATH / "outputs_ewcr" / "ewcr_scores.csv"
TOPSIS_PATH = BASE_PATH / "outputs_topsis" / "topsis_rank.csv"
NWCR_PATH   = BASE_PATH / "outputs_nmf" / "nmf_weighted_consensus_rank.csv"

OUTDIR = BASE_PATH / "outputs_compare"
OUTDIR.mkdir(exist_ok=True)

orig = pd.read_csv(CONS_PATH)
orig = orig.rename(columns={"overall_rank": "original_rank"})
print("Original consensus columns:", orig.columns.tolist())

ewcr = pd.read_csv(EWCR_PATH)
print("EWCR columns before:", ewcr.columns.tolist())

if "EWCR_rank" not in ewcr.columns:
    ewcr["EWCR_rank"] = ewcr["EWCR_score"].rank(ascending=False, method="min")

ewcr = ewcr.rename(columns={"EWCR_score": "ewcr_score", "EWCR_rank": "ewcr_rank"})
print("EWCR columns after:", ewcr.columns.tolist())

topsis = pd.read_csv(TOPSIS_PATH)

topsis = topsis.rename(columns={"topsis_score": "topsis_score", "topsis_rank": "topsis_rank"})
print("TOPSIS columns:", topsis.columns.tolist())

nwcr = pd.read_csv(NWCR_PATH)
nwcr = nwcr.rename(columns={"NWCR_score": "nwcr_score", "NWCR_rank": "nwcr_rank"})
print("NWCR columns:", nwcr.columns.tolist())

df = (
    orig[["factor", "original_rank"]]
    .merge(ewcr[["factor", "ewcr_score", "ewcr_rank"]], on="factor", how="outer")
    .merge(topsis[["factor", "topsis_score", "topsis_rank"]], on="factor", how="outer")
    .merge(nwcr[["factor", "nwcr_score", "nwcr_rank"]], on="factor", how="outer")
)

print("Merged shape:", df.shape)

df["shift_orig_vs_ewcr"] = df["original_rank"] - df["ewcr_rank"]
df["shift_orig_vs_topsis"] = df["original_rank"] - df["topsis_rank"]
df["shift_orig_vs_nwcr"] = df["original_rank"] - df["nwcr_rank"]

df["shift_ewcr_vs_nwcr"] = df["ewcr_rank"] - df["nwcr_rank"]
df["shift_topsis_vs_nwcr"] = df["topsis_rank"] - df["nwcr_rank"]

out_path = OUTDIR / "all_methods_comparison.csv"
df.to_csv(out_path, index=False)

print("Saved:", out_path)
print(df.head(10))