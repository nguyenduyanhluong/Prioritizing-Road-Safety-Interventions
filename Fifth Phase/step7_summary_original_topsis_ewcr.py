import pandas as pd
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
COMPARE_DIR = BASE_PATH / "outputs_compare"
TOPSIS_DIR = BASE_PATH / "outputs_topsis"
OUTDIR = COMPARE_DIR
OUTDIR.mkdir(exist_ok=True)

ewcr_df = pd.read_csv(COMPARE_DIR / "ewcr_vs_consensus.csv")

topsis_df = pd.read_csv(TOPSIS_DIR / "topsis_rank.csv")

merged = (
    ewcr_df[["factor", "overall_rank", "EWCR_rank"]]
    .merge(topsis_df[["factor", "topsis_rank"]], on="factor", how="left")
)

merged = merged.rename(columns={
    "overall_rank": "original_rank"
})

merged["shift_orig_vs_ewcr"] = merged["original_rank"] - merged["EWCR_rank"]
merged["shift_orig_vs_topsis"] = merged["original_rank"] - merged["topsis_rank"]
merged["shift_ewcr_vs_topsis"] = merged["EWCR_rank"] - merged["topsis_rank"]

out_path = OUTDIR / "summary_original_topsis_ewcr.csv"
merged.to_csv(out_path, index=False)

print("Saved:", out_path)
print(merged.head(10))