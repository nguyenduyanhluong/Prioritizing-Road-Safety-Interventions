import pandas as pd
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")

EWCR_DIR = BASE_PATH / "outputs_ewcr"
OUTDIR = BASE_PATH / "outputs_compare"
OUTDIR.mkdir(exist_ok=True)

ewcr = pd.read_csv(EWCR_DIR / "ewcr_scores.csv")

consensus_file = BASE_PATH / "consensus_rank_overall.csv"

if not consensus_file.exists():
    raise FileNotFoundError("consensus_rank_overall.csv not found in 19-11 folder!")

cons = pd.read_csv(consensus_file)

merged = ewcr.merge(cons, on="factor", how="left")

merged["EWCR_rank"] = merged["EWCR_score"].rank(ascending=False)

merged["rank_shift"] = merged["overall_rank"] - merged["EWCR_rank"]

merged.to_csv(OUTDIR / "ewcr_vs_consensus.csv", index=False)

print("Saved:", OUTDIR / "ewcr_vs_consensus.csv")
print(merged.head(15))