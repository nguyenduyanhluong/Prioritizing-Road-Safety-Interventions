import pandas as pd
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")

orig = pd.read_csv(BASE_PATH / "outputs_step_5" / "all_methods_comparison_normalized.csv")
nmf = pd.read_csv(BASE_PATH / "outputs_step_8" / "nmf_weighted_consensus_rank.csv")

df = (
    orig.merge(nmf, on="factor", how="left")
)

df = df.rename(columns={"nwcr_rank_x": "nwcr_rank"}) if "nwcr_rank_x" in df.columns else df

OUTDIR = BASE_PATH / "outputs_step_9a"
OUTDIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTDIR / "final_comparison_all_methods.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("Final columns:", df.columns.tolist())