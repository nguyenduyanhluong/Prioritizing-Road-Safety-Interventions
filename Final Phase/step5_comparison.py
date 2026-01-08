import pandas as pd
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")

R_PATH = BASE_PATH / "outputs_step_1" / "rank_matrix_R.csv"
EWCR_PATH = BASE_PATH / "outputs_step_2" / "ewcr_scores.csv"
TOPSIS_PATH = BASE_PATH / "outputs_step_3" / "topsis_scores.csv"
NWCR_PATH = BASE_PATH / "outputs_step_4" / "nwcr_scores.csv"

OUTDIR = BASE_PATH / "outputs_step_5"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(R_PATH, index_col=0)
orig_rank = R.mean(axis=1).rank(ascending=True)

orig_df = pd.DataFrame({
    "factor": R.index.to_list(),
    "original_rank": orig_rank.values
})

ewcr = pd.read_csv(EWCR_PATH)
topsis = pd.read_csv(TOPSIS_PATH)
nwcr = pd.read_csv(NWCR_PATH)

merged = orig_df.merge(ewcr, on="factor", how="left")
merged = merged.merge(topsis, on="factor", how="left")
merged = merged.merge(nwcr, on="factor", how="left")

merged.to_csv(OUTDIR / "all_methods_comparison.csv", index=False)

df_norm = merged.copy()
df_norm["orig_norm"] = (df_norm["original_rank"] - df_norm["original_rank"].min()) / (df_norm["original_rank"].max() - df_norm["original_rank"].min())
df_norm["ewcr_norm"] = (df_norm["ewcr_rank"] - df_norm["ewcr_rank"].min()) / (df_norm["ewcr_rank"].max() - df_norm["ewcr_rank"].min())
df_norm["topsis_norm"] = (df_norm["topsis_rank"] - df_norm["topsis_rank"].min()) / (df_norm["topsis_rank"].max() - df_norm["topsis_rank"].min())
df_norm["nwcr_norm"] = (df_norm["nwcr_rank"] - df_norm["nwcr_rank"].min()) / (df_norm["nwcr_rank"].max() - df_norm["nwcr_rank"].min())

cols = [
    "factor",
    "original_rank",
    "ewcr_rank",
    "topsis_rank",
    "nwcr_rank",
    "orig_norm",
    "ewcr_norm",
    "topsis_norm",
    "nwcr_norm",
]

df_norm[cols].to_csv(OUTDIR / "all_methods_comparison_normalized.csv", index=False)