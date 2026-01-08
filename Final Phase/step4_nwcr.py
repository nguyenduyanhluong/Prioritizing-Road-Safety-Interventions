import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")
INPUT_DIR = BASE_PATH / "outputs_step_1"
OUTDIR = BASE_PATH / "outputs_step_4"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(INPUT_DIR / "rank_matrix_R.csv", index_col=0)

R_np = R.values.astype(float)
min_vals = R_np.min(axis=0)
max_vals = R_np.max(axis=0)
R_norm = (R_np - min_vals) / (max_vals - min_vals)

nwcr_score = R_norm.mean(axis=1)

df_nwcr = pd.DataFrame({
    "factor": R.index,
    "nwcr_score": nwcr_score
})
df_nwcr["nwcr_rank"] = df_nwcr["nwcr_score"].rank(ascending=True)

df_nwcr.to_csv(OUTDIR / "nwcr_scores.csv", index=False)