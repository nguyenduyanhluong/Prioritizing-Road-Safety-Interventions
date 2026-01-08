import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")
INPUT_DIR_R = BASE_PATH / "outputs_step_1"
INPUT_DIR_W = BASE_PATH / "outputs_step_2"
OUTDIR = BASE_PATH / "outputs_step_3"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(INPUT_DIR_R / "rank_matrix_R.csv", index_col=0).astype(float)

weights = pd.read_csv(INPUT_DIR_W / "ewcr_weights.csv")["weight"].values
weights = weights / weights.sum()

X = R.values
col_norms = np.sqrt((X ** 2).sum(axis=0))
col_norms[col_norms == 0] = 1 

R_norm = X / col_norms

V = R_norm * weights

ideal_best = V.min(axis=0)
ideal_worst = V.max(axis=0)

dist_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
dist_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

topsis_score = dist_worst / (dist_worst + dist_best)

df_topsis = pd.DataFrame({
    "factor": R.index,
    "topsis_score": topsis_score
})

df_topsis["topsis_rank"] = df_topsis["topsis_score"].rank(ascending=False)

df_topsis.to_csv(OUTDIR / "topsis_scores.csv", index=False)