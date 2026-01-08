import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")

INPUT_DIR = BASE_PATH / "outputs_pca"

OUTDIR = BASE_PATH / "outputs_topsis"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(INPUT_DIR / "rank_matrix_R_filled.csv", index_col=0)
print("Loaded R matrix:", R.shape)

R_topsis = -1 * R.copy()

norm_R = R_topsis / np.sqrt((R_topsis ** 2).sum())

weights = np.ones(norm_R.shape[1]) / norm_R.shape[1]

weighted_R = norm_R * weights

ideal_best = weighted_R.max(axis=1)
ideal_worst = weighted_R.min(axis=1)

S_plus = np.sqrt(((weighted_R.sub(ideal_best, axis=0)) ** 2).sum(axis=1))
S_minus = np.sqrt(((weighted_R.sub(ideal_worst, axis=0)) ** 2).sum(axis=1))

topsis_score = S_minus / (S_plus + S_minus)

topsis_df = pd.DataFrame({
    "factor": R.index,
    "topsis_score": topsis_score
}).sort_values("topsis_score", ascending=False)

topsis_df["topsis_rank"] = topsis_df["topsis_score"].rank(ascending=False, method="min")

topsis_df.to_csv(OUTDIR / "topsis_scores.csv", index=False)

print("Saved:", OUTDIR / "topsis_scores.csv")