import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
PCA_DIR = BASE_PATH / "outputs_pca"
OUTDIR = BASE_PATH / "outputs_topsis"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(PCA_DIR / "rank_matrix_R_filled.csv", index_col=0)

R_norm = R / np.sqrt((R**2).sum())

ideal_best = R_norm.min()
ideal_worst = R_norm.max()

dist_best = np.sqrt(((R_norm - ideal_best) ** 2).sum(axis=1))
dist_worst = np.sqrt(((R_norm - ideal_worst) ** 2).sum(axis=1))

topsis_score = dist_worst / (dist_best + dist_worst)

topsis_df = pd.DataFrame({
    "factor": R.index,
    "topsis_score": topsis_score
}).reset_index(drop=True)

topsis_df["topsis_rank"] = topsis_df["topsis_score"].rank(ascending=False, method="min")

out_path = OUTDIR / "topsis_rank.csv"
topsis_df.to_csv(out_path, index=False)

print("Saved:", out_path)
print(topsis_df.head(10))