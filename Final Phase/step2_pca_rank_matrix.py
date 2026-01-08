import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")
INPUT_DIR = BASE_PATH / "outputs_step_1"
OUTDIR = BASE_PATH / "outputs_step_2"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(INPUT_DIR / "rank_matrix_R.csv", index_col=0)

corr_matrix = R.corr(method="spearman").values

eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

pc1 = eigenvectors[:, 0]          
pc1_abs = np.abs(pc1)

weights = pc1_abs / pc1_abs.sum()

weights_df = pd.DataFrame({
    "model_year": R.columns,
    "pc1_loading": pc1,
    "abs_loading": pc1_abs,
    "weight": weights
})
weights_df.to_csv(OUTDIR / "ewcr_weights.csv", index=False)

EWCR_scores = R.values.dot(weights)

EWCR_df = pd.DataFrame({
    "factor": R.index,
    "ewcr_score": EWCR_scores
})

EWCR_df["ewcr_rank"] = EWCR_df["ewcr_score"].rank(ascending=True)

EWCR_df.to_csv(OUTDIR / "ewcr_scores.csv", index=False)