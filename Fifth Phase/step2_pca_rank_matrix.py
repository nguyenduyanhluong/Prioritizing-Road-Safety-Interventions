import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")
INPUT_DIR = BASE_PATH / "outputs_step_1"
OUTDIR = BASE_PATH / "outputs_step_2"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(INPUT_DIR / "rank_matrix_R.csv", index_col=0)
print("Loaded R matrix:", R.shape)

pca = PCA()
pca.fit(R.T)

eigenvalues = pca.explained_variance_

weights = eigenvalues / eigenvalues.sum()

weights_df = pd.DataFrame({
    "model_year": R.columns,
    "eigenvalue": eigenvalues,
    "weight": weights
})
weights_df.to_csv(OUTDIR / "ewcr_weights.csv", index=False)

print("EWCR Weights Saved:", OUTDIR / "ewcr_weights.csv")

W = weights
R_np = R.values

EWCR_scores = R_np.dot(W)

EWCR_df = pd.DataFrame({
    "factor": R.index,
    "ewcr_score": EWCR_scores
})

EWCR_df["ewcr_rank"] = EWCR_df["ewcr_score"].rank(ascending=True)
EWCR_df.to_csv(OUTDIR / "ewcr_scores.csv", index=False)

print("Saved EWCR Scores:", OUTDIR / "ewcr_scores.csv")