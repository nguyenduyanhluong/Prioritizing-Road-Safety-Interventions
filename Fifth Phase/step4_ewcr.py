import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")

R_DIR = BASE_PATH / "outputs_pca"
LOAD_DIR = BASE_PATH / "outputs_pca"
WEIGHT_DIR = BASE_PATH / "outputs_weights"

OUTDIR = BASE_PATH / "outputs_ewcr"
OUTDIR.mkdir(exist_ok=True)

R_centered = pd.read_csv(R_DIR / "rank_matrix_R_centered.csv", index_col=0)
loadings = pd.read_csv(LOAD_DIR / "pca_loadings.csv", index_col=0)
weights = pd.read_csv(WEIGHT_DIR / "pc_weights.csv")["weight"].values

print("R_centered:", R_centered.shape)
print("Loadings:", loadings.shape)
print("weights:", len(weights))


Z = R_centered.values @ loadings.values.T
Z_df = pd.DataFrame(Z, index=R_centered.index, columns=loadings.index)

Z_df.to_csv(OUTDIR / "factor_pc_scores.csv")

EWCR = (np.abs(Z) * weights).sum(axis=1)

ewcr_df = pd.DataFrame({
    "factor": R_centered.index,
    "EWCR_score": EWCR
}).sort_values("EWCR_score", ascending=False)

ewcr_df.to_csv(OUTDIR / "ewcr_scores.csv", index=False)

print("Saved:", OUTDIR / "ewcr_scores.csv")
print(ewcr_df.head(10))