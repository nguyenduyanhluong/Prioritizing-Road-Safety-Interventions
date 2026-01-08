import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import SparsePCA

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")
INPUT_DIR = BASE_PATH / "outputs_step_1"
OUTDIR = BASE_PATH / "outputs_step_2b"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(INPUT_DIR / "rank_matrix_R.csv", index_col=0)
X = R.values.astype(float)
n_factors, n_my = X.shape

X_centered = X - X.mean(axis=0, keepdims=True)

Sigma = np.corrcoef(X_centered, rowvar=False)

M = 3

spca = SparsePCA(
    n_components=M,
    alpha=0.5,         
    ridge_alpha=1e-3,
    random_state=42
)

spca.fit(X_centered)

V_raw = spca.components_

V = []
lambdas = []

for m in range(M):
    v = V_raw[m].copy()

    v = np.maximum(v, 0.0)

    if v.sum() == 0:
        v = np.abs(V_raw[m])

    norm = np.linalg.norm(v)
    if norm == 0:
        v = np.ones(n_my) / np.sqrt(n_my)
    else:
        v = v / norm

    lam = float(v @ Sigma @ v)

    V.append(v)
    lambdas.append(lam)

V = np.vstack(V)              
lambdas = np.array(lambdas)   

if lambdas.sum() == 0:
    lambdas = np.ones_like(lambdas)

weights_num = np.zeros(n_my)

lambda_frac = lambdas / lambdas.sum()

for m in range(M):
    v = V[m]
    v_sum = v.sum()
    if v_sum == 0:
        v_sum = 1.0
    weights_num += lambda_frac[m] * (v / v_sum)

w = weights_num.clip(min=0.0)
w_sum = w.sum()
if w_sum == 0:
    w = np.ones(n_my) / n_my
else:
    w = w / w_sum

weights_df = pd.DataFrame({
    "model_year": R.columns,
    "nspca_weight": w
})
weights_df.to_csv(OUTDIR / "nspca_ewcr_weights.csv", index=False)

loadings_df = pd.DataFrame(
    V.T,
    index=R.columns,
    columns=[f"nspca_comp_{m+1}" for m in range(M)]
)
loadings_df["model_year"] = R.columns
loadings_df.to_csv(OUTDIR / "nspca_loadings_components.csv", index=False)

lambda_df = pd.DataFrame({
    "component": [f"nspca_comp_{m+1}" for m in range(M)],
    "lambda_sparse": lambdas
})
lambda_df.to_csv(OUTDIR / "nspca_component_variances.csv", index=False)

NSPCA_scores = X.dot(w)

df_score = pd.DataFrame({
    "factor": R.index,
    "ewcr_nspca_score": NSPCA_scores
})

df_score["ewcr_nspca_rank"] = df_score["ewcr_nspca_score"].rank(ascending=True)

df_score.to_csv(OUTDIR / "nspca_ewcr_scores.csv", index=False)