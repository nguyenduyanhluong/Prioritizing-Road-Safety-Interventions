import numpy as np
import pandas as pd
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11 and 3-12 and 10-12")
INPUT_DIR = BASE_PATH / "outputs_step_1"
PCA_DIR = BASE_PATH / "outputs_step_2"
OUTDIR = BASE_PATH / "outputs_step_10b"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(INPUT_DIR / "rank_matrix_R.csv", index_col=0)
X = R.values.astype(float)
factors = R.index
model_years = R.columns

ewcr_pca = pd.read_csv(PCA_DIR / "ewcr_scores.csv")

Sigma = np.corrcoef(X.T)
p = Sigma.shape[0]

def nspca_first_component(Sigma, alpha=0.5, step=0.01, iters=4000):
    v = np.abs(np.random.randn(p))
    v /= np.linalg.norm(v)

    for _ in range(iters):
        grad = 2 * Sigma @ v - alpha
        v = np.maximum(v + step * grad, 0)
        if v.sum() == 0:
            v = np.ones(p)
        v /= np.linalg.norm(v)

    return v

v_raw = nspca_first_component(Sigma, alpha=0.5)

sparsity_levels = {
    "PCA_baseline": 1.00,
    "NSPCA_30pct": 0.30,
    "NSPCA_20pct": 0.20,
    "NSPCA_10pct": 0.10
}

topN = 50
results = []

retention_matrix = pd.DataFrame(index=factors)

for label, frac in sparsity_levels.items():

    if frac < 1.0:
        k = int(np.ceil(frac * p))
        idx = np.argsort(-v_raw)[:k]

        v = np.zeros_like(v_raw)
        v[idx] = v_raw[idx]
    else:
        v = v_raw.copy()

    w = v / v.sum()

    scores = X @ w
    ranks = pd.Series(scores, index=factors).rank(ascending=True)

    df = pd.DataFrame({
        "factor": factors,
        "rank": ranks
    }).sort_values("rank")

    df.to_csv(OUTDIR / f"{label}_scores.csv", index=False)

    top_factors = set(df.head(topN)["factor"])

    retention_matrix[label] = factors.isin(top_factors).astype(int)

    if label != "PCA_baseline":
        overlap = len(top_factors & set(
            ewcr_pca.sort_values("ewcr_rank").head(topN)["factor"]
        ))
    else:
        overlap = topN

    results.append({
        "method": label,
        "nonzero_weights": int((w > 0).sum()),
        "top50_overlap_with_PCA": overlap
    })

summary_df = pd.DataFrame(results)
summary_df.to_csv(OUTDIR / "nspca_sparsity_summary.csv", index=False)

retention_matrix["retention_count"] = retention_matrix.sum(axis=1)
retention_matrix.to_csv(OUTDIR / "factor_retention_across_sparsity.csv")