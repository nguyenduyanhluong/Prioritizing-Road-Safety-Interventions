import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")

INPUT_PATH = BASE_PATH / "outputs_step_1" / "rank_matrix_R.csv"
OUTDIR = BASE_PATH / "outputs_step_8"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(INPUT_PATH, index_col=0)

R_scaled = R / R.max(axis=0)

nmf = NMF(
    n_components=3,
    init="nndsvda",
    random_state=42,
    max_iter=2000,
    tol=1e-4
)

W = nmf.fit_transform(R_scaled.values)
H = nmf.components_
recon = W.dot(H)
recon_error = np.linalg.norm(R_scaled.values - recon, "fro")

W_df = pd.DataFrame(W, index=R.index, columns=[f"NMF_comp{k+1}" for k in range(W.shape[1])])
H_df = pd.DataFrame(H, index=[f"NMF_comp{k+1}" for k in range(H.shape[0])], columns=R.columns)

W_df.to_csv(OUTDIR / "nmf_W_factor_components.csv")
H_df.to_csv(OUTDIR / "nmf_H_components_modelyear.csv")

with open(OUTDIR / "nmf_info.txt", "w") as f:
    f.write(f"reconstruction_error_fro={recon_error}\n")

plt.figure(figsize=(14,4))
sns.heatmap(H_df, cmap="viridis", vmin=0, vmax=H_df.max().max())
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(OUTDIR / "heatmap_H_modelyear.png", dpi=300)
plt.close()

top_factors = np.argsort(-W, axis=0)[:10, :]

rows = []
for comp in range(W.shape[1]):
    idx = top_factors[:, comp]
    tmp = pd.DataFrame({
        "factor": W_df.index[idx],
        "component": f"NMF_comp{comp+1}",
        "loading": W_df.iloc[idx, comp]
    })
    rows.append(tmp)

plot_df = pd.concat(rows)
pivot_df = plot_df.pivot(index="factor", columns="component", values="loading")

plt.figure(figsize=(8,10))
sns.heatmap(pivot_df, cmap="mako", vmin=0, vmax=pivot_df.max().max())
plt.tight_layout()
plt.savefig(OUTDIR / "heatmap_W_top10_factors.png", dpi=300)
plt.close()

raw_weights = W_df.sum(axis=0)
norm_weights = raw_weights / raw_weights.sum()

NWCR_scores = W_df.values.dot(norm_weights.values)

NWCR_df = pd.DataFrame({
    "factor": W_df.index,
    "nwcr_score": NWCR_scores
})
NWCR_df["nwcr_rank"] = NWCR_df["nwcr_score"].rank(ascending=False)

NWCR_df = NWCR_df.sort_values("nwcr_rank")
NWCR_df.to_csv(OUTDIR / "nmf_weighted_consensus_rank.csv", index=False)