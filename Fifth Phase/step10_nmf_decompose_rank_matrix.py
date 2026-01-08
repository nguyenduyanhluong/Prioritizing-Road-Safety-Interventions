import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import NMF

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
PCA_DIR = BASE_PATH / "outputs_pca"
OUTDIR = BASE_PATH / "outputs_nmf"
OUTDIR.mkdir(exist_ok=True)

R = pd.read_csv(PCA_DIR / "rank_matrix_R_filled.csv", index_col=0)
print("Loaded R matrix:", R.shape)

R_scaled = R / R.max(axis=0)

n_components = 3

nmf = NMF(
    n_components=n_components,
    init="nndsvda",
    random_state=42,
    max_iter=2000,
    tol=1e-4
)

W = nmf.fit_transform(R_scaled.values)   
H = nmf.components_                     

recon = np.dot(W, H)
recon_error = np.linalg.norm(R_scaled.values - recon, "fro")

print(f"NMF finished with n_components={n_components}")
print(f"Reconstruction error (Frobenius norm): {recon_error:.4f}")

W_df = pd.DataFrame(
    W,
    index=R.index,
    columns=[f"NMF_comp{k+1}" for k in range(n_components)]
)
W_df.to_csv(OUTDIR / "nmf_W_factor_components.csv")

H_df = pd.DataFrame(
    H,
    index=[f"NMF_comp{k+1}" for k in range(n_components)],
    columns=R.columns
)
H_df.to_csv(OUTDIR / "nmf_H_components_modelyear.csv")

with open(OUTDIR / "nmf_info.txt", "w") as f:
    f.write(f"n_components = {n_components}\n")
    f.write(f"reconstruction_error_fro = {recon_error:.6f}\n")

print("Saved:")