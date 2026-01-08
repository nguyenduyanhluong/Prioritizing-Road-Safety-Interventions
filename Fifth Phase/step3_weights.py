import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
INPUT_DIR = BASE_PATH / "outputs_pca"
OUTDIR = BASE_PATH / "outputs_weights"
OUTDIR.mkdir(exist_ok=True)

df_eig = pd.read_csv(INPUT_DIR / "pca_eigenvalues.csv")

eigenvalues = df_eig["eigenvalue"].values

weights = eigenvalues / eigenvalues.sum()

df_weights = pd.DataFrame({
    "component": df_eig["component"],
    "eigenvalue": eigenvalues,
    "weight": weights
})

df_weights.to_csv(OUTDIR / "pc_weights.csv", index=False)

print("Saved:", OUTDIR / "pc_weights.csv")
print(df_weights.head())