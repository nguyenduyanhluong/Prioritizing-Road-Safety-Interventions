import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
NMF_DIR = BASE_PATH / "outputs_nmf"
OUTDIR = BASE_PATH / "outputs_nmf"
OUTDIR.mkdir(exist_ok=True)

W = pd.read_csv(NMF_DIR / "nmf_W_factor_components.csv", index_col=0)

print("Loaded W:", W.shape)

raw_weights = W.sum(axis=0) 
raw_weights = raw_weights.to_frame(name="raw_weight")

raw_weights["nmf_weight"] = raw_weights["raw_weight"] / raw_weights["raw_weight"].sum()

raw_weights["component"] = raw_weights.index

weights_df = raw_weights[["component", "raw_weight", "nmf_weight"]]

weights_df.to_csv(OUTDIR / "nmf_component_weights.csv", index=False)

print("Saved:", OUTDIR / "nmf_component_weights.csv")
print(weights_df)