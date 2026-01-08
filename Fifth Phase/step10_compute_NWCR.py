import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
NMF_DIR = BASE_PATH / "outputs_nmf"

W = pd.read_csv(NMF_DIR / "nmf_W_factor_components.csv", index_col=0)
weights = pd.read_csv(NMF_DIR / "nmf_component_weights.csv")

print("Loaded W:", W.shape)
print("Loaded weights:", weights.shape)

nmf_weight_vector = weights.set_index("component")["nmf_weight"].reindex(W.columns).values

NWCR_scores = W.values.dot(nmf_weight_vector)

NWCR_df = pd.DataFrame({
    "factor": W.index,
    "NWCR_score": NWCR_scores
})

NWCR_df["NWCR_rank"] = NWCR_df["NWCR_score"].rank(ascending=False, method="dense")

NWCR_df = NWCR_df.sort_values("NWCR_rank")

OUTFILE = NMF_DIR / "nmf_weighted_consensus_rank.csv"
NWCR_df.to_csv(OUTFILE, index=False)

print("Saved:", OUTFILE)
print(NWCR_df.head(20))