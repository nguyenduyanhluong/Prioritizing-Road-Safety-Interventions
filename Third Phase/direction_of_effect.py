import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

OUTDIR = Path("outputs_covid_shift")
OUTDIR.mkdir(exist_ok=True)

CONSENSUS_FILE = OUTDIR / "consensus_feature_importance.csv"
LOGIT_FILE = OUTDIR / "logit_results_combined.csv"
TOP_N = 20
SORT_BY = "consensus" 

consensus = pd.read_csv(CONSENSUS_FILE)
logit = pd.read_csv(LOGIT_FILE)

if "Unnamed: 0" in logit.columns:
    logit = logit.rename(columns={"Unnamed: 0": "feature"})

logit = logit[["feature", "coef", "odds_ratio", "direction"]]

merged = consensus.merge(logit, on="feature", how="left")

out_csv = OUTDIR / "consensus_with_direction.csv"
merged.to_csv(out_csv, index=False)
print(f"Saved → {out_csv}")

if SORT_BY == "consensus":
    top_feats = merged.head(TOP_N)
elif SORT_BY == "abs_coef":
    top_feats = merged.reindex(
        merged["coef"].abs().sort_values(ascending=False).index
    ).head(TOP_N)
else:
    raise ValueError("SORT_BY must be 'consensus' or 'abs_coef'")

plt.figure(figsize=(12, 8))
barplot = sns.barplot(
    data=top_feats,
    x="coef", y="feature",
    hue="direction", dodge=False,
    palette={"positive": "red", "negative": "blue"}
)

for i, row in top_feats.iterrows():
    coef = row["coef"]
    or_val = row["odds_ratio"]
    barplot.text(
        coef + (0.2 if coef >= 0 else -0.2), 
        barplot.get_yticks()[list(top_feats.index).index(i)],
        f"OR={or_val:.2f}",
        va="center",
        ha="left" if coef >= 0 else "right",
        fontsize=9,
        color="black"
    )

plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Logit Coefficient (Direction of Effect)")
plt.ylabel("Feature")
plt.title(
    f"Top {TOP_N} Consensus Features with Direction\n(Logit Combined, Sorted by {SORT_BY})"
)
plt.legend(title="Direction", loc="lower right")
plt.tight_layout()

out_png = OUTDIR / f"consensus_feature_directions_{SORT_BY}.png"
plt.savefig(out_png, dpi=300)
plt.close()

print(f"Saved → {out_png}")