import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
OUTDIR = BASE / "outputs_compare"
df = pd.read_csv(OUTDIR / "all_methods_comparison_normalized.csv")

pairs = [
    ("orig_norm", "ewcr_norm"),
    ("orig_norm", "topsis_norm"),
    ("orig_norm", "nwcr_norm"),
    ("ewcr_norm", "topsis_norm"),
    ("ewcr_norm", "nwcr_norm"),
    ("topsis_norm", "nwcr_norm"),
]

for x, y in pairs:
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=df[x], y=df[y], alpha=0.5, s=30)
    plt.xlabel(x.replace("_", " ").upper())
    plt.ylabel(y.replace("_", " ").upper())
    plt.title(f"{x.upper()} vs {y.upper()}")
    plt.tight_layout()
    fname = OUTDIR / f"scatter_pairs_{x}_vs_{y}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved:", fname)