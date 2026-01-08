import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

orig = set(pd.read_csv(r"outputs_compare\top20_original.csv")["factor"])
ewcr = set(pd.read_csv(r"outputs_compare\top20_ewcr.csv")["factor"])
topsis = set(pd.read_csv(r"outputs_compare\top20_topsis.csv")["factor"])

plt.figure(figsize=(8,7))
venn3(
    [orig, ewcr, topsis],
    set_labels=("Original", "EWCR", "TOPSIS")
)
plt.title("Top-20 Feature Overlap: Original vs EWCR vs TOPSIS")
plt.tight_layout()

out_path = r"outputs_compare\top20_venn_diagram.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"Saved:")