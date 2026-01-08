import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

orig = set(pd.read_csv(r"outputs_compare\top20_original.csv")["factor"])
ewcr = set(pd.read_csv(r"outputs_compare\top20_ewcr.csv")["factor"])
topsis = set(pd.read_csv(r"outputs_compare\top20_topsis.csv")["factor"])

methods = ["Original", "EWCR", "TOPSIS"]
sets = [orig, ewcr, topsis]

overlap_matrix = []
for a in sets:
    row = []
    for b in sets:
        row.append(len(a & b))
    overlap_matrix.append(row)

df_overlap = pd.DataFrame(overlap_matrix, index=methods, columns=methods)

plt.figure(figsize=(6,5))
sns.heatmap(df_overlap, annot=True, cmap="Blues", fmt="d")
plt.title("Top-20 Overlap Heatmap (Count of Shared Factors)")
plt.tight_layout()

out_path = r"outputs_compare\top20_overlap_heatmap.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"Saved:")