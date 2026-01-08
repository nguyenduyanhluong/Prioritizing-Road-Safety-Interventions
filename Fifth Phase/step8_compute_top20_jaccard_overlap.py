import pandas as pd

orig = set(pd.read_csv(r"outputs_compare\top20_original.csv")["factor"])
ewcr = set(pd.read_csv(r"outputs_compare\top20_ewcr.csv")["factor"])
topsis = set(pd.read_csv(r"outputs_compare\top20_topsis.csv")["factor"])

def jaccard(a, b):
    return len(a & b) / len(a | b)

overlap = pd.DataFrame(
    {
        "Original": [1.0, jaccard(orig, ewcr), jaccard(orig, topsis)],
        "EWCR":     [jaccard(ewcr, orig), 1.0, jaccard(ewcr, topsis)],
        "TOPSIS":   [jaccard(topsis, orig), jaccard(topsis, ewcr), 1.0]
    },
    index=["Original", "EWCR", "TOPSIS"]
)

output_path = r"outputs_compare\top20_jaccard_overlap.csv"
overlap.to_csv(output_path)

print(f"Saved:")