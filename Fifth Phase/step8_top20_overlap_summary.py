import pandas as pd

orig = set(pd.read_csv(r"outputs_compare\top20_original.csv")["factor"])
ewcr = set(pd.read_csv(r"outputs_compare\top20_ewcr.csv")["factor"])
topsis = set(pd.read_csv(r"outputs_compare\top20_topsis.csv")["factor"])

overlap_orig_ewcr = len(orig & ewcr)
overlap_orig_topsis = len(orig & topsis)
overlap_ewcr_topsis = len(ewcr & topsis)

overlap_all_three = len(orig & ewcr & topsis)

df_overlap = pd.DataFrame({
    "Pair": [
        "Original ∩ EWCR",
        "Original ∩ TOPSIS",
        "EWCR ∩ TOPSIS",
        "All three (intersection)"
    ],
    "Overlap_Count": [
        overlap_orig_ewcr,
        overlap_orig_topsis,
        overlap_ewcr_topsis,
        overlap_all_three
    ]
})

output_path = r"outputs_compare\top20_overlap_counts.csv"
df_overlap.to_csv(output_path, index=False)

print(f"Saved:")