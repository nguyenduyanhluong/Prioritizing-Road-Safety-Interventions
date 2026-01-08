import pandas as pd

summary_path = r"outputs_compare\summary_original_topsis_ewcr.csv"

df = pd.read_csv(summary_path)

top20_orig = df.sort_values("original_rank").head(20)["factor"]
top20_ewcr = df.sort_values("EWCR_rank").head(20)["factor"]
top20_topsis = df.sort_values("topsis_rank").head(20)["factor"]

top20_orig.to_csv(r"outputs_compare\top20_original.csv", index=False)
top20_ewcr.to_csv(r"outputs_compare\top20_ewcr.csv", index=False)
top20_topsis.to_csv(r"outputs_compare\top20_topsis.csv", index=False)

print("Saved:")