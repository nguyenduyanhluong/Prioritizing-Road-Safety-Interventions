import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\19-11")
INPUT_DIR = BASE_PATH / "level_importances"
OUTDIR = BASE_PATH / "outputs_rank_matrix_r"
OUTDIR.mkdir(exist_ok=True)

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
MODELS = ["rf", "xgb", "nn"]

records = []
for year in YEARS:
    for model in MODELS:
        f = INPUT_DIR / f"level_importance_{model}_{year}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df.rename(columns={"feature": "factor", "importance": "importance"})
        df["model"] = model.upper()
        df["year"] = year
        df["rank"] = df["importance"].rank(ascending=False, method="min")
        records.append(df[["year", "model", "factor", "importance", "rank"]])

if not records:
    raise RuntimeError("No importance CSVs found in INPUT_DIR")

all_factors = pd.concat(records, ignore_index=True)

all_factors["model_year"] = all_factors["model"] + "_" + all_factors["year"].astype(str)

R_rank_df = all_factors.pivot_table(
    index="factor",
    columns="model_year",
    values="rank"
)

R_rank_df = R_rank_df.sort_index().sort_index(axis=1)

print("Rank matrix shape (factors × model-year):", R_rank_df.shape)
print("First few columns:", list(R_rank_df.columns)[:6])

R_rank_df.to_csv(OUTDIR / "rank_matrix_R.csv")

print("Saved:", OUTDIR / "rank_matrix_R.csv")