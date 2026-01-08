import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\26-11")
INPUT_DIR = BASE_PATH / "level_importances"
OUTDIR = BASE_PATH / "outputs_step_1"
OUTDIR.mkdir(exist_ok=True)

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
MODELS = ["rf", "xgb", "nn"]

records = []

for year in YEARS:
    for model in MODELS:
        fpath = INPUT_DIR / f"level_importance_{model}_{year}.csv"
        
        if not fpath.exists():
            print(f"WARNING: Missing file → {fpath.name}")
            continue
        
        df = pd.read_csv(fpath)

        df = df.rename(columns={"feature": "factor", "importance": "importance"})

        if "factor" not in df or "importance" not in df:
            raise ValueError(f"ERROR: Missing expected columns in {fpath.name}")
        
        df["model"] = model.upper()
        df["year"] = year

        df["rank"] = df["importance"].rank(ascending=False, method="min")
        
        records.append(df[["factor", "model", "year", "importance", "rank"]])

if not records:
    raise RuntimeError("No importance CSVs found — check INPUT_DIR")

all_factors = pd.concat(records, ignore_index=True)

all_factors["model_year"] = all_factors["model"] + "_" + all_factors["year"].astype(str)

R_rank_df = all_factors.pivot_table(
    index="factor",
    columns="model_year",
    values="rank"
)

R_rank_df = R_rank_df.sort_index().sort_index(axis=1)

print("Initial Rank Matrix Shape:", R_rank_df.shape)

for col in R_rank_df.columns:
    max_rank = R_rank_df[col].max()
    R_rank_df[col] = R_rank_df[col].fillna(max_rank)

print("NaN Values Remaining:", R_rank_df.isna().sum().sum())

output_path = OUTDIR / "rank_matrix_R.csv"
R_rank_df.to_csv(output_path)