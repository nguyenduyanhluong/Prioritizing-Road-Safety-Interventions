import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
INPUT_DIR = BASE_PATH / "outputs_step3b_importances"
OUTDIR = BASE_PATH / "outputs_step4_visuals"
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

consensus_by_year = (
    all_factors
    .groupby(["year", "factor"])["rank"]
    .mean()
    .reset_index()
    .rename(columns={"rank": "avg_rank"})
)
consensus_by_year["consensus_rank"] = consensus_by_year.groupby("year")["avg_rank"].rank(ascending=True, method="min")
consensus_by_year.to_csv(OUTDIR / "consensus_rank_by_year.csv", index=False)

consensus_overall = (
    all_factors
    .groupby("factor")["rank"]
    .mean()
    .reset_index()
    .rename(columns={"rank": "avg_rank_overall"})
)
consensus_overall["overall_rank"] = consensus_overall["avg_rank_overall"].rank(ascending=True, method="min")
consensus_overall.to_csv(OUTDIR / "consensus_rank_overall.csv", index=False)

top20 = consensus_overall.sort_values("overall_rank").head(20)
plt.figure(figsize=(8, 6))
sns.barplot(data=top20, y="factor", x="avg_rank_overall", orient="h")
plt.gca().invert_yaxis()
plt.title("Top 20 Consensus Factors Across All Models and Years")
plt.xlabel("Average Rank (Lower = More Important)")
plt.tight_layout()
plt.savefig(OUTDIR / "top20_consensus_factors.png", dpi=300)
plt.close()

print("Consensus ranking completed and saved to →", OUTDIR)