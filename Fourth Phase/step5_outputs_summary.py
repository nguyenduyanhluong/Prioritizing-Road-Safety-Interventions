import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
STEP3_DIR = BASE_PATH / "outputs_step3b_importances"
STEP4_DIR = BASE_PATH / "outputs_step4_visuals"
OUTDIR = BASE_PATH / "outputs_step5_final"
OUTDIR.mkdir(exist_ok=True)

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
MODELS = ["rf", "xgb", "nn"]

for year in YEARS:
    for model in MODELS:
        f = STEP3_DIR / f"level_importance_{model}_{year}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["factor"] = df["feature"].str.replace(r"_[0-9A-Za-z]+$", "", regex=True)
        grouped = df.groupby("factor")["importance"].sum().reset_index()
        grouped = grouped.sort_values("importance", ascending=False)
        grouped.to_csv(OUTDIR / f"factor_importance_grouped_{model}_{year}.csv", index=False)

stab_src = STEP4_DIR / "factor_stability.csv"
if stab_src.exists():
    stab_df = pd.read_csv(stab_src)
    stab_df.rename(columns={"mean_rank": "mean_rank_all", "rank_var": "rank_variance"}, inplace=True)
    stab_df.to_csv(OUTDIR / "cross_year_stability.csv", index=False)

consensus_overall = pd.read_csv(STEP4_DIR / "consensus_rank_overall.csv")
top20 = consensus_overall.sort_values("overall_rank").head(20)
stab_df = pd.read_csv(STEP4_DIR / "factor_stability.csv")
importance_df = pd.read_csv(STEP4_DIR / "method_agreement_by_year.csv")

importance_long = importance_df.melt(id_vars="year", var_name="model_pair", value_name="corr")

fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={"height_ratios": [2, 1]})
plt.subplots_adjust(hspace=0.4, wspace=0.3)

sns.barplot(ax=axes[0, 0], data=top20, y="factor", x="avg_rank_overall", orient="h")
axes[0, 0].invert_yaxis()
axes[0, 0].set_title("Top 20 Consensus Factors")
axes[0, 0].set_xlabel("Average Rank (Lower = More Important)")
axes[0, 0].set_ylabel("")

sns.scatterplot(ax=axes[0, 1], data=stab_df, x="mean_rank", y="rank_var", s=40)
axes[0, 1].set_title("Stability Across Years and Models")
axes[0, 1].set_xlabel("Mean Rank (Lower = More Important)")
axes[0, 1].set_ylabel("Variance of Rank (Lower = More Stable)")

pivot = importance_long.pivot(index="year", columns="model_pair", values="corr")
sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f", ax=axes[1, 0])
axes[1, 0].set_title("Method Agreement by Year (Spearman ρ)")
axes[1, 0].set_xlabel("Model Pair")
axes[1, 0].set_ylabel("Year")

axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(OUTDIR / "dashboard_summary.png", dpi=300)
plt.close()

print(f"Step 5 completed — grouped importances, stability file, and dashboard saved to {OUTDIR}")