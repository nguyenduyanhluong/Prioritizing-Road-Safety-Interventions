import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

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

corr_records = []
for year in YEARS:
    sub = all_factors[all_factors["year"] == year]
    pivot = sub.pivot_table(index="factor", columns="model", values="rank")
    if {"RF", "XGB", "NN"}.issubset(pivot.columns):
        corr_records.append({
            "year": year,
            "RF_XGB": spearmanr(pivot["RF"], pivot["XGB"]).correlation,
            "RF_NN": spearmanr(pivot["RF"], pivot["NN"]).correlation,
            "XGB_NN": spearmanr(pivot["XGB"], pivot["NN"]).correlation,
        })
pd.DataFrame(corr_records).to_csv(OUTDIR / "method_agreement_by_year.csv", index=False)

heatmap_data = all_factors.groupby(["year", "model"])["importance"].sum().unstack("model").fillna(0)
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="crest")
plt.title("Total Importance by Model and Year")
plt.tight_layout()
plt.savefig(OUTDIR / "heatmap_total_importance.png", dpi=300)
plt.close()

mean_ranks = all_factors.groupby("factor")["rank"].mean()
var_ranks = all_factors.groupby("factor")["rank"].var()
stability_df = pd.DataFrame({"mean_rank": mean_ranks, "rank_var": var_ranks}).reset_index()
stability_df.to_csv(OUTDIR / "factor_stability.csv", index=False)

stability_df["stability_score"] = stability_df["mean_rank"] + stability_df["rank_var"].rank()
top5 = stability_df.nsmallest(5, "stability_score")

print("\nTop 5 Most Stable & Important Factors:")
print(top5[["factor", "mean_rank", "rank_var", "stability_score"]].to_string(index=False))

plt.figure(figsize=(10, 7))
scatter = sns.scatterplot(
    data=stability_df,
    x="mean_rank",
    y="rank_var",
    hue="mean_rank",
    palette="viridis_r",
    size=-stability_df["rank_var"],
    sizes=(20, 200),
    alpha=0.7,
    legend=False
)
plt.title("Stability of Factors Across Years and Models\n(Top 5 Most Stable & Important Highlighted)")
plt.xlabel("Mean Rank (Lower = More Important)")
plt.ylabel("Variance of Rank (Lower = More Stable)")

colors = sns.color_palette("Reds", n_colors=5)

for i, ((_, row), c) in enumerate(zip(top5.iterrows(), colors)):
    x, y = row["mean_rank"], row["rank_var"]
    offset_x = 40 if i % 2 == 0 else -60
    offset_y = 2500 if i % 2 == 0 else -2500
    plt.annotate(
        row["factor"],
        xy=(x, y),
        xytext=(x + offset_x, y + offset_y),
        textcoords="data",
        fontsize=10,
        fontweight="bold",
        color=c,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color=c, lw=1)
    )
    plt.scatter(x, y, color=c, s=120, edgecolor="black", linewidth=0.5, zorder=5, label=row["factor"])

handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels[-5:], handles[-5:]))
plt.legend(unique.values(), unique.keys(), title="Top 5 Stable Factors", loc="upper right", frameon=True)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(OUTDIR / "stability_scatter_enhanced_top5_legend.png", dpi=300, bbox_inches="tight")
plt.close()

top_factors = (
    all_factors.groupby("factor")["importance"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .index
)
filtered = all_factors[all_factors["factor"].isin(top_factors)]
pivot_heat = filtered.pivot_table(index="factor", columns=["year", "model"], values="rank")

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_heat, cmap="viridis", cbar_kws={"label": "Rank (Lower = More Important)"})
plt.title("Rank Heatmap of Top 20 Factors by Year and Model")
plt.tight_layout()
plt.savefig(OUTDIR / "rank_heatmap_top20.png", dpi=300)
plt.close()

print(f"\nEnhanced stability_scatter_enhanced_top5_legend.png saved to → {OUTDIR}")