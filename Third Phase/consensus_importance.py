import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("outputs_covid_shift")

def load_results(model, year):
    path = OUTDIR / f"{model}_results_{year}.csv"
    if not path.exists():
        print(f"Missing: {path}")
        return None

    df = pd.read_csv(path)

    if model == "logit":
        df = df.rename(columns={"Unnamed: 0": "feature", "coef": "value"})
        df["importance"] = df["value"].abs()
        df = df[["feature", "importance"]]

    elif model in ["xgb", "rf"]:
        df = df[["feature", "importance"]]

    else:
        return None

    df["model"] = model
    df["year"] = year
    return df

def collect_all(years=["2019", "2020", "combined"]):
    dfs = []
    for model in ["logit", "xgb", "rf"]:
        for year in years:
            df = load_results(model, year)
            if df is not None:
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def build_consensus(df):
    df["rank"] = df.groupby(["model", "year"])["importance"].rank(
        ascending=False, method="dense"
    )

    agg = (
        df.groupby("feature")
        .agg(mean_rank=("rank", "mean"), count=("rank", "size"))
        .reset_index()
    )

    agg = agg.sort_values("mean_rank", ascending=True)
    return agg

def plot_heatmap(df, top_n=20):
    pivot = (
        df.pivot_table(
            index="feature",
            columns=["model", "year"],
            values="rank"
        )
    )

    top_features = (
        df.groupby("feature")["rank"]
        .mean()
        .nsmallest(top_n)
        .index
    )
    pivot = pivot.loc[top_features]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        cmap="viridis_r",
        cbar_kws={"label": "Rank"},
        fmt=".0f"
    )
    plt.title(f"Consensus Feature Ranks (Top {top_n})")
    plt.tight_layout()
    plt.savefig(OUTDIR / "consensus_feature_ranks_heatmap.png", dpi=300)
    plt.close()

all_df = collect_all()
consensus = build_consensus(all_df)

all_df.to_csv(OUTDIR / "all_feature_importance_long.csv", index=False)
consensus.to_csv(OUTDIR / "consensus_feature_importance.csv", index=False)

plot_heatmap(all_df, top_n=20)

print("Saved:")
print(f"- {OUTDIR}/all_feature_importance_long.csv (long format)")
print(f"- {OUTDIR}/consensus_feature_importance.csv (consensus ranking)")
print(f"- {OUTDIR}/consensus_feature_ranks_heatmap.png (visualization)")

print("\nTop 20 consensus features:")
print(consensus.head(20))