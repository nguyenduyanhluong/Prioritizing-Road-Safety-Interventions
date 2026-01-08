import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
OUTDIR = BASE_PATH / "outputs_step2_descriptives"
PLOT_DIR = OUTDIR / "plots_sensitivity"
PLOT_DIR.mkdir(exist_ok=True)

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
sns.set(style="whitegrid", font_scale=1.2)

def plot_top_factors(year):
    fpath = OUTDIR / f"factor_summary_{year}.csv"
    df = pd.read_csv(fpath)
    if df.empty:
        print(f"Missing or empty file: {fpath}")
        return

    df_full = df.sort_values("range_rate", ascending=False).head(10)

    df_no_isev = df[~df["feature"].str.contains("ISEV", case=False, na=False)]
    df_no_isev = df_no_isev.sort_values("range_rate", ascending=False).head(10)

    merged = (
        pd.merge(
            df_full[["feature", "range_rate"]],
            df_no_isev[["feature", "range_rate"]],
            on="feature",
            suffixes=("_with", "_no"),
            how="outer",
        )
    )
    merged = merged.fillna(0)
    corr = spearmanr(merged["range_rate_with"], merged["range_rate_no"]).correlation
    print(f"{year}: Spearman correlation (with vs without ISEV) = {corr:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.barplot(
        y="feature",
        x="range_rate",
        data=df_full,
        palette="coolwarm",
        hue="weighted_var_rate",
        dodge=False,
        ax=axes[0],
    )
    axes[0].set_title(f"With ISEV ({year})")
    axes[0].set_xlabel("Range of Fatality Rate")
    axes[0].set_ylabel("Risk Factor")

    sns.barplot(
        y="feature",
        x="range_rate",
        data=df_no_isev,
        palette="coolwarm",
        hue="weighted_var_rate",
        dodge=False,
        ax=axes[1],
    )
    axes[1].set_title(f"Without ISEV ({year})")
    axes[1].set_xlabel("Range of Fatality Rate")
    axes[1].set_ylabel("")

    plt.suptitle(f"Sensitivity Analysis – Top 10 Risk Factors ({year})\nSpearman Corr = {corr:.2f}")
    plt.tight_layout()
    out_path = PLOT_DIR / f"sensitivity_top10_risk_factors_{year}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved → {out_path.name}")

def plot_combined():
    fpath = OUTDIR / "factor_summary_all_years.csv"
    df = pd.read_csv(fpath)
    if df.empty:
        print("Missing all-years summary.")
        return

    df_no_isev = df[~df["feature"].str.contains("ISEV", case=False, na=False)]
    top_full = df.groupby("feature")["range_rate"].mean().sort_values(ascending=False).head(10).index
    top_no_isev = df_no_isev.groupby("feature")["range_rate"].mean().sort_values(ascending=False).head(10).index

    merged = (
        df.groupby("feature")["range_rate"].mean().rename("with")
        .to_frame()
        .join(df_no_isev.groupby("feature")["range_rate"].mean().rename("no"), how="outer")
        .fillna(0)
    )
    corr = spearmanr(merged["with"], merged["no"]).correlation

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.barplot(x="year", y="range_rate", hue="feature", data=df[df["feature"].isin(top_full)], ax=axes[0])
    axes[0].set_title("With ISEV")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Range of Fatality Rate")

    sns.barplot(x="year", y="range_rate", hue="feature", data=df_no_isev[df_no_isev["feature"].isin(top_no_isev)], ax=axes[1])
    axes[1].set_title("Without ISEV")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("")

    plt.suptitle(f"Top 10 Risk Factors Across Years\nSpearman Corr (With vs Without ISEV) = {corr:.2f}")
    plt.tight_layout()
    out_path = PLOT_DIR / "sensitivity_top10_risk_factors_across_years.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved → {out_path.name}")

for year in YEARS:
    plot_top_factors(year)

plot_combined()
print("All sensitivity analysis plots created successfully.")