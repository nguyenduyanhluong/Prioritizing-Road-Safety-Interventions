import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
OUTDIR = BASE_PATH / "outputs_step2_descriptives"
PLOT_DIR = OUTDIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

sns.set(style="whitegrid", font_scale=1.2)

def plot_top_factors(year):
    fpath = OUTDIR / f"factor_summary_{year}.csv"
    df = pd.read_csv(fpath)
    df = df.sort_values("range_rate", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        y="feature",
        x="range_rate",
        data=df,
        palette="coolwarm",
        hue="weighted_var_rate",
        dodge=False
    )
    plt.title(f"Top 10 Risk Factors by Fatality Rate Range ({year})")
    plt.xlabel("Range of Fatality Rate (Max - Min)")
    plt.ylabel("Risk Factor")
    plt.legend(title="Weighted Variance", loc="best")
    plt.tight_layout()
    out_path = PLOT_DIR / f"top10_risk_factors_{year}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved → {out_path.name}")

def plot_combined():
    fpath = OUTDIR / "factor_summary_all_years.csv"
    df = pd.read_csv(fpath)
    top_factors = df.groupby("feature")["range_rate"].mean().sort_values(ascending=False).head(10).index
    df = df[df["feature"].isin(top_factors)]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="year",
        y="range_rate",
        hue="feature",
        data=df,
    )
    plt.title("Top 10 Risk Factors - Variation Across Years")
    plt.xlabel("Year")
    plt.ylabel("Range of Fatality Rate")
    plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_path = PLOT_DIR / "top10_risk_factors_across_years.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved → {out_path.name}")

for year in YEARS:
    plot_top_factors(year)

plot_combined()
print("All risk-factor visualization PNGs created successfully.")