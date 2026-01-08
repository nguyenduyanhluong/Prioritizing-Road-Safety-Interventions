import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
OUTDIR = BASE_PATH / "outputs_step2_descriptives"
PLOT_DIR = OUTDIR / "plots_level"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
sns.set(style="whitegrid", font_scale=1.2)

def extract_level_value(df):
    level_cols = [c for c in df.columns if c.endswith("_level")]
    level_map = {c.replace("_level", "").strip().upper(): c for c in level_cols}

    def get_level(row):
        feat = str(row.get("feature", "")).strip().upper()
        col = level_map.get(feat)
        if col in df.columns:
            val = row.get(col, None)
            return val if pd.notna(val) else None
        return None

    df["level_value"] = df.apply(get_level, axis=1)
    df["level_value"] = df["level_value"].astype(str).replace(["nan", "None", ""], "Missing")
    df = df[~df["level_value"].isin(["*", "**", "Missing", "U", "UU"])]
    return df

def plot_top_levels(year):
    fpath = OUTDIR / f"level_risk_summary_{year}.csv"
    if not fpath.exists():
        print(f"Missing file: {fpath.name}")
        return
    df = pd.read_csv(fpath)
    df = extract_level_value(df)
    df = df[df["RR_vs_overall"].notna()]
    df["level_name"] = df["feature"] + " = " + df["level_value"]
    df = df.sort_values("RR_vs_overall", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(y="level_name", x="RR_vs_overall", hue="level_name", data=df, legend=False, palette="Reds_r", errorbar=None)
    plt.title(f"Top 10 Levels by Relative Risk (RR_vs_overall) ({year})")
    plt.xlabel("Relative Risk (vs Overall)")
    plt.ylabel("Feature Level")
    plt.tight_layout()
    out_path = PLOT_DIR / f"top10_levels_{year}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved → {out_path.name}")

def plot_levels_across_years():
    all_files = [OUTDIR / f"level_risk_summary_{y}.csv" for y in YEARS if (OUTDIR / f"level_risk_summary_{y}.csv").exists()]
    if not all_files:
        print("No level_risk_summary CSVs found.")
        return
    df_all = pd.concat((pd.read_csv(f).assign(year=int(f.stem.split('_')[-1])) for f in all_files), ignore_index=True)
    df_all = extract_level_value(df_all)
    df_all = df_all[df_all["RR_vs_overall"].notna()]
    df_all["level_name"] = df_all["feature"] + " = " + df_all["level_value"]
    mean_rates = df_all.groupby("level_name")["RR_vs_overall"].mean().sort_values(ascending=False).head(10)
    top_levels = mean_rates.index
    df_top = df_all[df_all["level_name"].isin(top_levels)]
    summary_table = (
        df_top.pivot_table(index="level_name", columns="year", values="RR_vs_overall", aggfunc="mean")
        .assign(mean_rate=mean_rates)
        .sort_values("mean_rate", ascending=False)
    )
    plt.figure(figsize=(12, 6))
    sns.heatmap(summary_table.iloc[:, :-1], annot=True, fmt=".3f", cmap="Reds", cbar_kws={"label": "Relative Risk"})
    plt.title("Top 10 Feature-Level Relative Risks Across Years (RR_vs_overall)")
    plt.xlabel("Year")
    plt.ylabel("Feature = Level")
    plt.tight_layout()
    out_path = PLOT_DIR / "top10_levels_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved → {out_path.name}")
    summary_table.to_csv(PLOT_DIR / "top10_levels_summary_table.csv")
    print("Saved → top10_levels_summary_table.csv")

for y in YEARS:
    plot_top_levels(y)

plot_levels_across_years()
print("All level-based visualization PNGs and summary table created successfully.")