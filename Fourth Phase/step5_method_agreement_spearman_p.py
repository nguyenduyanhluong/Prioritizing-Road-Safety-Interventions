import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
INPUT_DIR = BASE_PATH / "outputs_step3b_importances"
OUTDIR = BASE_PATH / "outputs_step5_final"
OUTDIR.mkdir(exist_ok=True)

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
MODELS = ["rf", "xgb", "nn"]

records = []
for year in YEARS:
    dfs = {}
    for model in MODELS:
        f = INPUT_DIR / f"level_importance_{model}_{year}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["rank"] = df["importance"].rank(ascending=False, method="min")
        dfs[model.upper()] = df.set_index("feature")["rank"]

    if {"RF", "XGB", "NN"}.issubset(dfs.keys()):
        merged = pd.concat(dfs, axis=1)
        rho_rf_xgb = spearmanr(merged["RF"], merged["XGB"]).correlation
        rho_rf_nn = spearmanr(merged["RF"], merged["NN"]).correlation
        rho_xgb_nn = spearmanr(merged["XGB"], merged["NN"]).correlation

        records.append({
            "year": year,
            "RF_XGB": rho_rf_xgb,
            "RF_NN": rho_rf_nn,
            "XGB_NN": rho_xgb_nn
        })

if records:
    result_df = pd.DataFrame(records)
    result_df.to_csv(OUTDIR / "method_agreement_by_year.csv", index=False)
    print(f"Spearman method agreement by year saved to: {OUTDIR / 'method_agreement_by_year.csv'}")

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        result_df.set_index("year"),
        annot=True, cmap="YlGnBu", fmt=".2f"
    )
    plt.title("Method Agreement by Year (Spearman ρ)")
    plt.xlabel("Model Pair")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(OUTDIR / "method_agreement_by_year_heatmap.png", dpi=300)
    plt.close()
    print(f"Heatmap saved to: {OUTDIR / 'method_agreement_by_year_heatmap.png'}")
else:
    print("No complete year data available (RF, XGB, NN all required).")