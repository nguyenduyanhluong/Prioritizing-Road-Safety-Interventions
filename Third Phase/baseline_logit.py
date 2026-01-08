import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt

OUTDIR = Path("outputs_covid_shift")
OUTDIR.mkdir(exist_ok=True)

DATA_2019 = OUTDIR / "prepared_2019.csv"
DATA_2020 = OUTDIR / "prepared_2020.csv"

def clean_X_y(df):
    y = df["is_fatal"].values
    X = df.drop(columns=["is_fatal"], errors="ignore")

    X = X.replace([np.inf, -np.inf], np.nan)
    keep = ~X.isna().any(axis=1)
    X, y = X.loc[keep], y[keep]

    X = sm.add_constant(X, has_constant="add")

    before = X.shape[1]
    X = X.loc[:, X.nunique() > 1]
    X = X.loc[:, ~X.T.duplicated()]
    after = X.shape[1]

    print(f"Cleaned predictors: dropped {before - after} redundant columns")
    return X, y

def plot_top_predictors(results, label, top_n=15, drop_biggest=False):
    top_features = results.reindex(results["coef"].abs().sort_values(ascending=False).index)

    if drop_biggest:
        top_features = top_features.iloc[1:top_n+1]
    else:
        top_features = top_features.head(top_n)

    plt.figure(figsize=(8, 6))
    colors = top_features["direction"].map({"positive": "red", "negative": "blue"})
    plt.barh(top_features.index, np.log10(top_features["odds_ratio"]), color=colors)

    plt.axvline(x=0, color="black", linestyle="--")
    plt.xlabel("log10(Odds Ratio)")
    plt.title(f"Top Predictors of Fatality Risk ({label})")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    suffix = "_no_P_ISEV_3" if drop_biggest else ""
    plt.savefig(OUTDIR / f"logit_top_predictors_{label}{suffix}.png", dpi=300)
    plt.close()

def run_logit(df, label, alpha=1.0):
    X, y = clean_X_y(df)

    print(f"Fitting REGULARIZED Logit for {label}: {X.shape[0]} rows, {X.shape[1]} predictors")

    model = sm.Logit(y, X).fit_regularized(alpha=alpha, disp=False)

    results = pd.DataFrame({
        "coef": model.params,
    })
    results["odds_ratio"] = np.exp(results["coef"])
    results["direction"] = np.where(results["coef"] > 0, "positive", "negative")

    results.to_csv(OUTDIR / f"logit_results_{label}.csv")
    print(f"Saved results for {label} → {OUTDIR}/logit_results_{label}.csv")

    plot_top_predictors(results, label, drop_biggest=False)
    plot_top_predictors(results, label, drop_biggest=True) 
    return results

if __name__ == "__main__":
    df_2019 = pd.read_csv(DATA_2019)
    df_2020 = pd.read_csv(DATA_2020)
    df_combined = pd.concat([df_2019, df_2020], ignore_index=True)

    res_2019 = run_logit(df_2019, "2019")
    res_2020 = run_logit(df_2020, "2020")
    res_combined = run_logit(df_combined, "combined")

    print("\nTop predictors 2019:")
    print(res_2019.sort_values("odds_ratio", ascending=False).head(10))

    print("\nTop predictors 2020:")
    print(res_2020.sort_values("odds_ratio", ascending=False).head(10))

    print("\nTop predictors Combined:")
    print(res_combined.sort_values("odds_ratio", ascending=False).head(10))