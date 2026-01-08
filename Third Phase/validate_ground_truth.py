import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt

OUTDIR = Path("outputs_covid_shift")
OUTDIR.mkdir(exist_ok=True)

DATA_2019 = OUTDIR / "prepared_2019.csv"
DATA_2020 = OUTDIR / "prepared_2020.csv"
CONSENSUS_FILE = OUTDIR / "consensus_with_direction.csv"

TARGET = "is_fatal"
TOP_N = 20

print("Loading datasets...")
df19 = pd.read_csv(DATA_2019, low_memory=False)
df20 = pd.read_csv(DATA_2020, low_memory=False)

print(f"2019 → {df19.shape}, target positives={df19[TARGET].sum()}")
print(f"2020 → {df20.shape}, target positives={df20[TARGET].sum()}")

dfc = pd.concat([df19.assign(year=2019), df20.assign(year=2020)], ignore_index=True)

consensus = pd.read_csv(CONSENSUS_FILE)
top_features = consensus["feature"].head(TOP_N).tolist()

available_features = [f for f in top_features if f in df19.columns and f in df20.columns]
missing = set(top_features) - set(available_features)
if missing:
    print(f"Skipping {len(missing)} missing features: {list(missing)[:10]}...")

print(f"Using {len(available_features)} features for validation")

def odds_ratio_and_pval(df, feature, target=TARGET):
    if feature not in df.columns:
        return np.nan, np.nan

    series = df[feature]
    if series.dtype == "O" or len(series.dropna().unique()) <= 10:
        ctab = pd.crosstab(series, df[target])
        if ctab.shape == (2, 2):
            (a, b), (c, d) = ctab.values
            or_val = (a * d) / (b * c + 1e-9)
            _, pval = fisher_exact(ctab)
        else:
            or_val = np.nan
            _, pval, _, _ = chi2_contingency(ctab)
    else:
        group0 = df[df[target] == 0][feature].dropna()
        group1 = df[df[target] == 1][feature].dropna()
        if len(group0) > 0 and len(group1) > 0:
            _, pval_t = ttest_ind(group0, group1, equal_var=False)
            _, pval_mw = mannwhitneyu(group0, group1, alternative="two-sided")
            pval = min(pval_t, pval_mw)
            or_val = (group1.mean() + 1e-9) / (group0.mean() + 1e-9)
        else:
            or_val, pval = np.nan, np.nan
    return or_val, pval

def sign_from_or(or_val, threshold=1e-6):
    if pd.isna(or_val):
        return 0
    if or_val > 1 + threshold:
        return +1
    elif or_val < 1 - threshold:
        return -1
    else:
        return 0

print("Running validation...")
results = []

for feat in available_features:
    for year, df in [("2019", df19), ("2020", df20), ("Combined", dfc)]:
        try:
            or_emp, pval = odds_ratio_and_pval(df, feat, target=TARGET)
            sign_emp = sign_from_or(or_emp)
        except Exception:
            or_emp, pval, sign_emp = np.nan, np.nan, 0

        results.append({
            "feature": feat,
            "year": year,
            "empirical_OR": or_emp,
            "pval": pval,
            "empirical_sign": sign_emp
        })

results_df = pd.DataFrame(results)

merged = results_df.merge(
    consensus[["feature", "coef", "odds_ratio", "direction"]],
    on="feature", how="left"
)

def mismatch_flag(row):
    if row["empirical_sign"] == 0 or pd.isna(row["direction"]):
        return "neutral"
    return "match" if ((row["empirical_sign"] == 1 and row["direction"] == "positive") or
                       (row["empirical_sign"] == -1 and row["direction"] == "negative")) else "mismatch"

merged["match_flag"] = merged.apply(mismatch_flag, axis=1)

out_csv = OUTDIR / "validation_results.csv"
merged.to_csv(out_csv, index=False)
print(f"Saved validation table with mismatch flags → {out_csv}")

print("Creating stability heatmap...")
heatmap_data = results_df.pivot(index="feature", columns="year", values="empirical_sign")

stability = (heatmap_data.nunique(axis=1) == 1).astype(int)
heatmap_data = heatmap_data.loc[stability.sort_values(ascending=False).index]

plt.figure(figsize=(10, len(heatmap_data) * 0.4))
sns.heatmap(
    heatmap_data.replace({1: 1, -1: -1, 0: 0}),
    annot=True, fmt=".0f", cmap="coolwarm", cbar=False,
    center=0, linewidths=0.5, linecolor="gray"
)

plt.title("Feature Sign Stability Across Years\n(+1=risk, -1=protective, 0=neutral)")
plt.ylabel("Feature (sorted by stability)")
plt.xlabel("Year")
plt.tight_layout()

out_png = OUTDIR / "feature_sign_stability_heatmap.png"
plt.savefig(out_png, dpi=300)
plt.close()
print(f"Saved stability-sorted heatmap → {out_png}")