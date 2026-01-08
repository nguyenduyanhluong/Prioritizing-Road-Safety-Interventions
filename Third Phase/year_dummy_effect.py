import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUTDIR = Path("outputs_covid_shift")
OUTDIR.mkdir(exist_ok=True)

PREP_2019 = OUTDIR / "prepared_2019.csv"
PREP_2020 = OUTDIR / "prepared_2020.csv"
CONSENSUS_FILE = OUTDIR / "consensus_feature_importance.csv"

print("Loading datasets...")
X19 = pd.read_csv(PREP_2019, low_memory=False)
X20 = pd.read_csv(PREP_2020, low_memory=False)

def extract_target(df):
    candidates = [c for c in df.columns if "ISEV" in c or "SEV" in c or "FATAL" in c]
    if candidates:
        y = df[candidates[0]]
    else:
        y = df.iloc[:, -1]
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    y = (y > 0).astype(int)
    return y

y19 = extract_target(X19)
y20 = extract_target(X20)

print(f"2019 → {X19.shape}, positives={y19.sum()}, negatives={(y19==0).sum()}, total={len(y19)}")
print(f"2020 → {X20.shape}, positives={y20.sum()}, negatives={(y20==0).sum()}, total={len(y20)}")

consensus = pd.read_csv(CONSENSUS_FILE)
top_features = consensus["feature"].tolist()
print(f"Using {len(top_features)} consensus features")

X19 = X19[[f for f in top_features if f in X19.columns]].copy()
X20 = X20[[f for f in top_features if f in X20.columns]].copy()

X19["year_2020"] = 0
X20["year_2020"] = 1

X_combined = pd.concat([X19, X20], axis=0).reset_index(drop=True)
y_combined = pd.concat([y19, y20], axis=0).reset_index(drop=True)

print(f"Combined dataset → {X_combined.shape}, target unique={np.unique(y_combined)}")

X_combined = X_combined.fillna(0)

if len(np.unique(y_combined)) < 2:
    raise ValueError(f"Target still has only one class → unique={np.unique(y_combined)}. Please check encoding!")

logit = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=500,
        n_jobs=-1
    )
)

print("Fitting regularized logistic regression...")
logit.fit(X_combined, y_combined)

coefs = logit.named_steps["logisticregression"].coef_[0]
features = X_combined.columns
coef_df = pd.DataFrame({
    "feature": features,
    "coef": coefs,
    "odds_ratio": np.exp(coefs)
}).sort_values("coef", key=abs, ascending=False)

coef_csv = OUTDIR / "year_dummy_effect_logit.csv"
coef_df.to_csv(coef_csv, index=False)
print(f"Saved results → {coef_csv}")

if "year_2020" in coef_df["feature"].values:
    row = coef_df[coef_df["feature"] == "year_2020"].iloc[0]
    print("\nYear Dummy Coefficient:")
    print(f"Coef = {row['coef']:.4f}, OR = {row['odds_ratio']:.4f}")