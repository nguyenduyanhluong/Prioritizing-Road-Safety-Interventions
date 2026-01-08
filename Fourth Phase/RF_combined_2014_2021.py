import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
RANDOM_STATE = 42
N_REPEATS_PI = 1
YEARS = list(range(2014, 2022))
OUTDIR = BASE_PATH / "outputs_step_3_RF_combined_2014_2021"
OUTDIR.mkdir(exist_ok=True, parents=True)

roc_results = {}
pr_results = {}
feature_importances_all = []

for year in YEARS:
    print(f"\n=== Processing {year} ===")
    df = pd.read_csv(BASE_PATH / f"{year}_prepared.csv", low_memory=False)
    y = df["is_fatal"].astype(int)
    drop_cols = ["is_fatal", "C_SEV"]
    for c in ["Unnamed: 0", "X"]:
        if c in df.columns:
            drop_cols.append(c)
    X = df.drop(columns=drop_cols, errors="ignore")
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    ct = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_train_t = ct.fit_transform(X_train)
    X_test_t = ct.transform(X_test)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample"
    )
    rf.fit(X_train_t, y_train)
    y_pred_prob = rf.predict_proba(X_test_t)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(rec, prec)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_results[year] = (fpr, tpr, roc_auc)
    pr_results[year] = (rec, prec, pr_auc)
    sample_size = min(10000, X_test_t.shape[0])
    sample_idx = np.random.choice(X_test_t.shape[0], sample_size, replace=False)
    X_sample = X_test_t[sample_idx]
    y_sample = y_test.iloc[sample_idx].copy()
    pi = permutation_importance(
        rf, X_sample, y_sample,
        n_repeats=N_REPEATS_PI,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    cat_features = ct.named_transformers_["cat"].get_feature_names_out(cat_cols)
    feature_names = list(num_cols) + list(cat_features)
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": pi.importances_mean,
        "year": year
    }).sort_values("importance", ascending=False).head(20)
    feature_importances_all.append(importances)
    print(f"Random Forest {year} completed | ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}")

plt.figure(figsize=(8, 6))
for year, (fpr, tpr, roc_auc) in roc_results.items():
    plt.plot(fpr, tpr, label=f"{year} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Random Forest (2014–2021)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "RF_ROC_AUC_2014_2021.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
for year, (rec, prec, pr_auc) in pr_results.items():
    plt.plot(rec, prec, label=f"{year} (AUC={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curves - Random Forest (2014–2021)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "RF_PR_AUC_2014_2021.png", dpi=300)
plt.close()

all_importances_df = pd.concat(feature_importances_all)
avg_importances = (
    all_importances_df.groupby("feature")["importance"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 6))
plt.barh(avg_importances.index, avg_importances.values)
plt.gca().invert_yaxis()
plt.title("Average Top 20 Feature Importances (2014–2021) - Random Forest")
plt.tight_layout()
plt.savefig(OUTDIR / "RF_Top20_Importances_2014_2021.png", dpi=300)
plt.close()

print("\nCombined plots saved in:")
print(OUTDIR)