import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier

OUTDIR = Path("outputs_covid_shift")
OUTDIR.mkdir(exist_ok=True)

DATA_2019 = OUTDIR / "prepared_2019.csv"
DATA_2020 = OUTDIR / "prepared_2020.csv"

def clean_X_y(df):
    y = df["is_fatal"].values
    X = df.drop(columns=["is_fatal"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y

def plot_curves(y_true, y_score, label):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_roc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"nn_roc_{label}.png", dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(recall, precision, label=f"AUC = {auc_pr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"nn_pr_{label}.png", dpi=300)
    plt.close()

    return auc_roc, auc_pr

def run_nn(df, label, test_size=0.2, seed=42):
    X, y = clean_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=50,
        random_state=seed,
        verbose=True
    )

    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", model)])

    print(f"Fitting Neural Network for {label}: {X_train.shape[0]} rows, {X_train.shape[1]} predictors")
    pipe.fit(X_train, y_train)

    y_score = pipe.predict_proba(X_test)[:, 1]

    auc_roc, auc_pr = plot_curves(y_test, y_score, label)

    results = pd.DataFrame({
        "metric": ["ROC AUC", "PR AUC"],
        "value": [auc_roc, auc_pr]
    })

    results.to_csv(OUTDIR / f"nn_results_{label}.csv", index=False)
    print(f"Saved results for {label} → {OUTDIR}/nn_results_{label}.csv")
    print(f"ROC AUC: {auc_roc:.4f}, PR AUC: {auc_pr:.4f}")

    return results

df_2019 = pd.read_csv(DATA_2019)
df_2020 = pd.read_csv(DATA_2020)
df_combined = pd.concat([df_2019, df_2020], ignore_index=True)

res_2019 = run_nn(df_2019, "2019")
res_2020 = run_nn(df_2020, "2020")
res_combined = run_nn(df_combined, "combined")

print("\nNN results 2019:")
print(res_2019)

print("\nNN results 2020:")
print(res_2020)

print("\nNN results Combined:")
print(res_combined)