import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

OUTDIR = Path("outputs_covid_shift")
OUTDIR.mkdir(exist_ok=True)

DATA_2019 = OUTDIR / "prepared_2019.csv"
DATA_2020 = OUTDIR / "prepared_2020.csv"
df_combined = pd.concat(
    [pd.read_csv(DATA_2019), pd.read_csv(DATA_2020)], ignore_index=True
)

def clean_X_y(df):
    y = df["is_fatal"].values
    X = df.drop(columns=["is_fatal"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y

X, y = clean_X_y(df_combined)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor_nn = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ]
)

models = {
    "XGBoost": Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="auc",
            n_jobs=-1,
            random_state=42,
            tree_method="hist"
        ))
    ]),
    "Random Forest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample"
        ))
    ]),
    "Neural Network": Pipeline([
        ("pre", preprocessor_nn),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=50,
            random_state=42,
            verbose=False
        ))
    ]),
}

plt.figure(figsize=(7, 6))

for name, pipe in models.items():
    print(f"Training {name}...")
    pipe.fit(X_train, y_train)
    y_score = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_val = roc_auc_score(y_test, y_score)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Combined ROC Curves (XGBoost, Random Forest, NN)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

save_path = OUTDIR / "combined_models_roc.png"
plt.savefig(save_path, dpi=400)
print(f"Saved combined ROC figure → {save_path}")
plt.show()