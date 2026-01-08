import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
OUTDIR = BASE_PATH / "outputs_step3b_importances"
OUTDIR.mkdir(exist_ok=True)
YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
RANDOM_STATE = 42

def build_nn(input_dim):
    model = Sequential([
        Dense(128, activation="relu", input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["AUC"])
    return model

for year in YEARS:
    df = pd.read_csv(BASE_PATH / f"{year}_prepared.csv", low_memory=False)
    y = df["is_fatal"].astype(int)
    X = df.drop(columns=["is_fatal", "C_SEV", "Unnamed: 0", "X"], errors="ignore")

    num_cols, cat_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    ct = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    X_train_t = ct.fit_transform(X_train)
    X_test_t = ct.transform(X_test)
    cat_features = ct.named_transformers_["cat"].get_feature_names_out(cat_cols)
    feature_names = list(num_cols) + list(cat_features)

    models = {
        "rf": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample"),
        "xgb": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
        "nn": build_nn(X_train_t.shape[1])
    }

    for mname, model in models.items():
        if mname == "nn":
            model.fit(X_train_t, y_train, epochs=10, batch_size=128, verbose=0, validation_data=(X_test_t, y_test))
            predict_fn = lambda X: model.predict(X).ravel()
        else:
            model.fit(X_train_t, y_train)
            predict_fn = lambda X: model.predict_proba(X)[:, 1]

        y_pred = predict_fn(X_test_t)
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f"{year} {mname.upper()} ROC-AUC={roc_auc:.3f}")

        sample_size = min(5000, X_test_t.shape[0])
        sample_idx = np.random.choice(X_test_t.shape[0], sample_size, replace=False)
        X_sample, y_sample = X_test_t[sample_idx], y_test.iloc[sample_idx]

        def score_fn(X):
            return roc_auc_score(y_sample, predict_fn(X))

        pi = permutation_importance(model, X_sample, y_sample, n_repeats=1, random_state=RANDOM_STATE, n_jobs=-1, scoring="roc_auc") if mname != "nn" else None
        if mname == "nn":
            base_score = score_fn(X_sample)
            imp_vals = []
            for j in range(X_sample.shape[1]):
                Xp = X_sample.copy()
                np.random.shuffle(Xp[:, j])
                imp_vals.append(base_score - score_fn(Xp))
            imp_vals = np.array(imp_vals)
            pi_df = pd.DataFrame({"feature": feature_names, "importance": imp_vals})
        else:
            pi_df = pd.DataFrame({"feature": feature_names, "importance": pi.importances_mean})

        pi_df.to_csv(OUTDIR / f"level_importance_{mname}_{year}.csv", index=False)

    print(f"{year}: saved level importances for RF, XGB, NN")