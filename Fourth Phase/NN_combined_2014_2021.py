import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
RANDOM_STATE = 42
YEARS = list(range(2014, 2022))
OUTDIR = BASE_PATH / "outputs_step_3_NN_combined_2014_2021"
OUTDIR.mkdir(exist_ok=True, parents=True)

roc_results = {}
pr_results = {}
history_records = []

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

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
    input_dim = X_train_t.shape[1]
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train_t, y_train,
        validation_data=(X_test_t, y_test),
        epochs=10,
        batch_size=128,
        callbacks=[early_stop],
        verbose=0
    )
    y_pred_prob = model.predict(X_test_t, verbose=0).ravel()
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(rec, prec)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_results[year] = (fpr, tpr, roc_auc)
    pr_results[year] = (rec, prec, pr_auc)
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = np.arange(len(history_df))
    history_df["year"] = year
    history_records.append(history_df)
    print(f"Neural Network {year} completed | ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}")

plt.figure(figsize=(8, 6))
for year, (fpr, tpr, roc_auc) in roc_results.items():
    plt.plot(fpr, tpr, label=f"{year} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Neural Network (2014–2021)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "NN_ROC_AUC_2014_2021.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
for year, (rec, prec, pr_auc) in pr_results.items():
    plt.plot(rec, prec, label=f"{year} (AUC={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curves - Neural Network (2014–2021)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "NN_PR_AUC_2014_2021.png", dpi=300)
plt.close()

history_all = pd.concat(history_records)
avg_history = history_all.groupby("epoch")[["loss", "val_loss"]].mean()
plt.figure(figsize=(8, 5))
plt.plot(avg_history.index, avg_history["loss"], label='Avg Train Loss')
plt.plot(avg_history.index, avg_history["val_loss"], label='Avg Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Average Training History - Neural Network (2014–2021)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "NN_Training_History_2014_2021.png", dpi=300)
plt.close()

print("\nCombined plots saved in:")
print(OUTDIR)