# Balanced RF & XGBoost with PR Curves + Confusion Matrices

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("y_2020_en.csv", low_memory=False)

num_like_cols = ['C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS']
for col in num_like_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

drop_cols = ['V_ID', 'P_ID', 'C_CASE']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

cat_cols = df.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

y = (df['C_SEV'] == 1).astype(int)
X = df_encoded.drop(columns=['C_SEV'])

mask = ~X.isna().any(axis=1)
X = X.loc[mask]
y = y.loc[mask]

print("Final dataset shape:", X.shape, y.shape)
print("Fatal collision proportion:", y.mean())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balanced Random Forest
brf = BalancedRandomForestClassifier(n_estimators=200, random_state=42)
brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)
y_prob_brf = brf.predict_proba(X_test)[:, 1]

print("\n--- Balanced Random Forest ---")
print(classification_report(y_test, y_pred_brf, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_brf))

# XGBoost (scale_pos_weight)
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("\n--- XGBoost (scale_pos_weight) ---")
print(classification_report(y_test, y_pred_xgb, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

# Precision-Recall Curves
plt.figure(figsize=(8,6))

precision_brf, recall_brf, _ = precision_recall_curve(y_test, y_prob_brf)
ap_brf = average_precision_score(y_test, y_prob_brf)
plt.plot(recall_brf, precision_brf, label=f"Balanced RF (AP={ap_brf:.3f})")

precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
ap_xgb = average_precision_score(y_test, y_prob_xgb)
plt.plot(recall_xgb, precision_xgb, label=f"XGBoost (AP={ap_xgb:.3f})")

no_skill = sum(y_test) / len(y_test)
plt.hlines(no_skill, 0, 1, colors="gray", linestyles="--", label="No Skill")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves (Fatal Collisions, 2020)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12,5))

cm_brf = confusion_matrix(y_test, y_pred_brf)
sns.heatmap(cm_brf, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Balanced Random Forest")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Oranges", ax=axes[1])
axes[1].set_title("XGBoost")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()