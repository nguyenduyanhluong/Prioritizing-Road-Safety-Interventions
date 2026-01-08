# STAGE 2: Handling Class Imbalance (NCDB 2020)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier
from xgboost import XGBClassifier

# 1. Load & preprocess (from Stage 1)
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

print("After dropping NaNs:", X.shape, y.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Logistic Regression (Balanced Weights)
logreg_bal = LogisticRegression(max_iter=5000, class_weight="balanced")
logreg_bal.fit(X_train_scaled, y_train)
y_pred_log = logreg_bal.predict(X_test_scaled)
print("\n--- Logistic Regression (Balanced Weights) ---")
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, logreg_bal.predict_proba(X_test_scaled)[:,1]))

# 3. XGBoost (scale_pos_weight)
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("\n--- XGBoost (scale_pos_weight) ---")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]))

# 4. Logistic Regression + SMOTE Oversampling
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

logreg_smote = LogisticRegression(max_iter=5000)
logreg_smote.fit(X_train_sm, y_train_sm)
y_pred_sm = logreg_smote.predict(X_test_scaled)
print("\n--- Logistic Regression (SMOTE Oversampling) ---")
print(classification_report(y_test, y_pred_sm))
print("ROC-AUC:", roc_auc_score(y_test, logreg_smote.predict_proba(X_test_scaled)[:,1]))

# 5. Logistic Regression + Undersampling
rus = RandomUnderSampler(random_state=42)
X_train_us, y_train_us = rus.fit_resample(X_train_scaled, y_train)

logreg_us = LogisticRegression(max_iter=5000)
logreg_us.fit(X_train_us, y_train_us)
y_pred_us = logreg_us.predict(X_test_scaled)
print("\n--- Logistic Regression (Random Undersampling) ---")
print(classification_report(y_test, y_pred_us))
print("ROC-AUC:", roc_auc_score(y_test, logreg_us.predict_proba(X_test_scaled)[:,1]))

# 6. EasyEnsemble Classifier
easy = EasyEnsembleClassifier(n_estimators=10, random_state=42)
easy.fit(X_train, y_train)
y_pred_easy = easy.predict(X_test)
print("\n--- EasyEnsemble Classifier ---")
print(classification_report(y_test, y_pred_easy))
print("ROC-AUC:", roc_auc_score(y_test, easy.predict_proba(X_test)[:,1]))

# 7. Balanced Random Forest
brf = BalancedRandomForestClassifier(n_estimators=200, random_state=42)
brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)
print("\n--- Balanced Random Forest ---")
print(classification_report(y_test, y_pred_brf))
print("ROC-AUC:", roc_auc_score(y_test, brf.predict_proba(X_test)[:,1]))