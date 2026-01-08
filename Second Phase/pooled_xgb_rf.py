import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

df19 = pd.read_csv("outputs_covid_shift/prepared_2019.csv")
df20 = pd.read_csv("outputs_covid_shift/prepared_2020.csv")

df19["covid_dummy"] = 0
df20["covid_dummy"] = 1

df = pd.concat([df19, df20], ignore_index=True).fillna(0)

X = df.drop(columns=["is_fatal"])
y = df["is_fatal"]

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1]),
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)
xgb.fit(X, y)

xgb_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb.feature_importances_
}).sort_values("importance", ascending=False)
xgb_importance.to_csv("outputs_covid_shift/pooled_xgb_importance.csv", index=False)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)

rf_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
rf_importance.to_csv("outputs_covid_shift/pooled_rf_importance.csv", index=False)

def plot_importance(imp_df, title, filename):
    top_feats = imp_df.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top_feats["feature"], top_feats["importance"], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"outputs_covid_shift/{filename}", dpi=300)
    plt.show()

plot_importance(xgb_importance, "Top Features (Pooled XGBoost)", "pooled_xgb_importance.png")
plot_importance(rf_importance, "Top Features (Pooled Random Forest)", "pooled_rf_importance.png")

print("\nXGBoost - Top 5 Features:")
print(xgb_importance.head(5))

print("\nRandom Forest - Top 5 Features:")
print(rf_importance.head(5))