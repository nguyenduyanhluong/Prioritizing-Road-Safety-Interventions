import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

df19 = pd.read_csv("outputs_covid_shift/prepared_2019.csv").fillna(0)
df20 = pd.read_csv("outputs_covid_shift/prepared_2020.csv").fillna(0)

X19, y19 = df19.drop(columns=["is_fatal"]), df19["is_fatal"]
X20, y20 = df20.drop(columns=["is_fatal"]), df20["is_fatal"]

def run_xgb(X, y):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1]),
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X, y)
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })
    return imp, model

imp19, model19 = run_xgb(X19, y19)
imp20, model20 = run_xgb(X20, y20)

imp_compare = imp19.merge(imp20, on="feature", suffixes=("_2019", "_2020"))
imp_compare["difference"] = imp_compare["importance_2020"] - imp_compare["importance_2019"]

imp_compare.to_csv("outputs_covid_shift/xgb_importance_comparison_2019_vs_2020.csv", index=False)

top_shifts = imp_compare.reindex(
    imp_compare["difference"].abs().sort_values(ascending=False).index
).head(20)

plt.figure(figsize=(10, 6))
plt.barh(
    top_shifts["feature"], 
    top_shifts["difference"], 
    color=np.where(top_shifts["difference"] > 0, "green", "red")
)
plt.xlabel("Feature Importance Shift (2020 - 2019)")
plt.title("Top XGBoost Feature Importance Shifts (2020 vs 2019)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs_covid_shift/xgb_shifts_2019_vs_2020.png", dpi=300)
plt.show()

print("\nTop 5 Positive Shifts (more important in 2020):")
print(imp_compare.sort_values("difference", ascending=False).head(5))

print("\nTop 5 Negative Shifts (less important in 2020):")
print(imp_compare.sort_values("difference", ascending=True).head(5))