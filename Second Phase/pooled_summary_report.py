import pandas as pd
import matplotlib.pyplot as plt
import os

logit_file = "outputs_covid_shift/pooled_logit_interactions.csv"
rf_file = "outputs_covid_shift/pooled_rf_importance.csv"
xgb_file = "outputs_covid_shift/pooled_xgb_importance.csv"
nn_file = "outputs_covid_shift/pooled_nn_importance.csv"

dfs = {}

if os.path.exists(logit_file):
    dfs["Logit"] = pd.read_csv(logit_file)
if os.path.exists(rf_file):
    dfs["RandomForest"] = pd.read_csv(rf_file)
if os.path.exists(xgb_file):
    dfs["XGBoost"] = pd.read_csv(xgb_file)
if os.path.exists(nn_file):
    dfs["NeuralNet"] = pd.read_csv(nn_file)

summary = {}
for model, df in dfs.items():
    if model == "Logit":
        df_clean = df.copy()
        df_clean["importance"] = df_clean["coefficient"].abs()
        summary[model] = df_clean[["feature", "coefficient", "importance"]]
    else:
        df_clean = df.copy()
        df_clean["importance"] = df_clean["importance"] / df_clean["importance"].sum()
        summary[model] = df_clean

out_excel = "outputs_covid_shift/pooled_summary.xlsx"
with pd.ExcelWriter(out_excel) as writer:
    for model, df in summary.items():
        df.to_excel(writer, sheet_name=model, index=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for ax, (model, df) in zip(axes, summary.items()):
    if model == "Logit":
        top = df.sort_values("importance", ascending=False).head(15)
        colors = top["coefficient"].apply(lambda x: "green" if x > 0 else "red")
        ax.barh(top["feature"], top["coefficient"], color=colors)
        ax.set_title(f"{model} Interactions")
        ax.set_xlabel("Coefficient")
    else:
        top = df.sort_values("importance", ascending=False).head(15)
        ax.barh(top["feature"], top["importance"], color="skyblue")
        ax.set_title(f"{model} Feature Importance")
        ax.set_xlabel("Normalized Importance")

    ax.invert_yaxis()

plt.tight_layout()
plt.savefig("outputs_covid_shift/pooled_summary.png")
plt.show()