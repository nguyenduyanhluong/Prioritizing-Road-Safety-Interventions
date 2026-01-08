import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logit = pd.read_csv("outputs_covid_shift/coef_comparison_2019_vs_2020.csv")
xgb = pd.read_csv("outputs_covid_shift/xgb_importance_comparison_2019_vs_2020.csv")
rf = pd.read_csv("outputs_covid_shift/rf_importance_comparison_2019_vs_2020.csv")
nn = pd.read_csv("outputs_covid_shift/nn_importance_comparison_2019_vs_2020.csv")

def get_top_shifts(df, col_diff="difference", top_n=10):
    return df.reindex(df[col_diff].abs().sort_values(ascending=False).index).head(top_n)

logit_top = get_top_shifts(logit, "difference", 10)
xgb_top   = get_top_shifts(xgb, "difference", 10)
rf_top    = get_top_shifts(rf, "difference", 10)
nn_top    = get_top_shifts(nn, "difference", 10)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

axes[0].barh(logit_top["feature"], logit_top["difference"], 
             color=np.where(logit_top["difference"] > 0, "green", "red"))
axes[0].set_title("Logistic Regression Shifts (2020 vs 2019)")
axes[0].set_xlabel("Coefficient Shift")

axes[1].barh(xgb_top["feature"], xgb_top["difference"], 
             color=np.where(xgb_top["difference"] > 0, "green", "red"))
axes[1].set_title("XGBoost Shifts (2020 vs 2019)")
axes[1].set_xlabel("Feature Importance Shift")

axes[2].barh(rf_top["feature"], rf_top["difference"], 
             color=np.where(rf_top["difference"] > 0, "green", "red"))
axes[2].set_title("Random Forest Shifts (2020 vs 2019)")
axes[2].set_xlabel("Feature Importance Shift")

axes[3].barh(nn_top["feature"], nn_top["difference"], 
             color=np.where(nn_top["difference"] > 0, "green", "red"))
axes[3].set_title("Neural Net Shifts (2020 vs 2019)")
axes[3].set_xlabel("NN Feature Importance Shift")

for ax in axes:
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)

fig.suptitle("Top Feature Shifts Across Models (2019 vs 2020)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig("outputs_covid_shift/combined_model_shifts.png", dpi=300)
plt.show()

with pd.ExcelWriter("outputs_covid_shift/combined_top_shifts.xlsx") as writer:
    logit_top.to_excel(writer, sheet_name="Logit", index=False)
    xgb_top.to_excel(writer, sheet_name="XGBoost", index=False)
    rf_top.to_excel(writer, sheet_name="RandomForest", index=False)
    nn_top.to_excel(writer, sheet_name="NeuralNet", index=False)

print("Combined top shifts saved to outputs_covid_shift/combined_top_shifts.xlsx")
print("Multi-model comparison figure saved to outputs_covid_shift/combined_model_shifts.png")