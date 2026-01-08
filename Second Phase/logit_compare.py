import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df19 = pd.read_csv("outputs_covid_shift/prepared_2019.csv")
df20 = pd.read_csv("outputs_covid_shift/prepared_2020.csv")

df19 = df19.fillna(0)
df20 = df20.fillna(0)

X19, y19 = df19.drop(columns=["is_fatal"]), df19["is_fatal"]
X20, y20 = df20.drop(columns=["is_fatal"]), df20["is_fatal"]

def run_logit(X, y):
    model = LogisticRegression(
        max_iter=5000, 
        class_weight="balanced", 
        solver="lbfgs"
    )
    model.fit(X, y)
    coefs = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0]
    })
    return coefs, model

coef19, model19 = run_logit(X19, y19)
coef20, model20 = run_logit(X20, y20)

coef_compare = coef19.merge(coef20, on="feature", suffixes=("_2019", "_2020"))
coef_compare["difference"] = coef_compare["coefficient_2020"] - coef_compare["coefficient_2019"]

coef_compare.to_csv("outputs_covid_shift/coef_comparison_2019_vs_2020.csv", index=False)

top_shifts = coef_compare.reindex(
    coef_compare["difference"].abs().sort_values(ascending=False).index
).head(20)

plt.figure(figsize=(10, 6))
plt.barh(
    top_shifts["feature"], 
    top_shifts["difference"], 
    color=np.where(top_shifts["difference"] > 0, "green", "red")
)
plt.xlabel("Coefficient Shift (2020 - 2019)")
plt.title("Top Logistic Regression Coefficient Shifts (2020 vs 2019)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs_covid_shift/coef_shifts_2019_vs_2020.png", dpi=300)
plt.show()

print("\nTop 5 Positive Shifts (stronger in 2020):")
print(coef_compare.sort_values("difference", ascending=False).head(5))

print("\nTop 5 Negative Shifts (weaker in 2020):")
print(coef_compare.sort_values("difference", ascending=True).head(5))