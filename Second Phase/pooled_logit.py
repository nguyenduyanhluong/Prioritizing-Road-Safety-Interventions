import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

df19 = pd.read_csv("outputs_covid_shift/prepared_2019.csv")
df20 = pd.read_csv("outputs_covid_shift/prepared_2020.csv")

df19["covid_dummy"] = 0
df20["covid_dummy"] = 1
df = pd.concat([df19, df20], axis=0).reset_index(drop=True)

X = df.drop(columns=["is_fatal"])
y = df["is_fatal"]

N_FEATURES = 50
variances = X.var().sort_values(ascending=False)
top_features = variances.head(N_FEATURES).index.tolist()
if "covid_dummy" not in top_features:
    top_features.append("covid_dummy")
X = X[top_features]

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
).astype(np.float32)

interaction_terms = X_scaled.drop(columns=["covid_dummy"]).multiply(
    X_scaled["covid_dummy"], axis=0
)
interaction_terms = interaction_terms.add_suffix("_x_covid")

X_inter = pd.concat([X_scaled, interaction_terms], axis=1)
X_inter = X_inter.replace([np.inf, -np.inf], 0).fillna(0)
X_inter = sm.add_constant(X_inter)

logit_model = sm.Logit(y, X_inter)
result = logit_model.fit(disp=True, maxiter=200)

coef_table = pd.DataFrame({
    "feature": X_inter.columns,
    "coefficient": result.params
}).reset_index(drop=True)

coef_table.to_csv("outputs_covid_shift/pooled_logit_interactions.csv", index=False)

inter_only = coef_table[coef_table["feature"].str.contains("_x_covid")].copy()
top_inter = inter_only.reindex(
    inter_only["coefficient"].abs().sort_values(ascending=False).index
).head(20)

plt.figure(figsize=(10, 6))
plt.barh(
    top_inter["feature"],
    top_inter["coefficient"],
    color=np.where(top_inter["coefficient"] > 0, "green", "red")
)
plt.xlabel("Coefficient (Interaction with COVID Dummy)")
plt.title("Top COVID Interaction Effects (Pooled Logistic Regression)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs_covid_shift/pooled_logit_interactions.png", dpi=300)
plt.show()

print(f"Used top {N_FEATURES} features by variance.")
print("\nTop 5 Interaction Effects:")
print(top_inter.head(5))