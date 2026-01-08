import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Load data
df19 = pd.read_csv("outputs_covid_shift/prepared_2019.csv").fillna(0)
df20 = pd.read_csv("outputs_covid_shift/prepared_2020.csv").fillna(0)

# Add COVID dummy
df19["covid_dummy"] = 0
df20["covid_dummy"] = 1

# Combine datasets
df_all = pd.concat([df19, df20], axis=0, ignore_index=True)

# Split features and target
y = df_all["is_fatal"].astype(int).values
X = df_all.drop(columns=["is_fatal"])
feature_names = X.columns.tolist()
X = X.values.astype(np.float32)

# Build NN model
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train for only 5 epochs
model.fit(X, y, epochs=5, batch_size=128, verbose=1)

# SHAP explainer (using a subset of the data for speed)
explainer = shap.DeepExplainer(model, X[:500])
shap_values = explainer.shap_values(X[:1000])

# Handle case where shap returns a list
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Compute mean absolute SHAP values as importance
importance = np.mean(np.abs(shap_values), axis=0).flatten()

print(f"Features: {len(feature_names)}, SHAP values length: {len(importance)}")

# Save importance to CSV
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values("importance", ascending=False)

importance_df.to_csv("outputs_covid_shift/pooled_nn_importance.csv", index=False)

# Plot top 20
topN = 20
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["feature"].head(topN)[::-1],
    importance_df["importance"].head(topN)[::-1],
    color="skyblue"
)
plt.title("Top Features (Pooled Neural Net with COVID Dummy)")
plt.xlabel("SHAP Importance")
plt.tight_layout()
plt.savefig("outputs_covid_shift/pooled_nn_importance.png", dpi=150)
plt.show()

print("Results saved to outputs_covid_shift/pooled_nn_importance.csv and .png")