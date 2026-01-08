import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

df19 = pd.read_csv("outputs_covid_shift/prepared_2019.csv")
df20 = pd.read_csv("outputs_covid_shift/prepared_2020.csv")

df19 = df19.fillna(0)
df20 = df20.fillna(0)

X19, y19 = df19.drop(columns=["is_fatal"]), df19["is_fatal"]
X20, y20 = df20.drop(columns=["is_fatal"]), df20["is_fatal"]

scaler = StandardScaler()
X19 = pd.DataFrame(scaler.fit_transform(X19), columns=X19.columns)
X20 = pd.DataFrame(scaler.fit_transform(X20), columns=X20.columns)

def run_nn(X, y, epochs=20, batch_size=256, lr=0.001):
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(32, activation="relu", name="first_hidden"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    first_layer = model.get_layer("first_hidden")
    weights, biases = first_layer.get_weights()

    importance = np.sum(np.abs(weights), axis=1)

    assert len(importance) == X.shape[1], \
        f"Mismatch: {len(importance)} importances vs {X.shape[1]} features"

    return pd.DataFrame({
        "feature": list(X.columns),
        "importance": importance
    }), model

imp19, model19 = run_nn(X19, y19)
imp20, model20 = run_nn(X20, y20)

imp_compare = imp19.merge(imp20, on="feature", suffixes=("_2019", "_2020"))
imp_compare["difference"] = imp_compare["importance_2020"] - imp_compare["importance_2019"]

imp_compare.to_csv("outputs_covid_shift/nn_importance_comparison_2019_vs_2020.csv", index=False)

top_shifts = imp_compare.reindex(
    imp_compare["difference"].abs().sort_values(ascending=False).index
).head(20)

plt.figure(figsize=(10, 6))
plt.barh(
    top_shifts["feature"],
    top_shifts["difference"],
    color=np.where(top_shifts["difference"] > 0, "green", "red")
)
plt.xlabel("NN Feature Importance Shift (2020 - 2019)")
plt.title("Top Neural Network Feature Importance Shifts (2020 vs 2019)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs_covid_shift/nn_shifts_2019_vs_2020.png", dpi=300)
plt.show()

print("\nTop 5 Positive Shifts (more important in 2020):")
print(imp_compare.sort_values("difference", ascending=False).head(5))

print("\nTop 5 Negative Shifts (less important in 2020):")
print(imp_compare.sort_values("difference", ascending=True).head(5))