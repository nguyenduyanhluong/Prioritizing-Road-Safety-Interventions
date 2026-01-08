import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("y_2020_en.csv", low_memory=False)

weather_map = {
    "1": "Clear and sunny",
    "2": "Overcast/Cloudy (no precipitation)",
    "3": "Rain",
    "4": "Snow",
    "5": "Freezing rain / sleet / hail",
    "6": "Visibility limitation (fog, smog, dust, drifting snow, smoke, mist)",
    "7": "Strong wind",
    "Q": "Other",
    "U": "Unknown",
    "X": "Not reported"
}
df["Weather"] = df["C_WTHR"].astype(str).map(weather_map).fillna("Unknown")

pisev_map = {
    "1": "No Injury",
    "2": "Injury",
    "3": "Fatality",
    "N": "Not applicable",
    "U": "Unknown",
    "X": "Not reported"
}
df["Severity"] = df["P_ISEV"].astype(str).map(pisev_map)

df = df[df["Severity"].isin(["No Injury", "Injury", "Fatality"])]

counts = df.groupby(["Weather", "Severity"]).size().unstack(fill_value=0)

severity_order = ["No Injury", "Injury", "Fatality"]
counts = counts.reindex(columns=severity_order, fill_value=0)

rates = counts.div(counts.sum(axis=1), axis=0) * 100

# Plot
ax = rates.plot(
    kind="bar",
    figsize=(12, 6),
    color=["lightgreen", "orange", "red"],
    width=0.8
)

plt.title("Collision Severity Distribution by Weather Condition (2020)")
plt.ylabel("Percentage of Collisions (%)")
plt.xlabel("Weather Condition")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Severity")

for i, weather in enumerate(rates.index):
    total = counts.loc[weather].sum()
    ax.text(i, 102, f"n={total}", ha="center", fontsize=8, rotation=90)

plt.ylim(0, 110)
plt.tight_layout()
plt.show()