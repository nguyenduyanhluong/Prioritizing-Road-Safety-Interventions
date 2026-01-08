import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("y_2020_en.csv", low_memory=False)

df['C_WTHR'] = pd.to_numeric(df['C_WTHR'], errors='coerce')

weather_map = {
    1: "Clear and Sunny",
    2: "Overcast/Cloudy (no precipitation)",
    3: "Raining",
    4: "Snowing",
    5: "Freezing Rain / Sleet / Hail",
    6: "Visibility Limited (fog, smog, dust, smoke, mist, drifting snow)",
    7: "Strong Wind",
}
df['C_WTHR_mapped'] = df['C_WTHR'].map(weather_map).fillna("Other/Unknown")

df['Fatal'] = (df['P_ISEV'].astype(str) == "3").astype(int)

df_valid = df[df['C_WTHR_mapped'] != "Other/Unknown"]

stats = df_valid.groupby('C_WTHR_mapped').agg(
    total=("P_ISEV", "count"),
    fatal=("Fatal", "sum")
)
stats["fatality_rate"] = (stats["fatal"] / stats["total"]) * 100
stats = stats.sort_values("fatality_rate", ascending=False)

# Plot
ax = stats["fatality_rate"].plot(
    kind="bar",
    color="skyblue",
    edgecolor="black",
    figsize=(10,6)
)
plt.title("Fatality Rate by Weather Condition (2020)")
plt.ylabel("Fatality Rate (%)")
plt.xlabel("Weather Condition")
plt.xticks(rotation=45, ha="right")

for i, (idx, row) in enumerate(stats.iterrows()):
    ax.text(i, row["fatality_rate"] + 0.2, f"n={row['total']}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.show()