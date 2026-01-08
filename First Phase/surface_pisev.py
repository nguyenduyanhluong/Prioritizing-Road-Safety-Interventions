import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("y_2020_en.csv", low_memory=False)

df = df[["C_RSUR", "P_ISEV"]].dropna()

severity_map = {
    "1": "No Injury",
    "2": "Injury",
    "3": "Fatality",
    "N": "Not applicable",
    "U": "Unknown",
    "X": "Not reported"
}
df["Severity"] = df["P_ISEV"].astype(str).map(severity_map)

surface_map = {
    "1": "Dry, normal",
    "2": "Wet",
    "3": "Snow (fresh/loose)",
    "4": "Slush (wet snow)",
    "5": "Icy (incl. packed snow)",
    "6": "Sand/Gravel/Dirt",
    "7": "Muddy",
    "8": "Oil/Spilled liquid",
    "9": "Flooded",
    "Q": "Other",
    "U": "Unknown",
    "X": "Not reported"
}
df["Road_Surface"] = df["C_RSUR"].astype(str).map(surface_map)

df = df[df["Severity"].isin(["No Injury", "Injury", "Fatality"])]

counts = df.groupby(["Road_Surface", "Severity"]).size().unstack(fill_value=0)
percentages = counts.div(counts.sum(axis=1), axis=0) * 100

# Plot
ax = percentages[["No Injury", "Injury", "Fatality"]].plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6),
    color=["lightgreen", "orange", "red"]
)

for i, surface in enumerate(percentages.index):
    total = counts.loc[surface].sum()
    ax.text(i, 102, f"n={total}", ha="center", fontsize=8, rotation=90)

plt.title("Collision Severity Distribution by Road Surface (2020)")
plt.xlabel("Road Surface")
plt.ylabel("Percentage of Collisions (%)")
plt.legend(title="Severity")
plt.ylim(0, 110)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()