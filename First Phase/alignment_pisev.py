import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("y_2020_en.csv", low_memory=False)

df = df[["C_RALN", "P_ISEV"]].dropna()

severity_map = {
    "1": "No Injury",
    "2": "Injury",
    "3": "Fatality",
    "N": "Not applicable",
    "U": "Unknown",
    "X": "Not reported"
}
df["Severity"] = df["P_ISEV"].astype(str).map(severity_map)

alignment_map = {
    "1": "Straight & Level",
    "2": "Straight with Gradient",
    "3": "Curved & Level",
    "4": "Curved with Gradient",
    "5": "Top of Hill/Gradient",
    "6": "Bottom of Hill/Gradient (Sag)",
    "Q": "Other",
    "U": "Unknown",
    "X": "Not reported"
}
df["Road_Alignment"] = df["C_RALN"].astype(str).map(alignment_map)

df = df[df["Severity"].isin(["No Injury", "Injury", "Fatality"])]

counts = df.groupby(["Road_Alignment", "Severity"]).size().unstack(fill_value=0)
percentages = counts.div(counts.sum(axis=1), axis=0) * 100

# Plot
ax = percentages[["No Injury", "Injury", "Fatality"]].plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6),
    color=["lightgreen", "orange", "red"]
)

for i, aln in enumerate(percentages.index):
    total = counts.loc[aln].sum()
    ax.text(i, 102, f"n={total}", ha="center", fontsize=8, rotation=90)

plt.title("Collision Severity Distribution by Road Alignment (2020)")
plt.xlabel("Road Alignment")
plt.ylabel("Percentage of Collisions (%)")
plt.legend(title="Severity")
plt.ylim(0, 110)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()