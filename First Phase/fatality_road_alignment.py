import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("y_2020_en.csv", low_memory=False)

df['C_RALN'] = pd.to_numeric(df['C_RALN'], errors='coerce')

alignment_map = {
    1: "Straight & Level",
    2: "Straight & Gradient",
    3: "Curved & Level",
    4: "Curved & Gradient",
    5: "Top of Hill/Gradient",
    6: "Bottom of Hill/Gradient (Sag)"
}
df['C_RALN_mapped'] = df['C_RALN'].map(alignment_map).fillna("Other/Unknown")

df['Fatal'] = (df['P_ISEV'].astype(str) == "3").astype(int)

df_valid = df[df['C_RALN_mapped'] != "Other/Unknown"]

stats = df_valid.groupby('C_RALN_mapped').agg(
    total=("P_ISEV", "count"),
    fatal=("Fatal", "sum")
)
stats["fatality_rate"] = (stats["fatal"] / stats["total"]) * 100
stats = stats.sort_values("fatality_rate", ascending=False)

# Plot
ax = stats["fatality_rate"].plot(
    kind="bar",
    color="seagreen",
    edgecolor="black",
    figsize=(10,6)
)
plt.title("Fatality Rate by Road Alignment (2020)")
plt.ylabel("Fatality Rate (%)")
plt.xlabel("Road Alignment")
plt.xticks(rotation=45, ha="right")

for i, (idx, row) in enumerate(stats.iterrows()):
    ax.text(i, row["fatality_rate"] + 0.2, f"n={row['total']}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.show()