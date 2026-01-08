import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("y_2020_en.csv", low_memory=False)

df['C_RSUR'] = pd.to_numeric(df['C_RSUR'], errors='coerce')

surface_map = {
    1: "Dry, normal",
    2: "Wet",
    3: "Snow (fresh/loose)",
    4: "Slush (wet snow)",
    5: "Icy (incl. packed snow)",
    6: "Sand/Gravel/Dirt",
    7: "Muddy",
    8: "Oil/Spilled liquid",
    9: "Flooded",
}
df['C_RSUR_mapped'] = df['C_RSUR'].map(surface_map).fillna("Other/Unknown")

df['Fatal'] = (df['P_ISEV'].astype(str) == "3").astype(int)

df_valid = df[df['C_RSUR_mapped'] != "Other/Unknown"]

stats = df_valid.groupby('C_RSUR_mapped').agg(
    total=("P_ISEV", "count"),
    fatal=("Fatal", "sum")
)
stats["fatality_rate"] = (stats["fatal"] / stats["total"]) * 100
stats = stats.sort_values("fatality_rate", ascending=False)

# Plot
ax = stats["fatality_rate"].plot(
    kind="bar",
    color="lightcoral",
    edgecolor="black",
    figsize=(10,6)
)
plt.title("Fatality Rate by Road Surface (2020)")
plt.ylabel("Fatality Rate (%)")
plt.xlabel("Road Surface")
plt.xticks(rotation=45, ha="right")

for i, (idx, row) in enumerate(stats.iterrows()):
    ax.text(i, row["fatality_rate"] + 0.2, f"n={row['total']}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.show()