import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("y_2020_en.csv", low_memory=False)

df['C_SEV'] = df['C_SEV'].astype(str)
df['C_MNTH'] = pd.to_numeric(df['C_MNTH'], errors='coerce')
df['C_HOUR'] = pd.to_numeric(df['C_HOUR'], errors='coerce')
df['C_WDAY'] = pd.to_numeric(df['C_WDAY'], errors='coerce')

sev_map = {
    "1": "Fatal",
    "2": "Injury",
    "U": "Unknown",
    "X": "Not reported"
}
df['C_SEV_label'] = df['C_SEV'].map(sev_map).fillna("Other/Unknown")

severity_counts = df['C_SEV_label'].value_counts().reindex(
    ["Fatal","Injury","Unknown","Not reported","Other/Unknown"]
).dropna()

ax = severity_counts.plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("Collision Severity Distribution (2020)")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.legend(["Collisions"])

for i, (label, count) in enumerate(severity_counts.items()):
    ax.text(i, count + 500, f"n={count}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()

fatal_ratio = (df['C_SEV'] == "1").mean()
print(f"Fatal collisions %: {fatal_ratio:.4%}")

df['C_MNTH'].dropna().astype(int).value_counts().sort_index().plot(
    kind="bar", color="orange", edgecolor="black"
)
plt.title("Collisions per Month (2020)")
plt.xlabel("Month")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

df['C_HOUR'].dropna().astype(int).value_counts().sort_index().plot(
    kind="bar", color="green", edgecolor="black"
)
plt.title("Collisions per Hour (2020)")
plt.xlabel("Hour of Day")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

weekday_map = {1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat", 7:"Sun"}
df['C_WDAY_mapped'] = df['C_WDAY'].map(weekday_map).fillna("Other/Unknown")

df['C_WDAY_mapped'].value_counts().reindex(
    ["Mon","Tue","Wed","Thu","Fri","Sat","Sun","Other/Unknown"]
).plot(kind="bar", color="purple", edgecolor="black")

plt.title("Collisions by Day of Week (2020)")
plt.xlabel("Day")
plt.ylabel("Count")
plt.tight_layout()
plt.show()