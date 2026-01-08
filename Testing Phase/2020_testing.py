# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from xgboost import XGBClassifier

# 2. Load dataset
df = pd.read_csv("y_2020_en.csv", low_memory=False)
print("Shape:", df.shape)
print(df.head())

# 3. Clean numeric-like columns
num_like_cols = ['C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS']
for col in num_like_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nConverted numeric-like columns. Null counts:")
print(df[num_like_cols].isna().sum())

# 4. Mapping dictionaries
weekday_map = {1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat", 7:"Sun"}
weather_map = {
    1:"Clear", 2:"Cloudy", 3:"Rain", 4:"Snow", 5:"Freezing Rain",
    6:"Low visibility", 7:"Strong wind"
}
surface_map = {
    1:"Dry", 2:"Wet", 3:"Snow", 4:"Slush", 5:"Ice",
    6:"Gravel/Dirt", 7:"Muddy", 8:"Oil", 9:"Flooded"
}
alignment_map = {
    1:"Straight/Level", 2:"Straight/Gradient",
    3:"Curve/Level", 4:"Curve/Gradient",
    5:"Top of Hill", 6:"Bottom of Hill"
}

# 5. Severity distribution
severity_counts = df['C_SEV'].value_counts().sort_index()
severity_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Collision Severity Distribution (2020)")
plt.xlabel("C_SEV (1=fatal, 2=injury, etc.)")
plt.ylabel("Count")
plt.show()

fatal_ratio = severity_counts.loc[1] / severity_counts.sum()
print(f"Fatal collisions %: {fatal_ratio:.4f}")

# 6. Exploratory Data Analysis (EDA)

# Collisions by month
df['C_MNTH'].dropna().astype(int).value_counts().sort_index().plot(
    kind="bar", color="orange"
)
plt.title("Collisions per Month (2020)")
plt.xlabel("Month")
plt.ylabel("Count")
plt.show()

# Collisions by hour
df['C_HOUR'].dropna().astype(int).value_counts().sort_index().plot(
    kind="bar", color="green"
)
plt.title("Collisions per Hour (2020)")
plt.xlabel("Hour of Day")
plt.ylabel("Count")
plt.show()

# Collisions by day of week
df['C_WDAY_mapped'] = df['C_WDAY'].map(weekday_map).fillna("Other/Unknown")
df['C_WDAY_mapped'].value_counts().reindex(
    ["Mon","Tue","Wed","Thu","Fri","Sat","Sun","Other/Unknown"]
).plot(kind="bar", color="purple")
plt.title("Collisions by Day of Week")
plt.xlabel("Day")
plt.ylabel("Count")
plt.show()

# --- FIXED PART: Fatality Rates instead of Counts ---

# Map categories
df['C_WTHR_mapped'] = df['C_WTHR'].map(weather_map).fillna("Other/Unknown")
df['C_RSUR_mapped'] = df['C_RSUR'].map(surface_map).fillna("Other/Unknown")
df['C_RALN_mapped'] = df['C_RALN'].map(alignment_map).fillna("Other/Unknown")

def plot_fatality_rate(df, col, title):
    # Drop "Other/Unknown"
    df_valid = df[df[col] != "Other/Unknown"].copy()

    if df_valid.empty:
        print(f"⚠ No valid data to plot for {col}")
        return

    # Compute fatality rate (%)
    stats = df_valid.groupby(col).agg(
        total=("C_SEV", "count"),
        fatal=("C_SEV", lambda x: (x == 1).sum())
    )
    stats["fatality_rate"] = (stats["fatal"] / stats["total"]) * 100

    # Sort
    stats = stats.sort_values("fatality_rate", ascending=False)

    # Plot
    ax = stats["fatality_rate"].plot(
        kind="bar", color="skyblue", edgecolor="black", figsize=(8, 5)
    )
    plt.title(title)
    plt.ylabel("Fatality Rate (%)")
    plt.xlabel(col.replace("_mapped", "").replace("C_", ""))
    plt.xticks(rotation=45, ha="right")

    # Add counts above bars
    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.text(i, row["fatality_rate"] + 0.1, f"n={row['total']}", 
                ha="center", va="bottom", fontsize=8)

    plt.show()

# Weather
plot_fatality_rate(df, "C_WTHR_mapped", "Fatality Rate by Weather Condition")

# Road Surface
plot_fatality_rate(df, "C_RSUR_mapped", "Fatality Rate by Road Surface")

# Road Alignment
plot_fatality_rate(df, "C_RALN_mapped", "Fatality Rate by Road Alignment")

# 7. Preprocessing
drop_cols = ['V_ID', 'P_ID', 'C_CASE']  # Drop IDs
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode categorical variables
cat_cols = df.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("Encoded shape:", df_encoded.shape)

# 8. Prepare features/target (drop NaNs)
y = (df['C_SEV'] == 1).astype(int) 
X = df_encoded.drop(columns=['C_SEV'])

# Drop rows with NaNs
mask = ~X.isna().any(axis=1)
X = X.loc[mask]
y = y.loc[mask]

print("After dropping NaNs:", X.shape, y.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 9. Baseline models
# Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)

print("\n--- Logistic Regression ---")
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1]))

# XGBoost
xgb = XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n--- XGBoost ---")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]))

# 10. Feature importance (XGBoost)
importances = pd.Series(xgb.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(15)

top_features.plot(kind="barh")
plt.title("Top 15 Features (XGBoost Importance)")
plt.show()

# 11. ROC & Precision-Recall Curves
def plot_curves(model, X_test, y_test, name):
    y_scores = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_scores):.3f})")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

plot_curves(logreg, X_test, y_test, "Logistic Regression")
plot_curves(xgb, X_test, y_test, "XGBoost")