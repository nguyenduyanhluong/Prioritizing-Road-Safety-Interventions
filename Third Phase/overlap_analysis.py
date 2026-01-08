import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import os

RESULTS_DIR = "outputs_covid_shift"
TOP_N = 20

FILES = {
    "2019": {
        "logit": os.path.join(RESULTS_DIR, "logit_results_2019.csv"),
        "rf": os.path.join(RESULTS_DIR, "rf_results_2019.csv"),
        "xgb": os.path.join(RESULTS_DIR, "xgb_results_2019.csv"),
    },
    "2020": {
        "logit": os.path.join(RESULTS_DIR, "logit_results_2020.csv"),
        "rf": os.path.join(RESULTS_DIR, "rf_results_2020.csv"),
        "xgb": os.path.join(RESULTS_DIR, "xgb_results_2020.csv"),
    },
    "combined": {
        "logit": os.path.join(RESULTS_DIR, "logit_results_combined.csv"),
        "rf": os.path.join(RESULTS_DIR, "rf_results_combined.csv"),
        "xgb": os.path.join(RESULTS_DIR, "xgb_results_combined.csv"),
    },
}

def load_top_features(file_path, top_n=TOP_N):
    df = pd.read_csv(file_path)

    if "feature" in df.columns:
        feature_col = "feature"
    elif "variable" in df.columns:
        feature_col = "variable"
    elif "Unnamed: 0" in df.columns: 
        feature_col = "Unnamed: 0"
    else:
        raise ValueError(f"No feature column found in {file_path}. Columns: {df.columns.tolist()}")

    if "importance" in df.columns:
        df = df.sort_values("importance", ascending=False)
    elif "coef" in df.columns:
        df["abs_coef"] = df["coef"].abs()
        df = df.sort_values("abs_coef", ascending=False)
    else:
        raise ValueError(f"No importance/coef column in {file_path}. Columns: {df.columns.tolist()}")

    return df[feature_col].head(top_n).tolist()

all_results = []

for year, models in FILES.items():
    top_features = {m: load_top_features(p, TOP_N) for m, p in models.items()}

    models_list = list(top_features.keys())
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            m1, m2 = models_list[i], models_list[j]
            set1, set2 = set(top_features[m1]), set(top_features[m2])
            overlap = len(set1 & set2)
            union = len(set1 | set2)
            all_results.append({
                "year": year,
                "model1": m1,
                "model2": m2,
                "overlap_count": overlap,
                "union_count": union,
                "jaccard": overlap / union if union > 0 else 0
            })

    plt.figure(figsize=(8,6))
    venn3(
        [set(top_features["logit"]), set(top_features["rf"]), set(top_features["xgb"])],
        set_labels=("Logit", "Random Forest", "XGBoost")
    )
    plt.title(f"Top-{TOP_N} Feature Overlap ({year})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"feature_overlap_venn_{year}.png"))
    plt.close()

    with open(os.path.join(RESULTS_DIR, f"top_features_{year}.txt"), "w") as f:
        for model, feats in top_features.items():
            f.write(f"\n=== {model.upper()} ({year}) Top-{TOP_N} ===\n")
            for rank, feat in enumerate(feats, 1):
                f.write(f"{rank}. {feat}\n")

overlap_df = pd.DataFrame(all_results)
overlap_df.to_csv(os.path.join(RESULTS_DIR, "feature_overlap_summary.csv"), index=False)

print("Saved all overlap results:")
print(f"- {RESULTS_DIR}/feature_overlap_summary.csv")
print(f"- {RESULTS_DIR}/feature_overlap_venn_YEAR.png (per year)")
print(f"- {RESULTS_DIR}/top_features_YEAR.txt (per year)")