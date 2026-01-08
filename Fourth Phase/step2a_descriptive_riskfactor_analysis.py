import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
OUTDIR = BASE_PATH / "outputs_step2_descriptives"
OUTDIR.mkdir(exist_ok=True)
YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
TARGET_COL = "is_fatal"
DROP_COLS = {"C_SEV", TARGET_COL}
CATEGORICAL_NUNIQUE_MAX = 200
SMALL_N_THRESHOLD = 20

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    margin = (z/denom) * np.sqrt((p*(1-p)/n) + (z**2)/(4*(n**2)))
    return (max(0.0, center - margin), min(1.0, center + margin))

def detect_columns(df):
    feats = [c for c in df.columns if c not in DROP_COLS]
    cat_cols, num_cols = [], []
    for c in feats:
        s = df[c]
        if s.dtype == "O":
            cat_cols.append(c)
        else:
            u = s.dropna().nunique()
            if u <= CATEGORICAL_NUNIQUE_MAX:
                cat_cols.append(c)
            else:
                num_cols.append(c)
    return cat_cols, num_cols

def level_table(df, col):
    overall_n = len(df)
    overall_k = int(df[TARGET_COL].sum())
    overall_rate = overall_k / overall_n if overall_n else np.nan
    g = df.groupby(col, dropna=False)[TARGET_COL].agg(["count", "sum"])
    g = g.rename(columns={"count": "n", "sum": "fatal_n"}).reset_index()
    g["fatal_rate"] = g["fatal_n"] / g["n"]
    ci = g.apply(lambda r: pd.Series(wilson_ci(int(r["fatal_n"]), int(r["n"])), index=["fatal_rate_L", "fatal_rate_U"]), axis=1)
    g = pd.concat([g, ci], axis=1)
    ref_idx = g.sort_values(["fatal_rate", "n"], ascending=[True, False]).index[0]
    ref_rate = float(g.loc[ref_idx, "fatal_rate"])
    g["RR_vs_overall"] = g["fatal_rate"] / overall_rate if overall_rate and overall_rate > 0 else np.nan
    g["RD_vs_overall"] = g["fatal_rate"] - overall_rate if pd.notna(overall_rate) else np.nan
    g["RR_vs_ref"] = g["fatal_rate"] / ref_rate if ref_rate and ref_rate > 0 else np.nan
    g["RD_vs_ref"] = g["fatal_rate"] - ref_rate
    g["small_n_flag"] = g["n"] < SMALL_N_THRESHOLD
    g.insert(0, "feature", col)
    lvl_name = f"{col}_level"
    g = g.rename(columns={col: lvl_name})
    g[lvl_name] = g[lvl_name].astype("string")
    g = g.sort_values(["feature", "fatal_rate", "n"], ascending=[True, False, False])
    return g, overall_rate

def factor_summary(level_df, feature):
    df = level_df[level_df["feature"] == feature].copy()
    n_levels = df.shape[0]
    max_row = df.sort_values(["fatal_rate", "n"], ascending=[False, False]).iloc[0]
    min_row = df.sort_values(["fatal_rate", "n"], ascending=[True, False]).iloc[0]
    rng = float(max_row["fatal_rate"] - min_row["fatal_rate"])
    w = df["n"].values
    p = df["fatal_rate"].values
    if w.sum() > 0:
        mean_p = np.average(p, weights=w)
        var_w = np.average((p - mean_p)**2, weights=w)
    else:
        var_w = np.nan
    out = {
        "feature": feature,
        "levels": n_levels,
        "max_rate": float(max_row["fatal_rate"]),
        "max_rate_level": max_row[f"{feature}_level"],
        "max_rate_n": int(max_row["n"]),
        "min_rate": float(min_row["fatal_rate"]),
        "min_rate_level": min_row[f"{feature}_level"],
        "min_rate_n": int(min_row["n"]),
        "range_rate": rng,
        "weighted_var_rate": float(var_w),
        "pct_small_n_levels": float((df["small_n_flag"].mean() if n_levels > 0 else np.nan)),
    }
    return out

def run_year(year):
    in_path = BASE_PATH / f"{year}_prepared.csv"
    df = pd.read_csv(in_path, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} missing in {in_path}")
    cat_cols, num_cols = detect_columns(df)
    level_frames = []
    for col in cat_cols:
        tmp = df[[col, TARGET_COL]].copy()
        tmp[col] = tmp[col].astype("string")
        lev_tab, overall_rate = level_table(tmp, col)
        level_frames.append(lev_tab)
    if level_frames:
        levels_all = pd.concat(level_frames, axis=0, ignore_index=True)
    else:
        levels_all = pd.DataFrame()
    factors = []
    for col in cat_cols:
        factors.append(factor_summary(levels_all, col))
    factor_df = pd.DataFrame(factors).sort_values(["range_rate", "weighted_var_rate", "max_rate"], ascending=[False, False, False])
    out_levels = OUTDIR / f"level_risk_summary_{year}.csv"
    out_factors = OUTDIR / f"factor_summary_{year}.csv"
    levels_all.to_csv(out_levels, index=False)
    factor_df.to_csv(out_factors, index=False)
    print(f"{year}: saved {out_levels.name}, {out_factors.name}")
    print(f"  features analyzed: {len(cat_cols)} | numeric skipped: {len(num_cols)}")

for y in YEARS:
    run_year(y)

all_factor = []
for y in YEARS:
    p = OUTDIR / f"factor_summary_{y}.csv"
    if p.exists():
        df = pd.read_csv(p)
        df.insert(0, "year", y)
        all_factor.append(df)
if all_factor:
    all_factor_df = pd.concat(all_factor, ignore_index=True)
    all_factor_df.to_csv(OUTDIR / "factor_summary_all_years.csv", index=False)
    print("Saved factor_summary_all_years.csv")