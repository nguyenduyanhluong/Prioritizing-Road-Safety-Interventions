import pandas as pd
import numpy as np
import inspect
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

OUTDIR = Path("outputs_covid_shift")
OUTDIR.mkdir(exist_ok=True)

DATA_2019 = "y_2019_en.csv"
DATA_2020 = "y_2020_en.csv"

def load_and_prepare(path, year2020_flag):
    df = pd.read_csv(path, low_memory=False)

    df = df[df["C_SEV"].isin([1, 2])].copy()
    df["is_fatal"] = (df["C_SEV"] == 1).astype(int)
    df["Year2020"] = int(year2020_flag)

    df = df.replace(["**", "NA", "NaN", "Unknown", "U", "X", "UU", "XX"], np.nan)

    return df

df19 = load_and_prepare(DATA_2019, 0)
df20 = load_and_prepare(DATA_2020, 1)

LEAK = ["is_fatal", "C_SEV", "Year2020"]
X19 = df19.drop(columns=LEAK, errors="ignore")
X20 = df20.drop(columns=LEAK, errors="ignore")

cat_cols, num_cols = [], []

for col in sorted(set(X19.columns) | set(X20.columns)):
    series = pd.concat([X19[col], X20[col]], axis=0)
    try:
        pd.to_numeric(series.dropna(), errors="raise")
        num_cols.append(col)
    except:
        cat_cols.append(col)

print("Detected numeric cols:", num_cols)
print("Detected categorical cols:", cat_cols)

if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", ohe, cat_cols),
], remainder="drop")

preproc.fit(pd.concat([X19, X20], axis=0, ignore_index=True))

def transform(df):
    X = df.drop(columns=LEAK, errors="ignore")
    Xt = preproc.transform(X)
    feat_names = list(num_cols) + list(preproc.named_transformers_["cat"].get_feature_names_out(cat_cols))
    return pd.DataFrame(Xt, columns=feat_names), df["is_fatal"].values

df19_out, y19 = transform(df19)
df20_out, y20 = transform(df20)

df19_out["is_fatal"] = y19
df20_out["is_fatal"] = y20

df19_out.to_csv(OUTDIR / "prepared_2019.csv", index=False)
df20_out.to_csv(OUTDIR / "prepared_2020.csv", index=False)

print("Data prepared & saved to:")
print("outputs_covid_shift/prepared_2019.csv")
print("outputs_covid_shift/prepared_2020.csv")