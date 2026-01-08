import pandas as pd
from pathlib import Path

BASE_PATH = Path(r"C:\Users\Andy's PC\OneDrive\Desktop\TMU - Research Assistant\Database Collision\Codes\29-10")
YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

def prepare_data(year):
    file_path = BASE_PATH / f"{year}.csv"
    df = pd.read_csv(file_path, low_memory=False)

    if "C_SEV" not in df.columns:
        raise ValueError(f"Column 'C_SEV' not found in {year}.csv")

    df = df[pd.to_numeric(df["C_SEV"], errors="coerce").notna()].copy()
    df["C_SEV"] = df["C_SEV"].astype(int)
    df = df[df["C_SEV"].between(0, 10)]

    if year <= 2017:
        df["is_fatal"] = (df["C_SEV"] == 0).astype(int)
    else:
        df["is_fatal"] = (df["C_SEV"] == 1).astype(int)

    total = len(df)
    fatal = int(df["is_fatal"].sum())
    nonfatal = total - fatal
    rate = fatal / total if total else 0

    print(f"Year {year}")
    print(f"Total cases: {total}")
    print(f"Fatal: {fatal} | Non-fatal: {nonfatal} | Fatality rate: {rate:.2%}")

    df.to_csv(BASE_PATH / f"{year}_prepared.csv", index=False)

for yr in YEARS:
    prepare_data(yr)

print("Step 1 complete — definitions corrected per year.")