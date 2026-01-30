import pandas as pd

def read_any(path: str):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = ["\t", ","]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] <= 1:
                    continue
                return df, enc, sep
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"Could not read {path}. Last error: {last_err}")

crime, enc1, sep1 = read_any("data/raw/crime.csv")
off, enc2, sep2 = read_any("data/raw/offense_codes.csv")

print(f"crime.csv -> encoding={enc1}, sep={repr(sep1)}, shape={crime.shape}")
print("crime columns:\n", crime.columns.tolist())

print(f"\noffense_codes.csv -> encoding={enc2}, sep={repr(sep2)}, shape={off.shape}")
print("offense_codes columns:\n", off.columns.tolist())

print("\nTop OFFENSE_CODE_GROUP values:")
print(crime["OFFENSE_CODE_GROUP"].astype(str).value_counts().head(10))