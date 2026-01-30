import pandas as pd

def _read_csv_robust(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [",", "\t"]
    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] <= 1:
                    continue
                df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
                return df
            except Exception as e:
                last_err = e

    raise RuntimeError(f"Could not read {path}. Last error: {last_err}")

def _find_col(df: pd.DataFrame, wanted: str) -> str:
    w = wanted.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == w:
            return c
    for c in df.columns:
        if w in str(c).strip().lower():
            return c
    raise KeyError(f"Could not find column like '{wanted}'. Available: {df.columns.tolist()}")

def load_boston_burglary(crime_path: str, offense_codes_path: str | None = None) -> pd.DataFrame:
    crime = _read_csv_robust(crime_path)

    date_col = _find_col(crime, "OCCURRED_ON_DATE")
    lat_col = _find_col(crime, "Lat")
    lon_col = _find_col(crime, "Long")

    crime["dt"] = pd.to_datetime(crime[date_col], errors="coerce")

    # convert coords to numeric + drop bad
    crime[lat_col] = pd.to_numeric(crime[lat_col], errors="coerce")
    crime[lon_col] = pd.to_numeric(crime[lon_col], errors="coerce")
    crime = crime.dropna(subset=["dt", lat_col, lon_col]).copy()

    crime = crime.rename(columns={lat_col: "lat", lon_col: "lon"})

    crime = crime[
        crime["lat"].between(42.0, 42.6) &
        crime["lon"].between(-71.3, -70.8)
    ].copy()

    # optional join
    if offense_codes_path:
        codes = _read_csv_robust(offense_codes_path)
        code_col = _find_col(codes, "CODE")
        name_col = _find_col(codes, "NAME")
        codes = codes.rename(columns={code_col: "OFFENSE_CODE", name_col: "OFFENSE_NAME"})
        if "OFFENSE_CODE" in crime.columns:
            crime = crime.merge(codes[["OFFENSE_CODE", "OFFENSE_NAME"]], on="OFFENSE_CODE", how="left")

    # burglary filter
    group_col = "OFFENSE_CODE_GROUP" if "OFFENSE_CODE_GROUP" in crime.columns else None
    desc_col = "OFFENSE_DESCRIPTION" if "OFFENSE_DESCRIPTION" in crime.columns else None

    if group_col is None and desc_col is None:
        raise KeyError("Need OFFENSE_CODE_GROUP or OFFENSE_DESCRIPTION to filter burglary.")

    group = crime[group_col].astype(str) if group_col else ""
    desc = crime[desc_col].astype(str) if desc_col else ""
    mask = (
        (group.str.contains("burglary", case=False, na=False) if group_col else False) |
        (desc.str.contains("burglary", case=False, na=False) if desc_col else False)
    )
    crime = crime[mask].copy()

    keep = ["dt", "lat", "lon"]
    for c in ["INCIDENT_NUMBER", "OFFENSE_CODE", group_col, desc_col, "OFFENSE_NAME"]:
        if c and c in crime.columns and c not in keep:
            keep.append(c)

    return crime[keep].copy()