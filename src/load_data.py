import pandas as pd

def _read_csv_robust(path: str) -> pd.DataFrame:
    """
    Boston files can be:
      - comma-delimited OR tab-delimited
      - encoded as utf-8 OR cp1252/latin1
    This tries combinations until it gets >1 column.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [",", "\t"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] <= 1:
                    continue  # wrong delimiter most likely
                # clean column names
                df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
                return df
            except Exception as e:
                last_err = e

    raise RuntimeError(f"Could not read {path}. Last error: {last_err}")

def _find_col(df: pd.DataFrame, wanted: str) -> str:
    """
    Find a column ignoring case and spaces.
    Example: wanted='OCCURRED_ON_DATE'
    """
    w = wanted.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == w:
            return c
    # fallback: partial match
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
    crime = crime.dropna(subset=["dt", lat_col, lon_col]).copy()

    crime = crime.rename(columns={lat_col: "lat", lon_col: "lon"})

    # Optional: join offense code names
    if offense_codes_path:
        codes = _read_csv_robust(offense_codes_path)
        code_col = _find_col(codes, "CODE")
        name_col = _find_col(codes, "NAME")
        codes = codes.rename(columns={code_col: "OFFENSE_CODE", name_col: "OFFENSE_NAME"})
        if "OFFENSE_CODE" in crime.columns:
            crime = crime.merge(codes[["OFFENSE_CODE", "OFFENSE_NAME"]], on="OFFENSE_CODE", how="left")

    # Burglary filter: use group/description if available
    group_col = None
    desc_col = None
    if any(str(c).strip().lower() == "offense_code_group" for c in crime.columns):
        group_col = _find_col(crime, "OFFENSE_CODE_GROUP")
    if any(str(c).strip().lower() == "offense_description" for c in crime.columns):
        desc_col = _find_col(crime, "OFFENSE_DESCRIPTION")

    if group_col is None and desc_col is None:
        raise KeyError("Need OFFENSE_CODE_GROUP or OFFENSE_DESCRIPTION in crime.csv to filter burglary.")

    group = crime[group_col].astype(str) if group_col else ""
    desc = crime[desc_col].astype(str) if desc_col else ""
    mask = (
        (group.str.contains("burglary", case=False, na=False) if group_col else False) |
        (desc.str.contains("burglary", case=False, na=False) if desc_col else False)
    )
    crime = crime[mask].copy()

    keep = ["dt", "lat", "lon"]
    for c in ["OFFENSE_CODE", group_col, desc_col, "OFFENSE_NAME", "INCIDENT_NUMBER"]:
        if c and c in crime.columns and c not in keep:
            keep.append(c)

    return crime[keep].copy()