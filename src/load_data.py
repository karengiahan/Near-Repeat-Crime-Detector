import pandas as pd

def read_boston_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="cp1252")

def load_boston_burglary(crime_path: str, offense_codes_path: str | None = None) -> pd.DataFrame:
    crime = read_boston_csv(crime_path)

    # Parse time + coords
    crime["dt"] = pd.to_datetime(crime["OCCURRED_ON_DATE"], errors="coerce")
    crime = crime.dropna(subset=["dt", "Lat", "Long"]).copy()
    crime = crime.rename(columns={"Lat": "lat", "Long": "lon"})

    # Optional: join offense_codes to get nicer offense names (not required for MVP)
    if offense_codes_path:
        codes = read_boston_csv(offense_codes_path)
        # your offense_codes sample shows columns: CODE, NAME
        codes = codes.rename(columns={"CODE": "OFFENSE_CODE", "NAME": "OFFENSE_NAME"})
        crime = crime.merge(codes[["OFFENSE_CODE", "OFFENSE_NAME"]], on="OFFENSE_CODE", how="left")

    # Filter: burglary
    # (sometimes burglary appears in group or description)
    group = crime["OFFENSE_CODE_GROUP"].astype(str)
    desc = crime["OFFENSE_DESCRIPTION"].astype(str)
    mask = group.str.contains("burglary", case=False, na=False) | desc.str.contains("burglary", case=False, na=False)
    crime = crime[mask].copy()

    # Keep just what we need
    keep = ["dt", "lat", "lon", "OFFENSE_CODE", "OFFENSE_CODE_GROUP", "OFFENSE_DESCRIPTION"]
    if "OFFENSE_NAME" in crime.columns:
        keep.append("OFFENSE_NAME")
    if "INCIDENT_NUMBER" in crime.columns:
        keep.append("INCIDENT_NUMBER")

    return crime[keep].copy()