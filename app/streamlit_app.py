import streamlit as st
import pandas as pd
import pydeck as pdk

from src.load_data import load_boston_burglary
from src.grid import add_xy_meters, make_grid
from src.risk_model import compute_risk

st.title("Boston Burglary Near-Repeat Crime Detector")

df = load_boston_burglary("data/raw/crime.csv", "data/raw/offense_codes.csv")
df_xy = add_xy_meters(df)

cell_size = 250
grid, spec = make_grid(df_xy, cell_size)

t0 = pd.Timestamp("2018-09-05")  # you can replace with a sidebar date input later
risk_map = compute_risk(df_xy, grid, t0, sigma_m=450, tau_days=7, lookback_days=21, radius_m=1500)

st.dataframe(risk_map.nlargest(20, "risk")[["lat", "lon", "risk"]])

layer = pdk.Layer(
    "ScatterplotLayer",
    data=risk_map[risk_map["risk"] > 0],
    get_position=["lon", "lat"],
    get_radius="risk * 60",
    pickable=True,
)

view = pdk.ViewState(latitude=float(df["lat"].median()), longitude=float(df["lon"].median()), zoom=11)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "risk: {risk}"}))