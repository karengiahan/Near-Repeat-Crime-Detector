import streamlit as st
import pandas as pd
import pydeck as pdk

from sklearn.neighbors import KDTree
from src.load_data import load_boston_burglary
from src.grid import add_xy_meters, add_lonlat_from_xy
from src.risk_model import compute_risk
from src.evaluate import backtest_range

st.set_page_config(page_title="Boston Near-Repeat", layout="wide")
st.title("Boston Near-Repeat Crime Detector")

@st.cache_data(show_spinner=False)
def load_data():
    df = load_boston_burglary("data/raw/crime.csv", "data/raw/offense_codes.csv")
    return df

df = load_data()
df_xy = add_xy_meters(df)

#controls
default_date = (df["dt"].max() - pd.Timedelta(days=1)).date()

t0_date = st.sidebar.date_input("Prediction date", value=default_date)

lookback_choice = st.sidebar.selectbox("Recent window", ["Last 7 days", "Last 14 days", "Last 21 days"])
lookback = {"Last 7 days": 7, "Last 14 days": 14, "Last 21 days": 21}[lookback_choice]

spread_choice = st.sidebar.selectbox("Neighborhood size", ["Small (~300m)", "Medium (~600m)", "Large (~900m)"])
sigma = {"Small (~300m)": 300, "Medium (~600m)": 600, "Large (~900m)": 900}[spread_choice]

cell_size = 300
tau = 7
radius = 1500
topk = 60

tab1, tab2 = st.tabs(["Map", "Backtest"])

#MAP TAB
with tab1:
    t0 = pd.Timestamp(t0_date)
    start = t0 - pd.Timedelta(days=lookback)
    recent = df_xy[(df_xy["dt"] < t0) & (df_xy["dt"] >= start)].copy()

    if recent.empty:
        st.warning("No burglary incidents in that window. Try a different date.")
        st.stop()

    # Build a grid only around the recent bounding box (+ padding) for speed
    xmin, xmax = recent["x"].min() - radius, recent["x"].max() + radius
    ymin, ymax = recent["y"].min() - radius, recent["y"].max() + radius

    xs = pd.Series(range(int((xmax - xmin) // cell_size) + 1)) * cell_size + xmin
    ys = pd.Series(range(int((ymax - ymin) // cell_size) + 1)) * cell_size + ymin
    xc = xs[:-1] + cell_size / 2
    yc = ys[:-1] + cell_size / 2
    grid = pd.DataFrame([(x, y) for y in yc for x in xc], columns=["x", "y"])

    with st.spinner("Computing risk map…"):
        risk_map = compute_risk(df_xy, grid, t0, sigma_m=sigma, tau_days=tau, lookback_days=lookback, radius_m=radius)

    risk_map = add_lonlat_from_xy(risk_map)
    max_r = float(risk_map["risk"].max()) if len(risk_map) else 1.0
    risk_map["risk_norm"] = (risk_map["risk"] / max_r) if max_r > 0 else 0.0
    
    top = risk_map.nlargest(topk, "risk").copy()

    st.caption("Redder areas = higher relative near-repeat risk from recent burglaries (distance- and time-decayed)")

    # Heatmap of risk + points for recent incidents
    heat = pdk.Layer(
        "HeatmapLayer",
        data=risk_map[risk_map["risk"] > 0],
        get_position=["lon", "lat"],
        get_weight="risk_norm",
        radiusPixels=65,
    )

    pts = pdk.Layer(
        "ScatterplotLayer",
        data=recent,
        get_position=["lon", "lat"],
        get_radius=25,
        pickable=True,
    )

    view = pdk.ViewState(
        latitude=float(recent["lat"].median()),
        longitude=float(recent["lon"].median()),
        zoom=11,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[heat, pts],
            initial_view_state=view,
            tooltip={"text": "Relative risk: {risk_norm}"},
        ),
        use_container_width=True,
    )

    tree = KDTree(recent[["x", "y"]].to_numpy())
    dist_m, idx = tree.query(top[["x", "y"]].to_numpy(), k=1)

    top2 = top.copy()
    top2["risk_pct"] = (top2["risk_norm"] * 100).round(1)
    top2["nearest_dist_m"] = dist_m[:, 0].round(0).astype(int)
    top2["nearest_dt"] = recent.iloc[idx[:, 0]]["dt"].dt.strftime("%Y-%m-%d").to_numpy()

    top2["Latitude"] = top2["lat"].round(5)
    top2["Longitude"] = top2["lon"].round(5)
    top2["Map"] = top2.apply(lambda row: f"https://www.openstreetmap.org/?mlat={row['lat']}&mlon={row['lon']}&zoom=15", axis=1)
    
    display = top2[["risk_pct", "nearest_dist_m", "nearest_dt", "Latitude", "Longitude", "Map"]].head(20)
    display = display.rename(columns={
        "risk_pct": "Relative risk (%)",
        "nearest_dist_m": "Nearest burglary (m)",
        "nearest_dt": "Nearest burglary date",
    })

    st.dataframe(display, use_container_width=True)
    st.caption(f"Hotspots are grid tiles of ~{cell_size}m × {cell_size}m. Risk is relative (0–100) for the selected day/window.")


#BACKTEST TAB
with tab2:
    st.caption("Backtest = run the model across many days and measure whether burglaries fall inside predicted hotspots.")
    # default: last 30 days in dataset
    end_bt = df["dt"].max().date()
    start_bt = (df["dt"].max() - pd.Timedelta(days=30)).date()

    col1, col2 = st.columns(2)
    with col1:
        bt_start = st.date_input("Backtest start", value=start_bt)
    with col2:
        bt_end = st.date_input("Backtest end", value=end_bt)

    if st.button("Run backtest"):
        with st.spinner("Running backtest (this can take a bit)…"):
            model_df, base_df = backtest_range(
                incidents_xy=df_xy,
                cell_size=cell_size,
                start_date=str(bt_start),
                end_date=str(bt_end),
                topk=topk,
                sigma_m=sigma,
                tau_days=tau,
                lookback_days=lookback,
                radius_m=radius,
            )

        st.subheader("Results (model vs baseline)")
        st.write("Baseline = same spatial model but **no time decay** (ignores recency).")

        # Merge for easy comparison
        m = model_df.rename(columns={"hit_rate": "hit_rate_model", "pai": "pai_model"})
        b = base_df.rename(columns={"hit_rate": "hit_rate_base", "pai": "pai_base"})
        merged = pd.merge(m[["date", "hit_rate_model", "pai_model"]],
                          b[["date", "hit_rate_base", "pai_base"]],
                          on="date", how="inner")

        st.dataframe(merged, use_container_width=True)

        st.subheader("Hit Rate over time")
        st.line_chart(merged.set_index("date")[["hit_rate_model", "hit_rate_base"]])

        st.subheader("PAI over time")
        st.line_chart(merged.set_index("date")[["pai_model", "pai_base"]])

        st.subheader("Summary")
        st.write({
            "avg_hit_rate_model": float(merged["hit_rate_model"].mean()),
            "avg_hit_rate_baseline": float(merged["hit_rate_base"].mean()),
            "avg_pai_model": float(merged["pai_model"].mean()),
            "avg_pai_baseline": float(merged["pai_base"].mean()),
        })