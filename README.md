# Boston Near-Repeat Crime Detector

A theory-driven, interactive hotspot forecasting demo for **Boston** incidents.  
This project implements a **near-repeat** model: after a burglary occurs, the risk of similar incidents increases nearby for a short time window, then fades.

Built with Python + Streamlit, using public incident report data.

---

## What this does

Given historical burglary incidents (latitude, longitude, timestamp), the app:

- Computes a risk heatmap for a chosen prediction date
- Highlights top hotspot tiles (ranked by risk)
- Displays recent burglaries that contribute to risk

---

## Core idea (near-repeat model)

Each incident contributes risk to nearby locations, weighted by:
- distance decay (closer matters more)
- time decay (more recent matters more)

For a grid cell center \(x\) at time \(t\):

\[
R(x,t)=\sum_{i:\, t_i<t}
\exp\left(-\frac{d(x,x_i)}{\sigma}\right)
\cdot
\exp\left(-\frac{t-t_i}{\tau}\right)
\]

Where:
- \(d(x, x_i)\) = distance (meters)
- \(\sigma\) = spatial spread (neighborhood size)
- \(\tau\) = time decay (how quickly risk fades)

To compute efficiently, the model uses a KDTree to query nearby incidents within a radius.

---

## Demo UI (Streamlit)

The dashboard includes:
- Map tab: heatmap + recent burglary points + top hotspot table (with Google Maps link)
- Backtest tab: daily evaluation across a time range

Controls:
- Prediction date
- Recent window (7 / 14 / 21 days)
- Neighborhood size (small / medium / large)