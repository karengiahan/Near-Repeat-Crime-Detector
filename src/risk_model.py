import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

def compute_risk(incidents_xy: pd.DataFrame, grid: pd.DataFrame, t0: pd.Timestamp,
                 sigma_m: float, tau_days: float, lookback_days: int, radius_m: float) -> pd.DataFrame:
    start = t0 - pd.Timedelta(days=lookback_days)
    recent = incidents_xy[(incidents_xy["dt"] < t0) & (incidents_xy["dt"] >= start)].copy()

    out = grid.copy()
    if recent.empty:
        out["risk"] = 0.0
        return out

    pts = recent[["x", "y"]].to_numpy()
    tree = KDTree(pts)

    grid_pts = grid[["x", "y"]].to_numpy()
    neighbors = tree.query_radius(grid_pts, r=radius_m)

    age_days = (t0 - recent["dt"]).dt.total_seconds().to_numpy() / (3600 * 24)
    time_w = np.exp(-age_days / tau_days)

    risks = np.zeros(len(grid), dtype=float)
    for j, idx in enumerate(neighbors):
        if len(idx) == 0:
            continue
        dx = pts[idx, 0] - grid_pts[j, 0]
        dy = pts[idx, 1] - grid_pts[j, 1]
        d = np.sqrt(dx * dx + dy * dy)

        space_w = np.exp(-d / sigma_m)
        risks[j] = float(np.sum(space_w * time_w[idx]))

    out["risk"] = risks
    return out