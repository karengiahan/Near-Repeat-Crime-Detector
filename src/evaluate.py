import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.risk_model import compute_risk

@dataclass(frozen=True)
class GridSpec:
    xmin: float
    ymin: float
    cell_size: int

def build_global_grid(df_xy: pd.DataFrame, cell_size: int) -> tuple[pd.DataFrame, GridSpec]:
    """
    Build a consistent grid over the whole dataset (after Boston coordinate cleaning).
    """
    xmin, xmax = float(df_xy["x"].min()), float(df_xy["x"].max())
    ymin, ymax = float(df_xy["y"].min()), float(df_xy["y"].max())

    xs = np.arange(xmin, xmax + cell_size, cell_size)
    ys = np.arange(ymin, ymax + cell_size, cell_size)
    xc = xs[:-1] + cell_size / 2
    yc = ys[:-1] + cell_size / 2

    X, Y = np.meshgrid(xc, yc)
    grid = pd.DataFrame({"x": X.ravel(), "y": Y.ravel()})

    return grid, GridSpec(xmin=xmin, ymin=ymin, cell_size=cell_size)

def cell_ids_from_xy(x: np.ndarray, y: np.ndarray, spec: GridSpec) -> np.ndarray:
    ix = np.floor((x - spec.xmin) / spec.cell_size).astype(int)
    iy = np.floor((y - spec.ymin) / spec.cell_size).astype(int)
    return ix.astype(str) + "_" + iy.astype(str)

def evaluate_one_day(
    incidents_xy: pd.DataFrame,
    grid: pd.DataFrame,
    spec: GridSpec,
    t0: pd.Timestamp,
    topk: int,
    sigma_m: float,
    tau_days: float,
    lookback_days: int,
    radius_m: float,
    baseline_no_time_decay: bool = False,
) -> dict | None:
    """
    Predict hotspots for day t0 using incidents prior to t0.
    Evaluate against incidents occurring during [t0, t0+1day).
    """
    # compute risk
    if baseline_no_time_decay:
        tau_days = 10**9  # effectively no time decay

    risk_map = compute_risk(
        incidents_xy=incidents_xy,
        grid=grid,
        t0=t0,
        sigma_m=sigma_m,
        tau_days=tau_days,
        lookback_days=lookback_days,
        radius_m=radius_m,
    )

    # actual burglaries in next day
    t1 = t0 + pd.Timedelta(days=1)
    actual = incidents_xy[(incidents_xy["dt"] >= t0) & (incidents_xy["dt"] < t1)]
    if len(actual) == 0:
        return None

    # top-k predicted cells
    top = risk_map.nlargest(topk, "risk")
    top_cells = set(cell_ids_from_xy(top["x"].to_numpy(), top["y"].to_numpy(), spec))

    actual_cells = cell_ids_from_xy(actual["x"].to_numpy(), actual["y"].to_numpy(), spec)
    hits = int(np.isin(actual_cells, list(top_cells)).sum())

    hit_rate = hits / len(actual)
    area_frac = topk / len(grid)
    pai = (hit_rate / area_frac) if area_frac > 0 else float("nan")

    return {
        "date": t0.date().isoformat(),
        "incidents": int(len(actual)),
        "hits": hits,
        "hit_rate": float(hit_rate),
        "pai": float(pai),
    }

def backtest_range(
    incidents_xy: pd.DataFrame,
    cell_size: int,
    start_date: str,
    end_date: str,
    topk: int,
    sigma_m: float,
    tau_days: float,
    lookback_days: int,
    radius_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (model_results, baseline_results)
    baseline = no-time-decay version of same model
    """
    grid, spec = build_global_grid(incidents_xy, cell_size)
    dates = pd.date_range(start_date, end_date, freq="D")

    model_rows = []
    base_rows = []

    for day in dates:
        t0 = pd.Timestamp(day)

        r1 = evaluate_one_day(
            incidents_xy, grid, spec, t0,
            topk, sigma_m, tau_days, lookback_days, radius_m,
            baseline_no_time_decay=False
        )
        if r1:
            model_rows.append(r1)

        r0 = evaluate_one_day(
            incidents_xy, grid, spec, t0,
            topk, sigma_m, tau_days, lookback_days, radius_m,
            baseline_no_time_decay=True
        )
        if r0:
            base_rows.append(r0)

    return pd.DataFrame(model_rows), pd.DataFrame(base_rows)