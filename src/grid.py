import numpy as np
import pandas as pd
from dataclasses import dataclass
from pyproj import Transformer

_WGS84 = "EPSG:4326"
_WEBM = "EPSG:3857"  # meters

_fwd = Transformer.from_crs(_WGS84, _WEBM, always_xy=True)
_inv = Transformer.from_crs(_WEBM, _WGS84, always_xy=True)

@dataclass(frozen=True)
class GridSpec:
    xmin: float
    ymin: float
    cell_size_m: int

def add_xy_meters(df: pd.DataFrame) -> pd.DataFrame:
    x, y = _fwd.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    out = df.copy()
    out["x"] = x
    out["y"] = y
    return out

def make_grid(df_xy: pd.DataFrame, cell_size_m: int):
    xmin, xmax = df_xy["x"].min(), df_xy["x"].max()
    ymin, ymax = df_xy["y"].min(), df_xy["y"].max()

    xs = np.arange(xmin, xmax + cell_size_m, cell_size_m)
    ys = np.arange(ymin, ymax + cell_size_m, cell_size_m)

    xc = xs[:-1] + cell_size_m / 2
    yc = ys[:-1] + cell_size_m / 2

    X, Y = np.meshgrid(xc, yc)
    grid = pd.DataFrame({"x": X.ravel(), "y": Y.ravel()})

    lon, lat = _inv.transform(grid["x"].to_numpy(), grid["y"].to_numpy())
    grid["lat"] = lat
    grid["lon"] = lon

    return grid, GridSpec(float(xmin), float(ymin), int(cell_size_m))