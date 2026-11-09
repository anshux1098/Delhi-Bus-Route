"""Delhi GTFS-Realtime Data Hygiene and Cleaning Pipeline.

Filters and cleans GPS pings from raw feeds:
1. Casts IDs and coordinate types to strict Polars types.
2. Filters to Delhi NCR bounding box ([28.35, 28.90], [76.80, 77.45]).
3. Deduplicates on (vehicle_id, timestamp) and sorts chronologically.
4. Projects (lat, lon) to planar meters (x_m, y_m) centered at Connaught Place.
5. Computes dt_sec, dx_m, dy_m, and step_dist_m over vehicle_id.
6. Collapses frozen GPS runs (step_dist_m < 1.0m) keeping first and last ping.
7. Filters out teleports (speed > 28.0 m/s or 100 km/h).
8. Computes unit movement vector (vx, vy) for steps >= 20.0m.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Sequence

import polars as pl

# Reference coordinate: Connaught Place, New Delhi
LAT0 = 28.6139
LON0 = 77.2090
M_PER_DEG_LAT = 111132.92
M_PER_DEG_LON = 97799.66

# Delhi NCR Bounding Box
LAT_MIN, LAT_MAX = 28.35, 28.90
LON_MIN, LON_MAX = 76.80, 77.45

# Thresholds
MAX_SPEED_MPS = 28.0  # ~100.8 km/h
FROZEN_THRESH_M = 1.0
BEARING_DIST_THRESH_M = 20.0


def find_project_root() -> Path:
    """Locate project root directory."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "data").exists():
        return candidate
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    if (cwd / "delhi-transit-planner" / "data").exists():
        return cwd / "delhi-transit-planner"
    return candidate


def load_and_clean_pings(
    source: str | Path | pl.DataFrame | Sequence[str | Path] | None = None,
) -> pl.DataFrame:
    """Load raw GTFS-Realtime pings and apply the full hygiene pipeline.

    Args:
        source: Optional DataFrame, directory path, glob pattern, or file list.
                Defaults to searching `data/historical/*.parquet`.

    Returns:
        Cleaned, planar-projected, deduplicated, and motion-scored Polars DataFrame.
    """
    # 1. Resolve inputs
    df: pl.DataFrame
    if isinstance(source, pl.DataFrame):
        df = source
    else:
        file_paths: list[Path] = []
        if source is None:
            root = find_project_root()
            file_paths = sorted(list((root / "data" / "historical").glob("*.parquet")))
        elif isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                file_paths = sorted(list(p.glob("*.parquet")) + list(p.glob("*.csv")))
            else:
                globbed = glob.glob(str(source))
                if globbed:
                    file_paths = [Path(f) for f in sorted(globbed)]
                elif p.exists():
                    file_paths = [p]
        elif isinstance(source, (list, tuple)):
            file_paths = [Path(f) for f in source]

        if not file_paths and not isinstance(source, pl.DataFrame):
            # Return empty schema if nothing found
            return pl.DataFrame(
                schema={
                    "vehicle_id": pl.Utf8,
                    "route_id": pl.Utf8,
                    "trip_id": pl.Utf8,
                    "latitude": pl.Float64,
                    "longitude": pl.Float64,
                    "timestamp": pl.Int64,
                    "x_m": pl.Float64,
                    "y_m": pl.Float64,
                    "dt_sec": pl.Int64,
                    "dx_m": pl.Float64,
                    "dy_m": pl.Float64,
                    "step_dist_m": pl.Float64,
                    "vx": pl.Float64,
                    "vy": pl.Float64,
                    "has_bearing": pl.Boolean,
                }
            )

        dfs: list[pl.DataFrame] = []
        for fp in file_paths:
            if fp.suffix.lower() == ".parquet":
                dfs.append(pl.read_parquet(fp))
            elif fp.suffix.lower() == ".csv":
                dfs.append(pl.read_csv(fp))
        if not dfs:
            return pl.DataFrame()
        df = pl.concat(dfs, how="diagonal")

    if df.is_empty():
        return df

    # 2. Strict type casting for IDs, coordinates, and timestamp
    casts: dict[str, pl.Expr] = {}
    if "vehicle_id" in df.columns:
        casts["vehicle_id"] = pl.col("vehicle_id").cast(pl.Utf8)
    if "route_id" in df.columns:
        casts["route_id"] = pl.col("route_id").cast(pl.Utf8)
    if "trip_id" in df.columns:
        casts["trip_id"] = pl.col("trip_id").cast(pl.Utf8)
    if "latitude" in df.columns:
        casts["latitude"] = pl.col("latitude").cast(pl.Float64)
    if "longitude" in df.columns:
        casts["longitude"] = pl.col("longitude").cast(pl.Float64)
    if "timestamp" in df.columns:
        casts["timestamp"] = pl.col("timestamp").cast(pl.Int64)

    df = df.with_columns(**casts)

    # Filter null coordinates / timestamps / vehicle IDs
    df = df.filter(
        pl.col("vehicle_id").is_not_null()
        & pl.col("latitude").is_not_null()
        & pl.col("longitude").is_not_null()
        & pl.col("timestamp").is_not_null()
        & (pl.col("timestamp") > 0)
    )

    # 3. Filter to Delhi NCR bounding box
    df = df.filter(
        pl.col("latitude").is_between(LAT_MIN, LAT_MAX)
        & pl.col("longitude").is_between(LON_MIN, LON_MAX)
    )

    if df.is_empty():
        return df

    # 4. Deduplicate on (vehicle_id, timestamp) and sort
    df = df.unique(subset=["vehicle_id", "timestamp"], keep="first").sort(
        ["vehicle_id", "timestamp"]
    )

    # 5. Project to planar meters (x_m, y_m) centered at Connaught Place
    df = df.with_columns(
        x_m=(pl.col("longitude") - LON0) * M_PER_DEG_LON,
        y_m=(pl.col("latitude") - LAT0) * M_PER_DEG_LAT,
    )

    # 6. Compute dt_sec, dx_m, dy_m, and step_dist_m over vehicle_id
    df = df.with_columns(
        dt_sec=pl.col("timestamp").diff().over("vehicle_id").fill_null(0),
        dx_m=pl.col("x_m").diff().over("vehicle_id").fill_null(0.0),
        dy_m=pl.col("y_m").diff().over("vehicle_id").fill_null(0.0),
    )
    df = df.with_columns(
        step_dist_m=(pl.col("dx_m").pow(2) + pl.col("dy_m").pow(2)).sqrt()
    )

    # 7. Collapse frozen GPS runs (step_dist_m < 1.0m across consecutive pings):
    # Keep only the first and last ping of each stationary run to preserve dwell time without middle pings.
    df = df.with_columns(
        is_stationary=(pl.col("dt_sec") > 0) & (pl.col("step_dist_m") < FROZEN_THRESH_M)
    )
    df = df.with_columns(
        prev_stat=pl.col("is_stationary").shift(1).over("vehicle_id").fill_null(False),
        next_stat=pl.col("is_stationary").shift(-1).over("vehicle_id").fill_null(False),
    )
    is_middle_stationary = (
        pl.col("is_stationary") & pl.col("prev_stat") & pl.col("next_stat")
    )
    df = df.filter(~is_middle_stationary).drop(["is_stationary", "prev_stat", "next_stat"])

    # Recompute steps after middle pings dropped
    df = df.with_columns(
        dt_sec=pl.col("timestamp").diff().over("vehicle_id").fill_null(0),
        dx_m=pl.col("x_m").diff().over("vehicle_id").fill_null(0.0),
        dy_m=pl.col("y_m").diff().over("vehicle_id").fill_null(0.0),
    )
    df = df.with_columns(
        step_dist_m=(pl.col("dx_m").pow(2) + pl.col("dy_m").pow(2)).sqrt()
    )

    # 8. Filter out impossible teleports where dt_sec > 0 and speed > 28.0 m/s (>100 km/h)
    is_teleport = (pl.col("dt_sec") > 0) & (
        (pl.col("step_dist_m") / pl.col("dt_sec").cast(pl.Float64)) > MAX_SPEED_MPS
    )
    df = df.filter(~is_teleport)

    # Recompute step metrics after teleport removal
    df = df.with_columns(
        dt_sec=pl.col("timestamp").diff().over("vehicle_id").fill_null(0),
        dx_m=pl.col("x_m").diff().over("vehicle_id").fill_null(0.0),
        dy_m=pl.col("y_m").diff().over("vehicle_id").fill_null(0.0),
    )
    df = df.with_columns(
        step_dist_m=(pl.col("dx_m").pow(2) + pl.col("dy_m").pow(2)).sqrt()
    )

    # 9. Compute unit movement vector (vx, vy) for steps >= 20.0m, else (0.0, 0.0)
    has_bearing_expr = pl.col("step_dist_m") >= BEARING_DIST_THRESH_M
    df = df.with_columns(
        vx=pl.when(has_bearing_expr)
        .then(pl.col("dx_m") / pl.col("step_dist_m"))
        .otherwise(0.0),
        vy=pl.when(has_bearing_expr)
        .then(pl.col("dy_m") / pl.col("step_dist_m"))
        .otherwise(0.0),
        has_bearing=has_bearing_expr,
    )

    return df


if __name__ == "__main__":
    import sys
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Running hygiene pipeline on data/historical/*.parquet...")
    clean_df = load_and_clean_pings()
    print(f"Cleaned pings: {clean_df.height:,}")
    print(f"Unique vehicles: {clean_df['vehicle_id'].n_unique():,}")
    print(f"Columns: {clean_df.columns}")
    if clean_df.height > 0:
        print(clean_df.select([
            "vehicle_id", "route_id", "timestamp", "x_m", "y_m", "step_dist_m", "vx", "vy", "has_bearing"
        ]).head(5))
