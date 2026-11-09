"""Build the Delhi bus network segment catalog.

Loads the cached ``StaticIndex`` and ``stops.txt``, extracts every consecutive
directed stop pair from each route shape, and groups identical short corridors
into unique physical segments.

Outputs:
    data/processed/segment_catalog.parquet
    data/processed/route_segments.parquet

Run from the project root:
    python -m src.model.segment_catalog

Notes:
- ``segment_order`` is 0-based within each route/shape stop sequence.
- ``bearing_rad`` is clockwise from north, in [0, 2*pi).
- Distances use the local planar approximation around Connaught Place.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.matching.shape_indexer import load_static_index

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "static_index.pkl"
STOPS_PATH = PROJECT_ROOT / "data" / "static" / "stops.txt"
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "segment_catalog.parquet"
ROUTE_SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "route_segments.parquet"

# Connaught Place reference point.
LAT0 = 28.6139
LON0 = 77.2090
M_PER_DEG_LAT = 111_132.92
M_PER_DEG_LON = 97_799.66

MIN_SEGMENT_LENGTH_M = 20.0
LENGTH_BUCKET_M = 100.0
HUB_ROUTE_THRESHOLD = 10
HUB_NAME_PATTERN = "ISBT|TERMINAL|METRO|CROSSING|CHOWK|DEPOT"

CATALOG_COLUMNS = [
    "corridor_segment_id",
    "stop_u",
    "stop_v",
    "segment_length_m",
    "lat_u",
    "lon_u",
    "lat_v",
    "lon_v",
    "mid_lat",
    "mid_lon",
    "dist_from_center_km",
    "bearing_rad",
    "n_routes_sharing",
    "is_major_hub_u",
    "is_major_hub_v",
]


def load_stops(path: Path = STOPS_PATH) -> pl.DataFrame:
    """Load stop coordinates and names, with stop_id forced to Utf8."""
    stops = pl.read_csv(
        path,
        schema_overrides={"stop_id": pl.Utf8, "stop_name": pl.Utf8},
    )
    return stops.with_columns(
        pl.col("stop_id").cast(pl.Utf8),
        pl.col("stop_name").cast(pl.Utf8),
    ).unique(subset=["stop_id"], keep="first", maintain_order=True)


def _corridor_id(stop_u: str, stop_v: str, length_m: float) -> str:
    bucket = int(round(length_m / LENGTH_BUCKET_M) * LENGTH_BUCKET_M)
    return f"{stop_u}_{stop_v}_{bucket}"


def _build_segment_instances(index: Any) -> pl.DataFrame:
    """One row per consecutive directed stop pair on each route shape."""
    rows: list[tuple[str, str, int, str, str, float, str]] = []

    for (route_id, shape_id), info in index.route_shape_info.items():
        stop_ids = info.stop_ids
        stop_dists = info.stop_dists_m

        for order in range(len(stop_ids) - 1):
            stop_u = stop_ids[order]
            stop_v = stop_ids[order + 1]
            length_m = max(
                MIN_SEGMENT_LENGTH_M,
                float(stop_dists[order + 1] - stop_dists[order]),
            )
            rows.append(
                (
                    route_id,
                    shape_id,
                    order,
                    stop_u,
                    stop_v,
                    length_m,
                    _corridor_id(stop_u, stop_v, length_m),
                )
            )

    return pl.DataFrame(
        rows,
        schema={
            "route_id": pl.Utf8,
            "shape_id": pl.Utf8,
            "segment_order": pl.Int32,
            "stop_u": pl.Utf8,
            "stop_v": pl.Utf8,
            "segment_length_m": pl.Float64,
            "corridor_segment_id": pl.Utf8,
        },
        orient="row",
    )


def _build_hub_flags(index: Any, stops: pl.DataFrame) -> pl.DataFrame:
    """Return stop_id -> is_major_hub (Int8)."""
    stop_routes = pl.DataFrame(
        [
            (info.route_id, stop_id)
            for info in index.route_shape_info.values()
            for stop_id in set(info.stop_ids)
        ],
        schema={"route_id": pl.Utf8, "stop_id": pl.Utf8},
        orient="row",
    ).unique()

    route_counts = stop_routes.group_by("stop_id").agg(
        pl.len().cast(pl.Int32).alias("n_routes_serving")
    )

    hubs = route_counts.join(
        stops.select("stop_id", "stop_name"), on="stop_id", how="left"
    ).with_columns(
        (
            (pl.col("n_routes_serving") > HUB_ROUTE_THRESHOLD)
            | pl.col("stop_name")
            .fill_null("")
            .str.to_uppercase()
            .str.contains(HUB_NAME_PATTERN)
        )
        .cast(pl.Int8)
        .alias("is_major_hub")
    )

    return hubs.select("stop_id", "is_major_hub")


def build_catalog(
    index: Any, stops: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (segment catalog, ordered route-to-segment mapping)."""
    segments = _build_segment_instances(index)

    route_segments = segments.select(
        "route_id", "shape_id", "segment_order", "corridor_segment_id"
    ).sort(["route_id", "shape_id", "segment_order"])

    coords = stops.select("stop_id", "stop_lat", "stop_lon")
    segments = segments.join(
        coords.rename(
            {"stop_id": "stop_u", "stop_lat": "lat_u", "stop_lon": "lon_u"}
        ),
        on="stop_u",
        how="left",
    ).join(
        coords.rename(
            {"stop_id": "stop_v", "stop_lat": "lat_v", "stop_lon": "lon_v"}
        ),
        on="stop_v",
        how="left",
    )

    # Collapse repeated traversals by the same route first, so each route
    # contributes one length to the cross-route median.
    per_route = segments.group_by(["corridor_segment_id", "route_id"]).agg(
        pl.col("stop_u").first(),
        pl.col("stop_v").first(),
        pl.col("segment_length_m").median().alias("segment_length_m"),
        pl.col("lat_u").median().alias("lat_u"),
        pl.col("lon_u").median().alias("lon_u"),
        pl.col("lat_v").median().alias("lat_v"),
        pl.col("lon_v").median().alias("lon_v"),
    )

    catalog = per_route.group_by("corridor_segment_id").agg(
        pl.col("stop_u").first(),
        pl.col("stop_v").first(),
        pl.col("segment_length_m").median().alias("segment_length_m"),
        pl.col("lat_u").median().alias("lat_u"),
        pl.col("lon_u").median().alias("lon_u"),
        pl.col("lat_v").median().alias("lat_v"),
        pl.col("lon_v").median().alias("lon_v"),
        pl.len().cast(pl.Int32).alias("n_routes_sharing"),
    )

    lat_u = catalog["lat_u"].to_numpy()
    lon_u = catalog["lon_u"].to_numpy()
    lat_v = catalog["lat_v"].to_numpy()
    lon_v = catalog["lon_v"].to_numpy()

    mid_lat = (lat_u + lat_v) / 2.0
    mid_lon = (lon_u + lon_v) / 2.0

    # Clockwise angle from north, using local east/north metre offsets.
    east_m = (lon_v - lon_u) * M_PER_DEG_LON
    north_m = (lat_v - lat_u) * M_PER_DEG_LAT
    bearing_rad = np.mod(np.arctan2(east_m, north_m), 2.0 * np.pi)

    # Euclidean distance of the segment midpoint from Connaught Place.
    dx_km = (mid_lon - LON0) * M_PER_DEG_LON / 1000.0
    dy_km = (mid_lat - LAT0) * M_PER_DEG_LAT / 1000.0
    dist_from_center_km = np.hypot(dx_km, dy_km)

    catalog = catalog.with_columns(
        pl.Series("mid_lat", mid_lat, dtype=pl.Float64),
        pl.Series("mid_lon", mid_lon, dtype=pl.Float64),
        pl.Series("dist_from_center_km", dist_from_center_km, dtype=pl.Float64),
        pl.Series("bearing_rad", bearing_rad, dtype=pl.Float64),
    )

    hubs = _build_hub_flags(index, stops)
    catalog = catalog.join(
        hubs.rename({"stop_id": "stop_u", "is_major_hub": "is_major_hub_u"}),
        on="stop_u",
        how="left",
    ).join(
        hubs.rename({"stop_id": "stop_v", "is_major_hub": "is_major_hub_v"}),
        on="stop_v",
        how="left",
    )

    catalog = catalog.with_columns(
        pl.col("is_major_hub_u").fill_null(0).cast(pl.Int8),
        pl.col("is_major_hub_v").fill_null(0).cast(pl.Int8),
    ).select(CATALOG_COLUMNS)

    catalog = catalog.sort(
        ["n_routes_sharing", "corridor_segment_id"], descending=[True, False]
    )
    return catalog, route_segments


def save_outputs(
    catalog: pl.DataFrame, route_segments: pl.DataFrame
) -> tuple[Path, Path]:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_parquet(CATALOG_PATH, compression="zstd")
    route_segments.write_parquet(ROUTE_SEGMENTS_PATH, compression="zstd")
    return CATALOG_PATH, ROUTE_SEGMENTS_PATH


def print_summary(catalog: pl.DataFrame, stops: pl.DataFrame) -> None:
    total = catalog.height
    average_routes = float(catalog["n_routes_sharing"].mean() or 0.0)
    stop_names = dict(
        zip(stops["stop_id"].to_list(), stops["stop_name"].to_list())
    )

    print(f"\n[OK] Total unique physical segments: {total:,}")
    print(f"[OK] Average routes per segment: {average_routes:.2f}")
    print("[OK] Top 5 busiest corridor segments:")
    for rank, row in enumerate(catalog.head(5).iter_rows(named=True), start=1):
        name_u = stop_names.get(row["stop_u"]) or row["stop_u"]
        name_v = stop_names.get(row["stop_v"]) or row["stop_v"]
        print(
            f"  {rank}. {row['corridor_segment_id']} | "
            f"{name_u} -> {name_v} | "
            f"routes={row['n_routes_sharing']} | "
            f"length={row['segment_length_m']:.0f}m"
        )


def main() -> int:
    started = time.perf_counter()
    print("Building Delhi bus network segment catalog")

    index = load_static_index(INDEX_PATH)
    stops = load_stops()

    catalog, route_segments = build_catalog(index, stops)
    catalog_path, route_path = save_outputs(catalog, route_segments)

    print_summary(catalog, stops)
    print(f"\n[OK] Wrote {catalog_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Wrote {route_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Routes mapped: {route_segments['route_id'].n_unique():,}")
    print(f"[DONE] {time.perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
