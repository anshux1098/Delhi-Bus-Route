"""Offline Delhi transit graph with footpaths, headways and common lines.

Compiles the static GTFS network and precomputed travel-time tensors into a
single adjacency structure for RAPTOR-style routing:

- Contiguous integer stop and route indices.
- CSR route-stop sequences with the matching ``mu_tensor`` corridor row.
- Transitively closed 500 m walking footpaths with Delhi transfer penalties.
- Empirical route headways, headway CV2, expected waits and RT coverage.
- Chriqui-Robillard attractive sets for common-line stop pairs.

Run from the project root::

    python -m src.router.transit_graph
"""

from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from src.matching.shape_indexer import load_static_index

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STATIC_DIR = PROJECT_ROOT / "data" / "static"
HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"

INDEX_PATH = PROCESSED_DIR / "static_index.pkl"
ROUTE_SEGMENTS_PATH = PROCESSED_DIR / "route_segments.parquet"
BASELINE_TENSORS_PATH = PROCESSED_DIR / "baseline_tensors.npz"
OBSERVED_SEGMENTS_PATH = PROCESSED_DIR / "observed_segments.parquet"
STOPS_PATH = STATIC_DIR / "stops.txt"
ROUTES_PATH = STATIC_DIR / "routes.txt"
TRIPS_PATH = STATIC_DIR / "trips.txt"
STOP_TIMES_PATH = STATIC_DIR / "stop_times.txt"
OUTPUT_PATH = PROCESSED_DIR / "transit_graph.pkl"

# Local planar approximation around Connaught Place.
LAT0 = 28.6139
LON0 = 77.2090
M_PER_DEG_LAT = 111_132.92
M_PER_DEG_LON = 97_799.66

MAX_TRANSFER_WALK_M = 500.0
WALK_DETOUR_FACTOR = 1.3
WALK_SPEED_MPS = 1.25  # 4.5 km/h

SAME_STOP_PENALTY_SEC = 180.0
CROSS_STOP_PENALTY_FACTOR = 1.5

BUCKET_NAMES = (
    "Night 00-06",
    "Morning Peak 06-11",
    "Midday 11-17",
    "Evening Peak 17-24",
)
N_BUCKETS = len(BUCKET_NAMES)
BUCKET_BIN_SLICES = ((0, 24), (24, 44), (44, 68), (68, 96))

EH_MIN_SEC = 180.0
EH_MAX_SEC = 2400.0
EH_FALLBACK_SEC = 1200.0
CV2_FALLBACK = 0.50
CV2_MIN_GPS_GAPS = 3
GPS_HEADWAY_MIN_SEC = 60.0
GPS_HEADWAY_MAX_SEC = 7200.0

COV_MIN = 0.15
COV_MAX = 0.90
COV_DEFAULT = 0.65

DIST_AGREEMENT_TOL = 0.15


@dataclass
class TransitGraph:
    """Compiled offline transit graph."""

    # Stops
    stop_ids: np.ndarray
    stop_names: np.ndarray
    stop_lat: np.ndarray
    stop_lon: np.ndarray
    stop_x_m: np.ndarray
    stop_y_m: np.ndarray
    stop_id_to_idx: dict[str, int]
    n_stops: int

    # Routes
    route_ids: np.ndarray
    route_short_names: np.ndarray
    route_long_names: np.ndarray
    route_id_to_idx: dict[str, int]
    n_routes: int
    n_scheduled_trips: np.ndarray

    # CSR route -> ordered stops and corridor rows
    route_stop_offsets: np.ndarray
    route_stops: np.ndarray
    route_stop_dists_m: np.ndarray
    route_corridor_idxs: np.ndarray

    # CSR walking footpaths
    footpath_offsets: np.ndarray
    footpath_target_stops: np.ndarray
    footpath_walk_sec: np.ndarray
    footpath_penalty_sec: np.ndarray
    footpath_dist_m: np.ndarray

    # Headways and reliability
    EH_sec: np.ndarray
    CV2: np.ndarray
    EW_sec: np.ndarray
    cov_r: np.ndarray

    # Common-line attractive sets, sorted by (u * n_stops + v) * 4 + bucket
    cl_keys: np.ndarray
    cl_offsets: np.ndarray
    cl_routes: np.ndarray
    cl_lambda_prior: np.ndarray
    cl_wait_prior: np.ndarray
    cl_tbar_prior: np.ndarray
    cl_vride_prior: np.ndarray
    cl_ff_ride_prior: np.ndarray
    cl_canonical_route: np.ndarray

    # Corridor travel-time summaries used to build common lines
    corridor_bucket_sec: np.ndarray
    corridor_free_flow_sec: np.ndarray

    bucket_names: tuple[str, ...]
    built_at: str
    crc_version: str = "chriqui-robillard-v1"

    def lookup_common_lines(
        self, u_idx: int, v_idx: int, bucket: int
    ) -> dict[str, Any] | None:
        """Return attractive-set data for a directed stop pair and bucket."""
        key = (int(u_idx) * self.n_stops + int(v_idx)) * N_BUCKETS + int(bucket)
        pos = int(np.searchsorted(self.cl_keys, key))
        if pos >= self.cl_keys.size or int(self.cl_keys[pos]) != key:
            return None
        start = int(self.cl_offsets[pos])
        end = int(self.cl_offsets[pos + 1])
        return {
            "routes": self.cl_routes[start:end].copy(),
            "lambda_prior": float(self.cl_lambda_prior[pos]),
            "wait_sec": float(self.cl_wait_prior[pos]),
            "t_bar_sec": float(self.cl_tbar_prior[pos]),
            "v_ride_prior": float(self.cl_vride_prior[pos]),
            "ff_ride_sec": float(self.cl_ff_ride_prior[pos]),
            "canonical_route": int(self.cl_canonical_route[pos]),
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _utf8(name: str) -> pl.DataType:
    return pl.Utf8


def _numeric_sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (10**18, text)


def _bucket_expr(hours: pl.Expr) -> pl.Expr:
    return (
        pl.when(hours < 6.0)
        .then(0)
        .when(hours < 11.0)
        .then(1)
        .when(hours < 17.0)
        .then(2)
        .otherwise(3)
        .cast(pl.Int32)
    )


def load_inputs() -> dict[str, Any]:
    t0 = time.perf_counter()
    index = load_static_index(INDEX_PATH)

    stops = pl.read_csv(
        STOPS_PATH,
        schema_overrides={"stop_id": pl.Utf8, "stop_name": pl.Utf8},
    ).with_columns(pl.col("stop_id").cast(pl.Utf8), pl.col("stop_name").cast(pl.Utf8))

    routes = pl.read_csv(
        ROUTES_PATH,
        schema_overrides={
            "route_id": pl.Utf8,
            "route_short_name": pl.Utf8,
            "route_long_name": pl.Utf8,
        },
    ).with_columns(
        pl.col("route_id").cast(pl.Utf8),
        pl.col("route_short_name").cast(pl.Utf8),
        pl.col("route_long_name").cast(pl.Utf8),
    )

    trips = pl.read_csv(
        TRIPS_PATH,
        schema_overrides={
            "route_id": pl.Utf8,
            "trip_id": pl.Utf8,
            "service_id": pl.Utf8,
            "shape_id": pl.Utf8,
            "direction_id": pl.Utf8,
        },
        infer_schema_length=50_000,
    )

    stop_times = pl.read_csv(
        STOP_TIMES_PATH,
        schema_overrides={"trip_id": pl.Utf8, "stop_id": pl.Utf8},
        infer_schema_length=50_000,
    )

    route_segments = pl.read_parquet(ROUTE_SEGMENTS_PATH)
    tensors = np.load(BASELINE_TENSORS_PATH, allow_pickle=True)

    print(f"[OK] Loaded inputs in {time.perf_counter() - t0:.2f}s")
    return {
        "index": index,
        "stops": stops,
        "routes": routes,
        "trips": trips,
        "stop_times": stop_times,
        "route_segments": route_segments,
        "mu_tensor": tensors["mu_tensor"],
        "free_flow_mu": tensors["free_flow_mu"],
        "corridors": tensors["corridors"],
    }


# ---------------------------------------------------------------------------
# Stops and routes
# ---------------------------------------------------------------------------


def build_stop_index(stops: pl.DataFrame) -> dict[str, Any]:
    ordered = sorted(stops["stop_id"].unique().to_list(), key=_numeric_sort_key)
    stop_ids = np.asarray(ordered, dtype=object)
    n_stops = len(ordered)

    lookup = stops.unique(subset=["stop_id"], keep="first").select(
        "stop_id", "stop_name", "stop_lat", "stop_lon"
    )
    by_id = {
        row["stop_id"]: row for row in lookup.iter_rows(named=True)
    }

    names = np.empty(n_stops, dtype=object)
    lat = np.empty(n_stops, dtype=np.float64)
    lon = np.empty(n_stops, dtype=np.float64)
    for i, stop_id in enumerate(ordered):
        row = by_id[stop_id]
        names[i] = row["stop_name"]
        lat[i] = float(row["stop_lat"])
        lon[i] = float(row["stop_lon"])

    x_m = (lon - LON0) * M_PER_DEG_LON
    y_m = (lat - LAT0) * M_PER_DEG_LAT

    return {
        "stop_ids": stop_ids,
        "stop_names": names,
        "stop_lat": lat,
        "stop_lon": lon,
        "stop_x_m": x_m,
        "stop_y_m": y_m,
        "stop_id_to_idx": {sid: i for i, sid in enumerate(ordered)},
        "n_stops": n_stops,
    }


def build_route_index(routes: pl.DataFrame) -> dict[str, Any]:
    ordered = sorted(routes["route_id"].unique().to_list(), key=_numeric_sort_key)
    lookup = routes.unique(subset=["route_id"], keep="first").select(
        "route_id", "route_short_name", "route_long_name"
    )
    by_id = {row["route_id"]: row for row in lookup.iter_rows(named=True)}

    n_routes = len(ordered)
    short_names = np.empty(n_routes, dtype=object)
    long_names = np.empty(n_routes, dtype=object)
    for i, route_id in enumerate(ordered):
        row = by_id[route_id]
        short_names[i] = row["route_short_name"]
        long_names[i] = row["route_long_name"]

    return {
        "route_ids": np.asarray(ordered, dtype=object),
        "route_short_names": short_names,
        "route_long_names": long_names,
        "route_id_to_idx": {rid: i for i, rid in enumerate(ordered)},
        "n_routes": n_routes,
    }


def build_route_csr(
    index: Any,
    route_index: dict[str, Any],
    stop_index: dict[str, Any],
    route_segments: pl.DataFrame,
    corridors: np.ndarray,
) -> dict[str, Any]:
    """Flatten route stop sequences and their corridor rows into CSR arrays."""
    n_routes = route_index["n_routes"]
    route_ids = route_index["route_ids"]
    stop_to_idx = stop_index["stop_id_to_idx"]

    corridor_row = {cid: i for i, cid in enumerate(corridors.tolist())}
    seg_frame = route_segments.select(
        "route_id", "segment_order", "corridor_segment_id"
    )
    seg_lookup = {
        (row["route_id"], int(row["segment_order"])): row["corridor_segment_id"]
        for row in seg_frame.iter_rows(named=True)
    }

    offsets = np.zeros(n_routes + 1, dtype=np.int32)
    flat_stops: list[int] = []
    flat_dists: list[float] = []
    flat_corridors: list[int] = []

    for route_idx, route_id in enumerate(route_ids):
        shapes = index.route_to_shapes.get(route_id)
        if not shapes:
            offsets[route_idx + 1] = offsets[route_idx]
            continue
        info = index.route_shape_info[(route_id, shapes[0])]
        stop_ids = info.stop_ids
        dists = np.asarray(info.stop_dists_m, dtype=np.float64)
        if dists.shape[0] != len(stop_ids):
            raise ValueError(
                f"stop_dists_m length mismatch for route {route_id}: "
                f"{dists.shape[0]} != {len(stop_ids)}"
            )

        for order, stop_id in enumerate(stop_ids):
            flat_stops.append(stop_to_idx[stop_id])
            flat_dists.append(float(dists[order]))
            if order + 1 < len(stop_ids):
                corridor_id = seg_lookup[(route_id, order)]
                flat_corridors.append(corridor_row[corridor_id])
            else:
                flat_corridors.append(-1)

        offsets[route_idx + 1] = len(flat_stops)

    return {
        "route_stop_offsets": offsets,
        "route_stops": np.asarray(flat_stops, dtype=np.int32),
        "route_stop_dists_m": np.asarray(flat_dists, dtype=np.float64),
        "route_corridor_idxs": np.asarray(flat_corridors, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Footpaths
# ---------------------------------------------------------------------------


def build_footpaths(stop_index: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    n_stops = stop_index["n_stops"]
    coords = np.column_stack([stop_index["stop_x_m"], stop_index["stop_y_m"]])

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=MAX_TRANSFER_WALK_M, output_type="ndarray")
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    euclid = np.hypot(
        coords[pairs[:, 0], 0] - coords[pairs[:, 1], 0],
        coords[pairs[:, 0], 1] - coords[pairs[:, 1], 1],
    ) if pairs.size else np.empty(0, dtype=np.float64)
    walk_dist = WALK_DETOUR_FACTOR * euclid
    walk_sec = walk_dist / WALK_SPEED_MPS
    penalty = CROSS_STOP_PENALTY_FACTOR * walk_sec + SAME_STOP_PENALTY_SEC

    # Both directions, plus a self edge for the 3-minute same-stop transfer.
    src = np.concatenate([np.arange(n_stops, dtype=np.int64), pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([np.arange(n_stops, dtype=np.int64), pairs[:, 1], pairs[:, 0]])
    dist = np.concatenate([np.zeros(n_stops), walk_dist, walk_dist])
    walk = np.concatenate([np.zeros(n_stops), walk_sec, walk_sec])
    pen = np.concatenate(
        [
            np.full(n_stops, SAME_STOP_PENALTY_SEC, dtype=np.float64),
            penalty,
            penalty,
        ]
    )

    order = np.lexsort((dst, src))
    src = src[order]
    dst = dst[order]
    dist = dist[order]
    walk = walk[order]
    pen = pen[order]

    offsets = np.zeros(n_stops + 1, dtype=np.int32)
    np.add.at(offsets, src + 1, 1)
    np.cumsum(offsets, out=offsets)

    print(
        f"[OK] Footpaths: {len(pairs):,} nearby undirected pairs, "
        f"{len(src):,} directed edges in {time.perf_counter() - t0:.2f}s"
    )
    return {
        "footpath_offsets": offsets,
        "footpath_target_stops": dst.astype(np.int32),
        "footpath_walk_sec": walk.astype(np.float64),
        "footpath_penalty_sec": pen.astype(np.float64),
        "footpath_dist_m": dist.astype(np.float64),
        "n_nearby_pairs": int(len(pairs)),
    }


# ---------------------------------------------------------------------------
# Headways, CV2, coverage
# ---------------------------------------------------------------------------


def _first_departures(
    stop_times: pl.DataFrame, trips: pl.DataFrame, route_index: dict[str, Any]
) -> pl.DataFrame:
    st = stop_times.with_columns(pl.col("stop_sequence").cast(pl.Int32))
    first = (
        st.sort(["trip_id", "stop_sequence"])
        .group_by("trip_id", maintain_order=True)
        .first()
        .select("trip_id", "departure_time")
    )

    parts = pl.col("departure_time").str.split(":")
    dep_sec = (
        parts.list.get(0).cast(pl.Int64) * 3600
        + parts.list.get(1).cast(pl.Int64) * 60
        + parts.list.get(2).cast(pl.Int64)
    )
    first = first.with_columns(dep_sec.alias("dep_sec")).filter(
        pl.col("dep_sec").is_not_null() & (pl.col("dep_sec") >= 0)
    )

    first = first.join(
        trips.select("trip_id", "route_id", "direction_id"),
        on="trip_id",
        how="left",
    ).filter(pl.col("route_id").is_not_null())

    mapping = route_index["route_id_to_idx"]
    first = first.with_columns(
        pl.col("route_id").replace_strict(mapping, return_dtype=pl.Int32).alias("route_idx")
    )
    first = first.with_columns(
        _bucket_expr(pl.col("dep_sec").cast(pl.Float64) / 3600.0).alias("bucket")
    )
    return first


def build_headways(
    first_departures: pl.DataFrame,
    route_index: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    n_routes = route_index["n_routes"]

    trips_per_route = (
        first_departures.group_by("route_idx")
        .agg(pl.len().cast(pl.Int32).alias("n_trips"))
    )
    n_scheduled = np.zeros(n_routes, dtype=np.int32)
    for row in trips_per_route.iter_rows(named=True):
        n_scheduled[row["route_idx"]] = row["n_trips"]

    ordered = first_departures.sort(
        ["route_idx", "direction_id", "bucket", "dep_sec"]
    )
    ordered = ordered.with_columns(
        (
            pl.col("dep_sec")
            - pl.col("dep_sec")
            .shift(1)
            .over(["route_idx", "direction_id", "bucket"])
        ).alias("gap")
    ).filter(pl.col("gap").is_not_null() & (pl.col("gap") > 0))

    per_dir = ordered.group_by(["route_idx", "direction_id", "bucket"]).agg(
        pl.col("gap").mean().alias("headway_sec")
    )

    # Combine directions: 1 / EH = sum over directions of 1 / h_direction.
    eh_raw = (
        per_dir.group_by(["route_idx", "bucket"])
        .agg((1.0 / pl.col("headway_sec")).sum().alias("inv_h"))
        .with_columns((1.0 / pl.col("inv_h")).alias("eh_raw"))
        .select("route_idx", "bucket", "eh_raw")
    )

    route_med = eh_raw.group_by("route_idx").agg(
        pl.col("eh_raw").median().alias("route_med")
    )
    global_med = eh_raw.group_by("bucket").agg(
        pl.col("eh_raw").median().alias("global_med")
    )

    grid = pl.DataFrame(
        {
            "route_idx": np.repeat(np.arange(n_routes, dtype=np.int32), N_BUCKETS),
            "bucket": np.tile(np.arange(N_BUCKETS, dtype=np.int32), n_routes),
        }
    )
    grid = (
        grid.join(eh_raw, on=["route_idx", "bucket"], how="left")
        .join(route_med, on="route_idx", how="left")
        .join(global_med, on="bucket", how="left")
        .with_columns(
            pl.coalesce(
                [
                    pl.col("eh_raw"),
                    pl.col("route_med"),
                    pl.col("global_med"),
                    pl.lit(EH_FALLBACK_SEC),
                ]
            )
            .clip(EH_MIN_SEC, EH_MAX_SEC)
            .alias("EH_sec")
        )
    )
    grid = grid.sort(["route_idx", "bucket"])
    EH = grid["EH_sec"].to_numpy().reshape(n_routes, N_BUCKETS)

    print(
        f"[OK] Scheduled headways from {first_departures.height:,} trips in "
        f"{time.perf_counter() - t0:.2f}s"
    )
    return {"EH_sec": EH.astype(np.float32), "n_scheduled_trips": n_scheduled}


def build_cv2(
    observed_segments: pl.DataFrame, route_index: dict[str, Any]
) -> np.ndarray:
    """Measure headway bunching from GPS arrival gaps at the same corridor."""
    t0 = time.perf_counter()
    n_routes = route_index["n_routes"]

    if observed_segments.height == 0:
        return np.full((n_routes, N_BUCKETS), CV2_FALLBACK, dtype=np.float32)

    obs = (
        observed_segments.select(
            "route_id",
            "corridor_segment_id",
            "trip_run_id",
            "dep_time_u",
            "hour_float",
        )
        .unique(subset=["route_id", "corridor_segment_id", "trip_run_id"])
        .sort(["route_id", "corridor_segment_id", "dep_time_u"])
    )
    obs = obs.with_columns(
        (
            pl.col("dep_time_u")
            - pl.col("dep_time_u")
            .shift(1)
            .over(["route_id", "corridor_segment_id"])
        ).alias("gap")
    ).filter(
        pl.col("gap").is_not_null()
        & (pl.col("gap") >= GPS_HEADWAY_MIN_SEC)
        & (pl.col("gap") <= GPS_HEADWAY_MAX_SEC)
    )

    obs = obs.with_columns(
        _bucket_expr(pl.col("hour_float")).alias("bucket"),
        pl.col("route_id")
        .replace_strict(
            route_index["route_id_to_idx"], default=-1, return_dtype=pl.Int32
        )
        .alias("route_idx"),
    ).filter(pl.col("route_idx") >= 0)

    agg = obs.group_by(["route_idx", "bucket"]).agg(
        pl.len().alias("n"),
        pl.col("gap").mean().alias("mean_gap"),
        pl.col("gap").std().alias("sd_gap"),
    )
    agg = agg.filter(
        (pl.col("n") >= CV2_MIN_GPS_GAPS)
        & (pl.col("mean_gap") > 0)
        & pl.col("sd_gap").is_not_null()
    ).with_columns(
        ((pl.col("sd_gap") / pl.col("mean_gap")).pow(2))
        .clip(0.0, 4.0)
        .alias("cv2")
    )

    cv2 = np.full((n_routes, N_BUCKETS), CV2_FALLBACK, dtype=np.float32)
    for row in agg.iter_rows(named=True):
        cv2[row["route_idx"], row["bucket"]] = float(row["cv2"])

    print(
        f"[OK] CV2 measured for {agg.height:,} route-bucket cells "
        f"({obs.height:,} GPS gaps) in {time.perf_counter() - t0:.2f}s"
    )
    return cv2


def build_coverage(
    first_departures: pl.DataFrame, route_index: dict[str, Any]
) -> np.ndarray:
    """RT fleet coverage: observed vehicles / scheduled vehicles in feed hours."""
    t0 = time.perf_counter()
    n_routes = route_index["n_routes"]
    cov = np.full(n_routes, COV_DEFAULT, dtype=np.float32)

    files = sorted(HISTORICAL_DIR.glob("*.parquet"))
    if not files:
        print("[WARN] No historical RT snapshots; using default coverage")
        return cov

    frames = []
    for path in files:
        try:
            frames.append(
                pl.read_parquet(path).select(
                    "vehicle_id", "route_id", "timestamp", "ist_time"
                )
            )
        except Exception:
            continue
    if not frames:
        return cov

    pings = pl.concat(frames).unique(
        subset=["vehicle_id", "route_id", "timestamp"]
    )
    pings = pings.with_columns(
        pl.col("ist_time").str.slice(11, 2).cast(pl.Int32).alias("hour")
    ).filter(pl.col("route_id").is_not_null())

    observed_hours = sorted(
        int(h) for h in pings["hour"].drop_nulls().unique().to_list() if h is not None
    )
    if not observed_hours:
        return cov

    obs_veh = pings.group_by("route_id").agg(
        pl.col("vehicle_id").n_unique().cast(pl.Int32).alias("obs_veh")
    )
    obs_veh = obs_veh.with_columns(
        pl.col("route_id")
        .replace_strict(
            route_index["route_id_to_idx"], default=-1, return_dtype=pl.Int32
        )
        .alias("route_idx")
    ).filter(pl.col("route_idx") >= 0).select("route_idx", "obs_veh")

    sched = first_departures.with_columns(
        (pl.col("dep_sec") // 3600).cast(pl.Int32).alias("hour")
    ).filter(pl.col("hour").is_in(observed_hours))
    sched = sched.group_by("route_idx").agg(
        pl.len().cast(pl.Int32).alias("sched_vehicles")
    )

    merged = obs_veh.join(sched, on="route_idx", how="left").with_columns(
        pl.col("sched_vehicles").fill_null(1).clip(1, None)
    )
    merged = merged.with_columns(
        (pl.col("obs_veh") / pl.col("sched_vehicles"))
        .clip(COV_MIN, COV_MAX)
        .alias("cov")
    )
    for row in merged.iter_rows(named=True):
        cov[row["route_idx"]] = float(row["cov"])

    print(
        f"[OK] RT coverage from {pings.height:,} pings across "
        f"{len(observed_hours)} observed hours in {time.perf_counter() - t0:.2f}s"
    )
    return cov


# ---------------------------------------------------------------------------
# Common lines (Chriqui-Robillard)
# ---------------------------------------------------------------------------


def corridor_bucket_times(
    mu_tensor: np.ndarray, free_flow_mu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weekday mean travel time per corridor for each time bucket."""
    weekday = mu_tensor[:, :, 0]
    bucket_sec = np.empty((mu_tensor.shape[0], N_BUCKETS), dtype=np.float64)
    for b, (start, end) in enumerate(BUCKET_BIN_SLICES):
        bucket_sec[:, b] = weekday[:, start:end].mean(axis=1)
    return bucket_sec, free_flow_mu.astype(np.float64)


def _route_prefix_times(
    route_stop_offsets: np.ndarray,
    route_corridor_idxs: np.ndarray,
    bucket_sec: np.ndarray,
    ff_sec: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return per-route cumulative time prefixes (N_BUCKETS, FF)."""
    n_routes = len(route_stop_offsets) - 1
    bucket_prefix: list[np.ndarray] = []
    ff_prefix: list[np.ndarray] = []

    for r in range(n_routes):
        start = int(route_stop_offsets[r])
        end = int(route_stop_offsets[r + 1])
        n = end - start
        corr = route_corridor_idxs[start:end]
        valid = corr >= 0
        seg_b = np.zeros((n, N_BUCKETS), dtype=np.float64)
        seg_f = np.zeros(n, dtype=np.float64)
        seg_b[valid] = bucket_sec[corr[valid]]
        seg_f[valid] = ff_sec[corr[valid]]

        pre_b = np.zeros((n, N_BUCKETS), dtype=np.float64)
        pre_f = np.zeros(n, dtype=np.float64)
        if n > 1:
            pre_b[1:] = np.cumsum(seg_b[:-1], axis=0)
            pre_f[1:] = np.cumsum(seg_f[:-1])
        bucket_prefix.append(pre_b)
        ff_prefix.append(pre_f)

    return bucket_prefix, ff_prefix


def _build_common_line_entries(
    route_index: dict[str, Any],
    stop_index: dict[str, Any],
    route_csr: dict[str, Any],
    bucket_sec: np.ndarray,
    ff_sec: np.ndarray,
    EW_sec: np.ndarray,
    index: Any,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    n_routes = route_index["n_routes"]
    n_stops = stop_index["n_stops"]
    route_ids = route_index["route_ids"]
    stop_to_idx = stop_index["stop_id_to_idx"]

    bucket_prefix, ff_prefix = _route_prefix_times(
        route_csr["route_stop_offsets"],
        route_csr["route_corridor_idxs"],
        bucket_sec,
        ff_sec,
    )

    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    rids: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    times: list[np.ndarray] = []
    ffs: list[np.ndarray] = []

    for r, route_id in enumerate(route_ids):
        shapes = index.route_to_shapes.get(route_id)
        if not shapes:
            continue
        info = index.route_shape_info[(route_id, shapes[0])]
        stop_ids = info.stop_ids
        n = len(stop_ids)
        if n < 2:
            continue

        s = np.fromiter(
            (stop_to_idx[sid] for sid in stop_ids), dtype=np.int64, count=n
        )
        d = np.asarray(info.stop_dists_m, dtype=np.float64)
        iu, iv = np.triu_indices(n, 1)

        us.append(s[iu])
        vs.append(s[iv])
        rids.append(np.full(len(iu), r, dtype=np.int32))
        dists.append(d[iv] - d[iu])
        times.append(bucket_prefix[r][iv] - bucket_prefix[r][iu])
        ffs.append(ff_prefix[r][iv] - ff_prefix[r][iu])

    p_u = np.concatenate(us)
    p_v = np.concatenate(vs)
    p_r = np.concatenate(rids)
    p_d = np.concatenate(dists)
    p_t = np.concatenate(times)
    p_f = np.concatenate(ffs)

    # Drop repeated traversals of the same (u, v) by the same route.
    unique_key = (p_u * n_stops + p_v) * n_routes + p_r
    _, first_idx = np.unique(unique_key, return_index=True)
    first_idx.sort()
    p_u = p_u[first_idx]
    p_v = p_v[first_idx]
    p_r = p_r[first_idx]
    p_d = p_d[first_idx]
    p_t = p_t[first_idx]
    p_f = p_f[first_idx]

    pair_key = p_u * n_stops + p_v
    order = np.lexsort((p_d, p_r, pair_key))
    p_u = p_u[order]
    p_v = p_v[order]
    p_r = p_r[order]
    p_d = p_d[order]
    p_t = p_t[order]
    p_f = p_f[order]
    pair_key = pair_key[order]

    starts = np.flatnonzero(
        np.r_[True, pair_key[1:] != pair_key[:-1]]
    )
    counts = np.diff(np.r_[starts, pair_key.size])

    ew = EW_sec.astype(np.float64)
    out_keys: list[int] = []
    out_routes: list[np.ndarray] = []
    out_lambda: list[float] = []
    out_wait: list[float] = []
    out_tbar: list[float] = []
    out_vride: list[float] = []
    out_ff: list[float] = []
    out_canonical: list[int] = []

    for st, count in zip(starts, counts):
        if count < 2:
            continue
        en = st + int(count)

        d = p_d[st:en]
        if d.min() <= 0:
            continue
        med = float(np.median(d))
        if med <= 0:
            continue
        keep = np.abs(d - med) <= DIST_AGREEMENT_TOL * med
        if int(keep.sum()) < 2:
            continue

        cand_r = p_r[st:en][keep]
        cand_t = p_t[st:en][keep]
        cand_f = p_f[st:en][keep]
        u = int(p_u[st])
        v = int(p_v[st])
        base_key = (u * n_stops + v) * N_BUCKETS

        for b in range(N_BUCKETS):
            t_b = cand_t[:, b]
            sort_idx = np.argsort(t_b, kind="stable")

            chosen: list[int] = []
            lam = 0.0
            weight_time = 0.0
            first_added = True

            for pos in sort_idx:
                r = int(cand_r[pos])
                t_r = float(t_b[pos])
                w_r = 1.0 / ew[r, b]

                if first_added:
                    accept = True
                else:
                    wait_a = 1.0 / lam
                    t_bar_a = weight_time / lam
                    accept = t_r < wait_a + t_bar_a

                if not accept:
                    break

                chosen.append(pos)
                lam += w_r
                weight_time += w_r * t_r
                first_added = False

            if len(chosen) < 2:
                continue

            chosen_arr = np.asarray(chosen, dtype=np.int64)
            routes_a = cand_r[chosen_arr].astype(np.int32)
            t_a = t_b[chosen_arr]
            f_a = cand_f[chosen_arr]
            w_a = 1.0 / ew[routes_a, b]

            lam_total = float(w_a.sum())
            t_bar = float((w_a * t_a).sum() / lam_total)
            v_ride = float((w_a * (t_a - t_bar) ** 2).sum() / lam_total)
            ff_bar = float((w_a * f_a).sum() / lam_total)
            canonical = int(routes_a[int(np.argmin(t_a))])

            out_keys.append(base_key + b)
            out_routes.append(routes_a)
            out_lambda.append(lam_total)
            out_wait.append(1.0 / lam_total)
            out_tbar.append(t_bar)
            out_vride.append(v_ride)
            out_ff.append(ff_bar)
            out_canonical.append(canonical)

    if out_keys:
        cl_keys = np.asarray(out_keys, dtype=np.int64)
        sort_order = np.argsort(cl_keys, kind="stable")
        cl_keys = cl_keys[sort_order]
        out_routes = [out_routes[i] for i in sort_order]
        cl_lambda = np.asarray(out_lambda, dtype=np.float64)[sort_order]
        cl_wait = np.asarray(out_wait, dtype=np.float64)[sort_order]
        cl_tbar = np.asarray(out_tbar, dtype=np.float64)[sort_order]
        cl_vride = np.asarray(out_vride, dtype=np.float64)[sort_order]
        cl_ff = np.asarray(out_ff, dtype=np.float64)[sort_order]
        cl_canonical = np.asarray(out_canonical, dtype=np.int32)[sort_order]

        lengths = np.fromiter(
            (len(a) for a in out_routes), dtype=np.int32, count=len(out_routes)
        )
        cl_offsets = np.zeros(len(out_routes) + 1, dtype=np.int32)
        np.cumsum(lengths, out=cl_offsets[1:])
        cl_routes = np.concatenate(out_routes).astype(np.int32, copy=False)

        unique_pairs = len(
            set(int(k) // N_BUCKETS for k in cl_keys.tolist())
        )
    else:
        cl_keys = np.empty(0, dtype=np.int64)
        cl_offsets = np.zeros(1, dtype=np.int32)
        cl_routes = np.empty(0, dtype=np.int32)
        cl_lambda = np.empty(0, dtype=np.float64)
        cl_wait = np.empty(0, dtype=np.float64)
        cl_tbar = np.empty(0, dtype=np.float64)
        cl_vride = np.empty(0, dtype=np.float64)
        cl_ff = np.empty(0, dtype=np.float64)
        cl_canonical = np.empty(0, dtype=np.int32)
        unique_pairs = 0

    print(
        f"[OK] Common lines: {unique_pairs:,} stop pairs, "
        f"{len(out_keys):,} (pair, bucket) entries, "
        f"{len(cl_routes):,} attractive route slots in "
        f"{time.perf_counter() - t0:.2f}s"
    )
    return {
        "cl_keys": cl_keys,
        "cl_offsets": cl_offsets,
        "cl_routes": cl_routes,
        "cl_lambda_prior": cl_lambda,
        "cl_wait_prior": cl_wait,
        "cl_tbar_prior": cl_tbar,
        "cl_vride_prior": cl_vride,
        "cl_ff_ride_prior": cl_ff,
        "cl_canonical_route": cl_canonical,
        "cl_unique_pairs": unique_pairs,
        "cl_entries": len(out_keys),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_transit_graph(path: Path = OUTPUT_PATH) -> TransitGraph:
    """Load a compiled graph from disk."""
    with Path(path).open("rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, TransitGraph):
        raise TypeError(f"{path} does not contain a TransitGraph")
    return graph


def compile_graph() -> TransitGraph:
    started = time.perf_counter()
    print("Compiling Delhi transit graph")
    inputs = load_inputs()
    stop_index = build_stop_index(inputs["stops"])
    route_index = build_route_index(inputs["routes"])
    print(
        f"[OK] Stops: {stop_index['n_stops']:,} | "
        f"Routes: {route_index['n_routes']:,}"
    )

    route_csr = build_route_csr(
        inputs["index"],
        route_index,
        stop_index,
        inputs["route_segments"],
        inputs["corridors"],
    )
    print(
        f"[OK] Route CSR: {len(route_csr['route_stops']):,} stop positions"
    )

    footpaths = build_footpaths(stop_index)

    first_departures = _first_departures(
        inputs["stop_times"], inputs["trips"], route_index
    )
    headways = build_headways(first_departures, route_index)

    try:
        observed = pl.read_parquet(OBSERVED_SEGMENTS_PATH)
    except FileNotFoundError:
        observed = pl.DataFrame()
    cv2 = build_cv2(observed, route_index)
    coverage = build_coverage(first_departures, route_index)

    EH = headways["EH_sec"]
    EW = (0.5 * EH * (1.0 + cv2)).astype(np.float32)

    bucket_sec, ff_sec = corridor_bucket_times(
        inputs["mu_tensor"], inputs["free_flow_mu"]
    )
    common_lines = _build_common_line_entries(
        route_index,
        stop_index,
        route_csr,
        bucket_sec,
        ff_sec,
        EW,
        inputs["index"],
    )

    graph = TransitGraph(
        stop_ids=stop_index["stop_ids"],
        stop_names=stop_index["stop_names"],
        stop_lat=stop_index["stop_lat"],
        stop_lon=stop_index["stop_lon"],
        stop_x_m=stop_index["stop_x_m"],
        stop_y_m=stop_index["stop_y_m"],
        stop_id_to_idx=stop_index["stop_id_to_idx"],
        n_stops=stop_index["n_stops"],
        route_ids=route_index["route_ids"],
        route_short_names=route_index["route_short_names"],
        route_long_names=route_index["route_long_names"],
        route_id_to_idx=route_index["route_id_to_idx"],
        n_routes=route_index["n_routes"],
        n_scheduled_trips=headways["n_scheduled_trips"],
        route_stop_offsets=route_csr["route_stop_offsets"],
        route_stops=route_csr["route_stops"],
        route_stop_dists_m=route_csr["route_stop_dists_m"],
        route_corridor_idxs=route_csr["route_corridor_idxs"],
        footpath_offsets=footpaths["footpath_offsets"],
        footpath_target_stops=footpaths["footpath_target_stops"],
        footpath_walk_sec=footpaths["footpath_walk_sec"],
        footpath_penalty_sec=footpaths["footpath_penalty_sec"],
        footpath_dist_m=footpaths["footpath_dist_m"],
        EH_sec=EH,
        CV2=cv2,
        EW_sec=EW,
        cov_r=coverage,
        cl_keys=common_lines["cl_keys"],
        cl_offsets=common_lines["cl_offsets"],
        cl_routes=common_lines["cl_routes"],
        cl_lambda_prior=common_lines["cl_lambda_prior"],
        cl_wait_prior=common_lines["cl_wait_prior"],
        cl_tbar_prior=common_lines["cl_tbar_prior"],
        cl_vride_prior=common_lines["cl_vride_prior"],
        cl_ff_ride_prior=common_lines["cl_ff_ride_prior"],
        cl_canonical_route=common_lines["cl_canonical_route"],
        corridor_bucket_sec=bucket_sec.astype(np.float32),
        corridor_free_flow_sec=ff_sec.astype(np.float32),
        bucket_names=BUCKET_NAMES,
        built_at=datetime.now(timezone.utc).isoformat(),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    print_summary(graph, footpaths["n_nearby_pairs"], common_lines)
    print(f"[OK] Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[DONE] {time.perf_counter() - started:.2f}s")
    return graph


def print_summary(graph: TransitGraph, nearby_pairs: int, cl: dict[str, Any]) -> None:
    print("\n=== Delhi transit graph ===")
    print(f"Stops                        : {graph.n_stops:,}")
    print(f"Routes                       : {graph.n_routes:,}")
    print(
        f"Footpath nearby pairs        : {nearby_pairs:,} "
        f"({len(graph.footpath_target_stops):,} directed edges)"
    )
    print(f"Common-line stop pairs       : {cl['cl_unique_pairs']:,}")
    print(f"Common-line pair-bucket rows : {cl['cl_entries']:,}")
    print(f"Attractive route slots       : {len(graph.cl_routes):,}")
    print(
        f"EH range (sec)               : "
        f"{graph.EH_sec.min():.0f} .. {graph.EH_sec.max():.0f}"
    )
    print(
        f"CV2 median / fallback        : "
        f"{np.median(graph.CV2):.2f} / {CV2_FALLBACK:.2f}"
    )
    print(
        f"Coverage range               : "
        f"{graph.cov_r.min():.2f} .. {graph.cov_r.max():.2f}"
    )
    print(
        f"Route stop positions         : {len(graph.route_stops):,}"
    )


def main() -> int:
    compile_graph()
    return 0


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    # Re-enter through the canonical module path so pickled dataclasses are
    # recorded as ``src.router.transit_graph.TransitGraph`` rather than
    # ``__main__.TransitGraph``.
    from src.router.transit_graph import main as _canonical_main

    raise SystemExit(_canonical_main())
