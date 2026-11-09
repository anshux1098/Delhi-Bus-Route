"""Google Maps transit ETA client and static GTFS timetable baseline.

The validation benchmark needs an external reference for each origin and
destination pair. When a valid ``GOOGLE_MAPS_API_KEY`` is available this module
queries the Google Maps Routes API. Otherwise it evaluates the published Delhi
GTFS timetable with a connection-scan earliest-arrival search, which mirrors
the scheduled baseline Google Transit falls back to when live TripUpdates are
unavailable.

Public entry point::

    fetch_google_maps_transit_eta(
        origin_lat, origin_lon, dest_lat, dest_lon, departure_epoch
    )
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

# Generic CPU target keeps Numba usable on Windows ARM64 / Snapdragon X.
os.environ.setdefault("NUMBA_CPU_NAME", "generic")

import numpy as np
import polars as pl
from dotenv import load_dotenv
from numba import njit
from scipy.spatial import cKDTree

from src.ingestion.hygiene import find_project_root
from src.matching.shape_indexer import LAT0, LON0, M_PER_DEG_LAT, M_PER_DEG_LON

__all__ = [
    "GtfsTimetable",
    "fetch_google_maps_transit_eta",
    "load_gtfs_timetable",
]

PROJECT_ROOT = find_project_root()
STATIC_DIR = PROJECT_ROOT / "data" / "static"
STOPS_PATH = STATIC_DIR / "stops.txt"
ROUTES_PATH = STATIC_DIR / "routes.txt"
TRIPS_PATH = STATIC_DIR / "trips.txt"
STOP_TIMES_PATH = STATIC_DIR / "stop_times.txt"

load_dotenv(PROJECT_ROOT / ".env", override=False)

GOOGLE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_FIELD_MASK = ",".join(
    [
        "routes.duration",
        "routes.distanceMeters",
        "routes.legs.departureTime",
        "routes.legs.arrivalTime",
        "routes.legs.steps.travelMode",
        "routes.legs.steps.staticDuration",
        "routes.legs.steps.distanceMeters",
        "routes.legs.steps.transitDetails",
        "routes.legs.steps.startLocation",
        "routes.legs.steps.endLocation",
        "routes.legs.steps.navigationInstruction",
    ]
)

IST_TZ = timezone(timedelta(hours=5, minutes=30))
WALK_DETOUR_FACTOR = 1.3
WALK_SPEED_MPS = 1.25
TRANSFER_BUFFER_SEC = 120.0
MAX_ACCESS_RADIUS_M = 1200.0
MAX_ACCESS_STOPS = 8
MAX_JOURNEY_HORIZON_SEC = 6 * 3600.0

SOURCE_GOOGLE = "Google_Maps_Live_API"
SOURCE_STATIC = "Static_GTFS_Schedule_Baseline"

_PLACEHOLDER_KEYS = {
    "",
    "your_key",
    "your_api_key",
    "your-google-maps-api-key",
    "changeme",
    "none",
    "null",
    "test",
}


# ---------------------------------------------------------------------------
# Static GTFS timetable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GtfsTimetable:
    """Compact static GTFS timetable for connection-scan routing."""

    stop_ids: np.ndarray
    stop_names: np.ndarray
    stop_lat: np.ndarray
    stop_lon: np.ndarray
    stop_x_m: np.ndarray
    stop_y_m: np.ndarray
    stop_id_to_idx: dict[str, int]

    conn_dep_stop: np.ndarray
    conn_arr_stop: np.ndarray
    conn_dep_sec: np.ndarray
    conn_arr_sec: np.ndarray
    conn_trip_idx: np.ndarray

    trip_ids: np.ndarray
    trip_route_ids: np.ndarray
    trip_route_short: np.ndarray
    trip_route_long: np.ndarray
    trip_headsigns: np.ndarray

    tree: cKDTree


_TIMETABLE_LOCK = threading.Lock()
_TIMETABLE: GtfsTimetable | None = None


def _time_to_seconds_expr(column: str) -> pl.Expr:
    parts = pl.col(column).str.split(":")
    return (
        parts.list.get(0).cast(pl.Int64) * 3600
        + parts.list.get(1).cast(pl.Int64) * 60
        + parts.list.get(2).cast(pl.Int64)
    )


def _numeric_sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (10**18, text)


def _build_timetable() -> GtfsTimetable:
    stops = (
        pl.read_csv(
            STOPS_PATH,
            schema_overrides={"stop_id": pl.Utf8, "stop_name": pl.Utf8},
        )
        .with_columns(
            pl.col("stop_id").cast(pl.Utf8),
            pl.col("stop_name").cast(pl.Utf8),
            pl.col("stop_lat").cast(pl.Float64),
            pl.col("stop_lon").cast(pl.Float64),
        )
        .unique(subset=["stop_id"], keep="first")
    )
    ordered_stop_ids = sorted(stops["stop_id"].to_list(), key=_numeric_sort_key)
    stop_to_idx = {stop_id: i for i, stop_id in enumerate(ordered_stop_ids)}
    stops = stops.with_columns(
        pl.col("stop_id")
        .replace_strict(stop_to_idx, return_dtype=pl.Int32)
        .alias("stop_idx")
    ).sort("stop_idx")

    stop_ids = np.asarray(stops["stop_id"].to_list(), dtype=object)
    stop_names = np.asarray(stops["stop_name"].to_list(), dtype=object)
    stop_lat = stops["stop_lat"].to_numpy().astype(np.float64)
    stop_lon = stops["stop_lon"].to_numpy().astype(np.float64)
    stop_x = (stop_lon - LON0) * M_PER_DEG_LON
    stop_y = (stop_lat - LAT0) * M_PER_DEG_LAT

    routes = (
        pl.read_csv(
            ROUTES_PATH,
            schema_overrides={
                "route_id": pl.Utf8,
                "route_short_name": pl.Utf8,
                "route_long_name": pl.Utf8,
            },
        )
        .with_columns(
            pl.col("route_id").cast(pl.Utf8),
            pl.col("route_short_name").cast(pl.Utf8),
            pl.col("route_long_name").cast(pl.Utf8),
        )
        .unique(subset=["route_id"], keep="first")
    )

    trips = (
        pl.read_csv(
            TRIPS_PATH,
            schema_overrides={
                "trip_id": pl.Utf8,
                "route_id": pl.Utf8,
                "trip_headsign": pl.Utf8,
                "service_id": pl.Utf8,
            },
            infer_schema_length=50_000,
        )
        .with_columns(
            pl.col("trip_id").cast(pl.Utf8),
            pl.col("route_id").cast(pl.Utf8),
            pl.col("trip_headsign").cast(pl.Utf8),
        )
        .unique(subset=["trip_id"], keep="first")
        .sort("trip_id")
    )

    trip_ids = np.asarray(trips["trip_id"].to_list(), dtype=object)
    trip_to_idx = {trip_id: i for i, trip_id in enumerate(trip_ids)}

    trip_meta = trips.join(
        routes.select(
            "route_id",
            pl.col("route_short_name").alias("short_name"),
            pl.col("route_long_name").alias("long_name"),
        ),
        on="route_id",
        how="left",
    ).sort("trip_id")
    trip_route_ids = np.asarray(trip_meta["route_id"].to_list(), dtype=object)
    trip_route_short = np.asarray(
        [x or "" for x in trip_meta["short_name"].to_list()], dtype=object
    )
    trip_route_long = np.asarray(
        [x or "" for x in trip_meta["long_name"].to_list()], dtype=object
    )
    trip_headsigns = np.asarray(
        [x or "" for x in trip_meta["trip_headsign"].to_list()], dtype=object
    )

    stop_times = (
        pl.read_csv(
            STOP_TIMES_PATH,
            schema_overrides={"trip_id": pl.Utf8, "stop_id": pl.Utf8},
            infer_schema_length=50_000,
        )
        .with_columns(
            pl.col("trip_id").cast(pl.Utf8),
            pl.col("stop_id").cast(pl.Utf8),
            pl.col("stop_sequence").cast(pl.Int32),
        )
        .with_columns(
            _time_to_seconds_expr("arrival_time").alias("arr_sec"),
            _time_to_seconds_expr("departure_time").alias("dep_sec"),
        )
        .filter(
            pl.col("arr_sec").is_not_null()
            & pl.col("dep_sec").is_not_null()
            & (pl.col("arr_sec") >= 0)
            & (pl.col("dep_sec") >= 0)
        )
        .with_columns(
            pl.col("trip_id")
            .replace_strict(trip_to_idx, return_dtype=pl.Int32)
            .alias("trip_idx"),
            pl.col("stop_id")
            .replace_strict(stop_to_idx, default=-1, return_dtype=pl.Int32)
            .alias("stop_idx"),
        )
        .filter(pl.col("stop_idx") >= 0)
        .sort(["trip_idx", "stop_sequence"])
    )

    stop_times = stop_times.with_columns(
        pl.col("stop_idx").shift(-1).over("trip_idx").alias("next_stop_idx"),
        pl.col("arr_sec").shift(-1).over("trip_idx").alias("next_arr_sec"),
    ).filter(
        pl.col("next_stop_idx").is_not_null()
        & pl.col("next_arr_sec").is_not_null()
        & (pl.col("next_arr_sec") >= pl.col("dep_sec"))
    )

    connections = (
        stop_times.select(
            pl.col("stop_idx").cast(pl.Int32).alias("dep_stop"),
            pl.col("next_stop_idx").cast(pl.Int32).alias("arr_stop"),
            pl.col("dep_sec").cast(pl.Float64).alias("dep_sec"),
            pl.col("next_arr_sec").cast(pl.Float64).alias("arr_sec"),
            pl.col("trip_idx").cast(pl.Int32).alias("trip_idx"),
        )
        .sort("dep_sec")
    )

    conn_dep_stop = connections["dep_stop"].to_numpy().astype(np.int32, copy=False)
    conn_arr_stop = connections["arr_stop"].to_numpy().astype(np.int32, copy=False)
    conn_dep_sec = connections["dep_sec"].to_numpy().astype(np.float64, copy=False)
    conn_arr_sec = connections["arr_sec"].to_numpy().astype(np.float64, copy=False)
    conn_trip = connections["trip_idx"].to_numpy().astype(np.int32, copy=False)

    tree = cKDTree(np.column_stack([stop_x, stop_y]))

    return GtfsTimetable(
        stop_ids=stop_ids,
        stop_names=stop_names,
        stop_lat=stop_lat,
        stop_lon=stop_lon,
        stop_x_m=stop_x,
        stop_y_m=stop_y,
        stop_id_to_idx=stop_to_idx,
        conn_dep_stop=conn_dep_stop,
        conn_arr_stop=conn_arr_stop,
        conn_dep_sec=conn_dep_sec,
        conn_arr_sec=conn_arr_sec,
        conn_trip_idx=conn_trip,
        trip_ids=trip_ids,
        trip_route_ids=trip_route_ids,
        trip_route_short=trip_route_short,
        trip_route_long=trip_route_long,
        trip_headsigns=trip_headsigns,
        tree=tree,
    )


def load_gtfs_timetable() -> GtfsTimetable:
    """Load and cache the static GTFS timetable."""
    global _TIMETABLE
    if _TIMETABLE is None:
        with _TIMETABLE_LOCK:
            if _TIMETABLE is None:
                _TIMETABLE = _build_timetable()
    return _TIMETABLE


# ---------------------------------------------------------------------------
# Connection-scan earliest arrival
# ---------------------------------------------------------------------------


@njit(cache=True)
def _csa_scan(
    dep_stop: np.ndarray,
    arr_stop: np.ndarray,
    dep_sec: np.ndarray,
    arr_sec: np.ndarray,
    trip_idx: np.ndarray,
    start_idx: int,
    max_time: float,
    arrival: np.ndarray,
    arrival_trip: np.ndarray,
    transfer_sec: float,
) -> np.ndarray:
    """Scan departure-sorted connections and return predecessor links."""
    n_connections = dep_sec.shape[0]
    n_stops = arrival.shape[0]
    pred = np.full(n_stops, -1, dtype=np.int32)

    for c in range(start_idx, n_connections):
        if dep_sec[c] > max_time:
            break

        u = dep_stop[c]
        if arrival[u] > dep_sec[c]:
            continue

        # Enforce a transfer buffer only when changing trips.
        if (
            arrival_trip[u] != -1
            and arrival_trip[u] != trip_idx[c]
            and arrival[u] + transfer_sec > dep_sec[c]
        ):
            continue

        v = arr_stop[c]
        t = arr_sec[c]
        if t < arrival[v]:
            arrival[v] = t
            arrival_trip[v] = trip_idx[c]
            pred[v] = c

    return pred


def _nearby_stops(
    timetable: GtfsTimetable,
    lat: float,
    lon: float,
    radius_m: float = MAX_ACCESS_RADIUS_M,
    limit: int = MAX_ACCESS_STOPS,
) -> list[tuple[int, float, float]]:
    """Return (stop_idx, walk_sec, walk_dist_m) sorted by walking time."""
    x = (float(lon) - LON0) * M_PER_DEG_LON
    y = (float(lat) - LAT0) * M_PER_DEG_LAT

    indices = timetable.tree.query_ball_point([x, y], r=float(radius_m))
    if not indices:
        _, nearest = timetable.tree.query([x, y], k=1)
        indices = [int(nearest)]

    candidates: list[tuple[int, float, float]] = []
    for idx in indices:
        idx = int(idx)
        euclid = math.hypot(x - float(timetable.stop_x_m[idx]), y - float(timetable.stop_y_m[idx]))
        walk_dist = WALK_DETOUR_FACTOR * euclid
        walk_sec = walk_dist / WALK_SPEED_MPS
        candidates.append((idx, walk_sec, walk_dist))

    candidates.sort(key=lambda item: item[1])
    return candidates[:limit]


def _format_ist(service_midnight: datetime, seconds_of_day: float) -> str:
    moment = service_midnight + timedelta(seconds=float(seconds_of_day))
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _reconstruct_legs(
    timetable: GtfsTimetable,
    pred: np.ndarray,
    final_stop: int,
) -> list[dict[str, Any]]:
    """Turn predecessor links into chronological bus ride segments."""
    rides: list[dict[str, Any]] = []
    current_stop = int(final_stop)

    while True:
        connection = int(pred[current_stop])
        if connection < 0:
            break

        trip = int(timetable.conn_trip_idx[connection])
        chain = [connection]
        cursor = connection
        board_stop = int(timetable.conn_dep_stop[cursor])

        while True:
            previous = int(pred[board_stop])
            if (
                previous >= 0
                and int(timetable.conn_trip_idx[previous]) == trip
                and float(timetable.conn_arr_sec[previous])
                <= float(timetable.conn_dep_sec[cursor])
            ):
                chain.append(previous)
                cursor = previous
                board_stop = int(timetable.conn_dep_stop[cursor])
            else:
                break

        if board_stop == current_stop:
            break

        rides.append(
            {
                "trip_idx": trip,
                "board_stop": board_stop,
                "alight_stop": current_stop,
                "dep_sec": float(timetable.conn_dep_sec[chain[-1]]),
                "arr_sec": float(timetable.conn_arr_sec[chain[0]]),
                "stops_count": len(chain),
            }
        )
        current_stop = board_stop

    rides.reverse()
    return rides


def _walk_leg(
    timetable: GtfsTimetable,
    from_stop: int | None,
    to_stop: int | None,
    from_name: str,
    from_lat: float,
    from_lon: float,
    to_name: str,
    to_lat: float,
    to_lon: float,
    walk_sec: float,
    walk_dist_m: float,
    departure_sec: float,
    service_midnight: datetime,
) -> dict[str, Any]:
    return {
        "mode": "WALK",
        "source": SOURCE_STATIC,
        "from_stop_id": None if from_stop is None else str(timetable.stop_ids[from_stop]),
        "from_stop_name": from_name,
        "from_lat": float(from_lat),
        "from_lon": float(from_lon),
        "to_stop_id": None if to_stop is None else str(timetable.stop_ids[to_stop]),
        "to_stop_name": to_name,
        "to_lat": float(to_lat),
        "to_lon": float(to_lon),
        "departure_time_ist": _format_ist(service_midnight, departure_sec),
        "arrival_time_ist": _format_ist(service_midnight, departure_sec + walk_sec),
        "duration_sec": round(float(walk_sec), 1),
        "distance_m": round(float(walk_dist_m), 1),
        "stops_count": 0,
    }


def _static_gtfs_baseline(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    departure_epoch: float,
) -> dict[str, Any] | None:
    timetable = load_gtfs_timetable()

    departure_dt = datetime.fromtimestamp(float(departure_epoch), tz=IST_TZ)
    query_midnight = departure_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # GTFS encodes after-midnight service as 24:xx-28:xx. For early-morning
    # queries, evaluate against the previous service day so those trips are
    # still visible to the scan.
    if departure_dt.hour < 4:
        service_midnight = query_midnight - timedelta(days=1)
    else:
        service_midnight = query_midnight
    start_sec = (departure_dt - service_midnight).total_seconds()

    origin_candidates = _nearby_stops(timetable, origin_lat, origin_lon)
    dest_candidates = _nearby_stops(timetable, dest_lat, dest_lon)
    if not origin_candidates or not dest_candidates:
        return None

    origin_access = {idx: (walk_sec, walk_dist) for idx, walk_sec, walk_dist in origin_candidates}

    n_stops = len(timetable.stop_ids)
    arrival = np.full(n_stops, np.inf, dtype=np.float64)
    arrival_trip = np.full(n_stops, -1, dtype=np.int32)

    for idx, walk_sec, _ in origin_candidates:
        access_time = start_sec + walk_sec
        if access_time < arrival[idx]:
            arrival[idx] = access_time
            arrival_trip[idx] = -1

    max_time = start_sec + MAX_JOURNEY_HORIZON_SEC
    start_idx = int(np.searchsorted(timetable.conn_dep_sec, start_sec, side="left"))

    pred = _csa_scan(
        timetable.conn_dep_stop,
        timetable.conn_arr_stop,
        timetable.conn_dep_sec,
        timetable.conn_arr_sec,
        timetable.conn_trip_idx,
        start_idx,
        max_time,
        arrival,
        arrival_trip,
        TRANSFER_BUFFER_SEC,
    )

    best: tuple[float, int, float, float] | None = None
    for idx, walk_sec, walk_dist in dest_candidates:
        if np.isfinite(arrival[idx]):
            total = float(arrival[idx]) + walk_sec
            if best is None or total < best[0]:
                best = (total, idx, walk_sec, walk_dist)

    if best is None:
        return None

    total_arrival_sec, final_stop, final_walk_sec, final_walk_dist = best
    final_arrival_sec = float(arrival[final_stop])

    rides = _reconstruct_legs(timetable, pred, final_stop)
    if not rides:
        # The earliest "journey" is walking from the origin to a nearby
        # destination stop (or straight to the destination) before service
        # starts. Return an explicit walk-only baseline instead of None.
        x0 = (float(origin_lon) - LON0) * M_PER_DEG_LON
        y0 = (float(origin_lat) - LAT0) * M_PER_DEG_LAT
        x1 = (float(dest_lon) - LON0) * M_PER_DEG_LON
        y1 = (float(dest_lat) - LAT0) * M_PER_DEG_LAT
        walk_dist = WALK_DETOUR_FACTOR * math.hypot(x1 - x0, y1 - y0)
        walk_sec = walk_dist / WALK_SPEED_MPS
        walk_leg = _walk_leg(
            timetable,
            None,
            None,
            "Origin",
            origin_lat,
            origin_lon,
            "Destination",
            dest_lat,
            dest_lon,
            walk_sec,
            walk_dist,
            start_sec,
            service_midnight,
        )
        return {
            "source": SOURCE_STATIC,
            "provider": "static_gtfs_connection_scan",
            "query_departure_epoch": float(departure_epoch),
            "departure_time_ist": departure_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "arrival_time_ist": _format_ist(service_midnight, start_sec + walk_sec),
            "duration_sec": round(walk_sec, 1),
            "distance_m": round(walk_dist, 1),
            "num_transfers": 0,
            "legs": [walk_leg],
            "transit_legs": [],
            "note": (
                "Walking is faster than the first available static GTFS trip; "
                "nominal baseline uses the walk-only itinerary."
            ),
        }

    first_board_stop = int(rides[0]["board_stop"])
    first_walk_sec, first_walk_dist = origin_access[first_board_stop]

    legs: list[dict[str, Any]] = []
    if first_walk_sec > 0.5:
        legs.append(
            _walk_leg(
                timetable,
                None,
                first_board_stop,
                "Origin",
                origin_lat,
                origin_lon,
                str(timetable.stop_names[first_board_stop]),
                float(timetable.stop_lat[first_board_stop]),
                float(timetable.stop_lon[first_board_stop]),
                first_walk_sec,
                first_walk_dist,
                start_sec,
                service_midnight,
            )
        )

    for ride in rides:
        trip = int(ride["trip_idx"])
        board_stop = int(ride["board_stop"])
        alight_stop = int(ride["alight_stop"])
        legs.append(
            {
                "mode": "BUS",
                "source": SOURCE_STATIC,
                "route_id": str(timetable.trip_route_ids[trip]),
                "route_short_name": str(timetable.trip_route_short[trip]),
                "route_long_name": str(timetable.trip_route_long[trip]),
                "trip_id": str(timetable.trip_ids[trip]),
                "headsign": str(timetable.trip_headsigns[trip]),
                "from_stop_id": str(timetable.stop_ids[board_stop]),
                "from_stop_name": str(timetable.stop_names[board_stop]),
                "from_lat": float(timetable.stop_lat[board_stop]),
                "from_lon": float(timetable.stop_lon[board_stop]),
                "to_stop_id": str(timetable.stop_ids[alight_stop]),
                "to_stop_name": str(timetable.stop_names[alight_stop]),
                "to_lat": float(timetable.stop_lat[alight_stop]),
                "to_lon": float(timetable.stop_lon[alight_stop]),
                "departure_time_ist": _format_ist(service_midnight, ride["dep_sec"]),
                "arrival_time_ist": _format_ist(service_midnight, ride["arr_sec"]),
                "duration_sec": round(float(ride["arr_sec"] - ride["dep_sec"]), 1),
                "stops_count": int(ride["stops_count"]),
            }
        )

    if final_walk_sec > 0.5:
        legs.append(
            _walk_leg(
                timetable,
                final_stop,
                None,
                str(timetable.stop_names[final_stop]),
                float(timetable.stop_lat[final_stop]),
                float(timetable.stop_lon[final_stop]),
                "Destination",
                dest_lat,
                dest_lon,
                final_walk_sec,
                final_walk_dist,
                final_arrival_sec,
                service_midnight,
            )
        )

    duration_sec = max(0.0, total_arrival_sec - start_sec)
    transit_legs = [leg for leg in legs if leg["mode"] == "BUS"]

    return {
        "source": SOURCE_STATIC,
        "provider": "static_gtfs_connection_scan",
        "query_departure_epoch": float(departure_epoch),
        "departure_time_ist": departure_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "arrival_time_ist": _format_ist(service_midnight, total_arrival_sec),
        "duration_sec": round(duration_sec, 1),
        "distance_m": None,
        "num_transfers": max(0, len(transit_legs) - 1),
        "legs": legs,
        "transit_legs": transit_legs,
        "note": (
            "Nominal Google Transit baseline computed from the published static "
            "GTFS schedule; live TripUpdates are absent."
        ),
    }


# ---------------------------------------------------------------------------
# Google Routes API
# ---------------------------------------------------------------------------


def _api_key() -> str | None:
    key = (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip().strip('"').strip("'")
    if key.lower() in _PLACEHOLDER_KEYS:
        return None
    if len(key) < 20:
        return None
    return key


def _epoch_to_rfc3339(epoch: float) -> str:
    return (
        datetime.fromtimestamp(float(epoch), tz=timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _parse_duration(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("s"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _iso_to_ist(value: Any) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(IST_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(value)


def _lat_lng(location: Any) -> tuple[float, float]:
    payload = ((location or {}).get("latLng") or {}) if isinstance(location, dict) else {}
    return float(payload.get("latitude", 0.0)), float(payload.get("longitude", 0.0))


def _google_legs(route: dict[str, Any]) -> list[dict[str, Any]]:
    legs: list[dict[str, Any]] = []

    for route_leg in route.get("legs", []) or []:
        for step in route_leg.get("steps", []) or []:
            travel_mode = str(step.get("travelMode", "UNKNOWN"))
            details = step.get("transitDetails")
            step_duration = _parse_duration(step.get("staticDuration"))

            if details:
                stop_details = details.get("stopDetails", {}) or {}
                departure_stop = stop_details.get("departureStop", {}) or {}
                arrival_stop = stop_details.get("arrivalStop", {}) or {}
                line = details.get("transitLine", {}) or {}
                vehicle = line.get("vehicle", {}) or {}

                from_lat, from_lon = _lat_lng(departure_stop.get("location"))
                to_lat, to_lon = _lat_lng(arrival_stop.get("location"))
                legs.append(
                    {
                        "mode": "BUS",
                        "source": SOURCE_GOOGLE,
                        "route_id": line.get("nameShort") or line.get("name") or "",
                        "route_short_name": line.get("nameShort") or "",
                        "route_long_name": line.get("name") or "",
                        "trip_id": None,
                        "headsign": details.get("headsign") or "",
                        "vehicle_type": vehicle.get("type") or "",
                        "from_stop_id": None,
                        "from_stop_name": departure_stop.get("name") or "",
                        "from_lat": from_lat,
                        "from_lon": from_lon,
                        "to_stop_id": None,
                        "to_stop_name": arrival_stop.get("name") or "",
                        "to_lat": to_lat,
                        "to_lon": to_lon,
                        "departure_time_ist": _iso_to_ist(stop_details.get("departureTime")),
                        "arrival_time_ist": _iso_to_ist(stop_details.get("arrivalTime")),
                        "duration_sec": step_duration,
                        "distance_m": step.get("distanceMeters"),
                        "stops_count": int(details.get("stopCount") or 0),
                    }
                )
            elif travel_mode == "WALK":
                from_lat, from_lon = _lat_lng(step.get("startLocation"))
                to_lat, to_lon = _lat_lng(step.get("endLocation"))
                legs.append(
                    {
                        "mode": "WALK",
                        "source": SOURCE_GOOGLE,
                        "from_stop_id": None,
                        "from_stop_name": (step.get("navigationInstruction") or {}).get(
                            "instructions", "Walk"
                        ),
                        "from_lat": from_lat,
                        "from_lon": from_lon,
                        "to_stop_id": None,
                        "to_stop_name": "Walk destination",
                        "to_lat": to_lat,
                        "to_lon": to_lon,
                        "departure_time_ist": "",
                        "arrival_time_ist": "",
                        "duration_sec": step_duration,
                        "distance_m": step.get("distanceMeters"),
                        "stops_count": 0,
                    }
                )

    return legs


def _fetch_google_routes(
    key: str,
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    departure_epoch: float,
) -> dict[str, Any] | None:
    import httpx

    payload = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": float(origin_lat),
                    "longitude": float(origin_lon),
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": float(dest_lat),
                    "longitude": float(dest_lon),
                }
            }
        },
        "travelMode": "TRANSIT",
        "departureTime": _epoch_to_rfc3339(departure_epoch),
        "computeAlternativeRoutes": False,
        "languageCode": "en-IN",
        "regionCode": "IN",
        "transitPreferences": {"routingPreference": "FEWER_TRANSFERS"},
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": GOOGLE_FIELD_MASK,
    }

    response = httpx.post(GOOGLE_ROUTES_URL, json=payload, headers=headers, timeout=15.0)
    response.raise_for_status()
    data = response.json()
    routes = data.get("routes") or []
    if not routes:
        return None

    route = routes[0]
    duration_sec = _parse_duration(route.get("duration"))
    legs = _google_legs(route)
    transit_legs = [leg for leg in legs if leg["mode"] == "BUS"]

    route_legs = route.get("legs") or [{}]
    departure_ist = _iso_to_ist(route_legs[0].get("departureTime"))
    arrival_ist = _iso_to_ist(route_legs[-1].get("arrivalTime"))

    return {
        "source": SOURCE_GOOGLE,
        "provider": "google_routes_api",
        "departure_time_ist": departure_ist
        or datetime.fromtimestamp(float(departure_epoch), tz=IST_TZ).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "arrival_time_ist": arrival_ist,
        "duration_sec": round(duration_sec, 1) if duration_sec is not None else None,
        "distance_m": route.get("distanceMeters"),
        "num_transfers": max(0, len(transit_legs) - 1),
        "legs": legs,
        "transit_legs": transit_legs,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_google_maps_transit_eta(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    departure_epoch: float,
) -> dict[str, Any] | None:
    """Fetch a transit ETA from Google Maps or the static GTFS baseline.

    Args:
        origin_lat/origin_lon: WGS84 origin.
        dest_lat/dest_lon: WGS84 destination.
        departure_epoch: Unix departure timestamp.

    Returns:
        A dictionary with ``source``, ``duration_sec``, ``legs`` and
        ``transit_legs``. ``source`` is ``"Google_Maps_Live_API"`` when the
        Routes API succeeded, otherwise ``"Static_GTFS_Schedule_Baseline"``.
        Returns ``None`` only when the static timetable finds no journey.
    """
    key = _api_key()
    api_error: str | None = None

    if not key:
        raw_key = (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip()
        if raw_key:
            api_error = "GOOGLE_MAPS_API_KEY is present but failed format validation"

    if key:
        try:
            result = _fetch_google_routes(
                key,
                origin_lat,
                origin_lon,
                dest_lat,
                dest_lon,
                departure_epoch,
            )
            if result is not None:
                return result
            api_error = "Google Routes API returned no transit route"
        except Exception as exc:  # network, quota, auth, malformed response
            api_error = f"{type(exc).__name__}: {exc}"

    baseline = _static_gtfs_baseline(
        origin_lat,
        origin_lon,
        dest_lat,
        dest_lon,
        departure_epoch,
    )
    if baseline is not None and api_error:
        baseline["google_api_error"] = api_error
    return baseline
