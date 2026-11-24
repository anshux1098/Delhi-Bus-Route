"""Delhi GTFS-Realtime Ingestion Setup Check.

Verifies end-to-end setup:
1. Loads .env and fetches a snapshot of VehiclePositions.pb from TRANSPORT_STACK_API_URL
   (automatic fallback to OTD_RT_URL if needed).
2. Parses protobuf bytes with google.transit.gtfs_realtime_pb2.FeedMessage.
3. Computes ist_time with fixed UTC+05:30 offset (Windows tz safe).
4. Extracts all vehicle entities into a Polars DataFrame with columns:
   [vehicle_id, route_id, trip_id, latitude, longitude, bearing, speed, timestamp, ist_time]
5. Saves snapshot to data/historical/live_batch_001.parquet.
6. Prints first 5 rows of live vehicles DataFrame with pl.Config.set_tbl_cols(10).
7. Reads data/static/routes.txt and data/static/trips.txt with Polars and prints first 5 rows.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_project_root() -> Path:
    """Locate the project root directory containing data and .env."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "data").exists():
        return candidate
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    if (cwd / "delhi-transit-planner" / "data").exists():
        return cwd / "delhi-transit-planner"
    return candidate


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = DATA_DIR / "static"
HISTORICAL_DIR = DATA_DIR / "historical"
ENV_PATH = PROJECT_ROOT / ".env"

# Windows-safe fixed timezone offset (+05:30)
IST = timezone(timedelta(hours=5, minutes=30))

VEHICLE_SCHEMA: dict[str, pl.DataType] = {
    "vehicle_id": pl.Utf8,
    "route_id": pl.Utf8,
    "trip_id": pl.Utf8,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "bearing": pl.Float32,
    "speed": pl.Float32,
    "timestamp": pl.Int64,
    "ist_time": pl.Utf8,
}


def compute_ist_time(ts: int | None) -> str | None:
    """Format Unix timestamp as IST string without external timezone DB."""
    if ts is None or ts <= 0:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def fetch_gtfs_rt_snapshot() -> bytes:
    """Fetch VehiclePositions.pb from TRANSPORT_STACK_API_URL (primary) or fallback to OTD_RT_URL."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)
    else:
        load_dotenv(override=True)

    otd_url = os.getenv("OTD_RT_URL", "").strip()
    stack_url = os.getenv("TRANSPORT_STACK_API_URL", "").strip()
    stack_key = os.getenv("TRANSPORT_STACK_API_KEY", "").strip()

    headers = {"User-Agent": "delhi-transit-planner/0.1 (setup-check)"}

    has_valid_stack = (
        bool(stack_url)
        and "PASTE_" not in stack_url
        and bool(stack_key)
        and "PASTE_" not in stack_key
        and "your_" not in stack_key.lower()
    )

    # 1. Try primary TRANSPORT_STACK_API_URL then fallback OTD_RT_URL using one pooled client
    with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
        if has_valid_stack:
            print(f"Fetching live snapshot from PRIMARY (TRANSPORT_STACK_API_URL): {stack_url}")
            try:
                response = client.get(stack_url, headers={"x-api-key": stack_key})
                response.raise_for_status()
                if len(response.content) > 1000 and not response.content.startswith(b"<!doctype"):
                    print(f"[OK] [PRIMARY: Transport Stack] Received {len(response.content):,} bytes.")
                    return response.content
                print(
                    f"[WARN] Transport Stack response appears non-protobuf ({len(response.content)} bytes). "
                    "Trying fallback..."
                )
            except Exception as exc:
                print(f"[WARN] Primary TRANSPORT_STACK_API_URL fetch failed: {exc}. Trying fallback...")

        if otd_url and "PASTE_" not in otd_url:
            masked_url = otd_url.split("key=")[0] + "key=***" if "key=" in otd_url else otd_url
            print(f"Fetching live snapshot from FALLBACK (OTD_RT_URL): {masked_url}")
            try:
                response = client.get(otd_url)
                response.raise_for_status()
                if len(response.content) > 1000 and not response.content.startswith(b"<!doctype"):
                    print(f"[OK] [FALLBACK: OTD_RT_URL] Received {len(response.content):,} bytes.")
                    return response.content
                print(f"[WARN] OTD response appears non-protobuf ({len(response.content)} bytes).")
            except Exception as exc:
                print(f"[WARN] Fallback OTD_RT_URL fetch failed: {exc}")

    raise RuntimeError("Failed to fetch valid GTFS-Realtime protobuf feed from configured URLs.")


def parse_vehicle_positions(payload: bytes) -> tuple[int, pl.DataFrame]:
    """Parse FeedMessage bytes into Polars DataFrame matching schema."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(payload)

    feed_ts = int(feed.header.timestamp) if feed.header.HasField("timestamp") else 0
    default_ts = feed_ts if feed_ts > 0 else 0
    ts_cache: dict[int, str | None] = {0: None}
    isnan_or_inf = math.isfinite

    vehicle_ids: list[str | None] = []
    route_ids: list[str | None] = []
    trip_ids: list[str | None] = []
    latitudes: list[float | None] = []
    longitudes: list[float | None] = []
    bearings: list[float | None] = []
    speeds: list[float | None] = []
    timestamps: list[int] = []
    ist_times: list[str | None] = []

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        vehicle = entity.vehicle
        if vehicle.HasField("position"):
            pos = vehicle.position
            lat = float(pos.latitude) if (pos.HasField("latitude") and isnan_or_inf(pos.latitude)) else None
            lon = float(pos.longitude) if (pos.HasField("longitude") and isnan_or_inf(pos.longitude)) else None
            bearing = float(pos.bearing) if (pos.HasField("bearing") and isnan_or_inf(pos.bearing)) else None
            speed = float(pos.speed) if (pos.HasField("speed") and isnan_or_inf(pos.speed)) else None
        else:
            lat = lon = bearing = speed = None

        vid = vehicle.vehicle.id or vehicle.vehicle.label or entity.id or None
        has_trip = vehicle.HasField("trip")
        route_id = vehicle.trip.route_id if (has_trip and vehicle.trip.route_id) else None
        trip_id = vehicle.trip.trip_id if (has_trip and vehicle.trip.trip_id) else None

        ts = int(vehicle.timestamp) if vehicle.timestamp > 0 else default_ts
        ist_time = ts_cache.get(ts)
        if ist_time is None and ts > 0:
            ist_time = compute_ist_time(ts)
            ts_cache[ts] = ist_time

        vehicle_ids.append(str(vid) if vid is not None else None)
        route_ids.append(str(route_id) if route_id is not None else None)
        trip_ids.append(str(trip_id) if trip_id is not None else None)
        latitudes.append(lat)
        longitudes.append(lon)
        bearings.append(bearing)
        speeds.append(speed)
        timestamps.append(ts)
        ist_times.append(ist_time)

    df = pl.DataFrame(
        {
            "vehicle_id": vehicle_ids,
            "route_id": route_ids,
            "trip_id": trip_ids,
            "latitude": latitudes,
            "longitude": longitudes,
            "bearing": bearings,
            "speed": speeds,
            "timestamp": timestamps,
            "ist_time": ist_times,
        },
        schema=VEHICLE_SCHEMA,
    )
    return feed_ts, df


def main() -> int:
    print("=================================================================")
    print("Delhi GTFS-Realtime Setup Check")
    print("=================================================================")
    print(f"Project root: {PROJECT_ROOT}")

    # 1. Fetch live snapshot
    try:
        payload = fetch_gtfs_rt_snapshot()
    except Exception as exc:
        print(f"[FAIL] Error fetching live feed snapshot: {exc}")
        return 1

    # 2. Parse protobuf into Polars DataFrame
    try:
        feed_ts, df = parse_vehicle_positions(payload)
        feed_ist = compute_ist_time(feed_ts)
        print(f"[OK] Protobuf parsed successfully. Feed header timestamp: {feed_ts} ({feed_ist})")
        print(f"[OK] Total vehicles extracted: {df.height:,}")
    except Exception as exc:
        print(f"[FAIL] Error parsing protobuf feed: {exc}")
        return 1

    # 3. Save snapshot to data/historical/live_batch_001.parquet
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_file = HISTORICAL_DIR / "live_batch_001.parquet"
    try:
        df.write_parquet(snapshot_file, compression="zstd")
        file_size_kb = snapshot_file.stat().st_size / 1024.0
        print(f"[OK] Saved snapshot to {snapshot_file.relative_to(PROJECT_ROOT)} ({file_size_kb:.1f} KB)")
    except Exception as exc:
        print(f"[FAIL] Error saving parquet snapshot: {exc}")
        return 1

    # 4. Print first 5 rows of live vehicles DataFrame with pl.Config.set_tbl_cols(10)
    pl.Config.set_tbl_cols(10)
    print("\n-----------------------------------------------------------------")
    print("Live Vehicles Snapshot (First 5 rows):")
    print("-----------------------------------------------------------------")
    print(df.head(5))

    # 5. Read data/static/routes.txt and data/static/trips.txt with Polars and print first 5 rows
    routes_file = STATIC_DIR / "routes.txt"
    trips_file = STATIC_DIR / "trips.txt"

    print("\n-----------------------------------------------------------------")
    print(f"Static Routes (First 5 rows from {routes_file.name}):")
    print("-----------------------------------------------------------------")
    if routes_file.exists():
        routes_df = pl.read_csv(routes_file, n_rows=5)
        print(routes_df)
    else:
        print(f"[WARN] File not found: {routes_file}")

    print("\n-----------------------------------------------------------------")
    print(f"Static Trips (First 5 rows from {trips_file.name}):")
    print("-----------------------------------------------------------------")
    if trips_file.exists():
        trips_df = pl.read_csv(trips_file, n_rows=5)
        print(trips_df)
    else:
        print(f"[WARN] File not found: {trips_file}")

    print("\n=================================================================")
    print("[SUCCESS] Setup verification complete.")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
