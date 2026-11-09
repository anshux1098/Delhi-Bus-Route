"""Delhi GTFS-Realtime Live Collector.

Polls OTD_RT_URL (or fallback TRANSPORT_STACK_API_URL) periodically:
1. Atomically updates data/live_buffer/latest_vehicles.parquet on every poll.
2. Tracks last_seen_ts per vehicle_id in memory to buffer only genuinely new GPS updates.
3. Every 30 polls (~5 minutes), flushes accumulated unique pings to
   data/historical/pings_YYYYMMDD_HHMMSS.parquet and clears the buffer.
4. Resilient to transient timeouts or network drops.
5. Supports --once flag for single-iteration test runs.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
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
LIVE_BUFFER_DIR = DATA_DIR / "live_buffer"
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

COL_ORDER = [
    "vehicle_id",
    "route_id",
    "trip_id",
    "latitude",
    "longitude",
    "bearing",
    "speed",
    "timestamp",
    "ist_time",
]


def compute_ist_time(ts: int | None) -> str | None:
    """Format Unix timestamp as IST string without external timezone DB."""
    if ts is None or ts <= 0:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def fetch_gtfs_rt(client: httpx.Client, otd_url: str, stack_url: str, stack_key: str) -> bytes:
    """Fetch protobuf payload from primary or fallback feed URL."""
    headers = {"User-Agent": "delhi-transit-planner/0.1 (live-collector)"}

    # 1. Primary: OTD_RT_URL
    if otd_url and "PASTE_" not in otd_url:
        try:
            resp = client.get(otd_url, headers=headers)
            resp.raise_for_status()
            if len(resp.content) > 1000 and not resp.content.startswith(b"<!doctype"):
                return resp.content
        except Exception as exc:
            print(f"[WARN] Primary OTD poll error: {exc}")

    # 2. Fallback: TRANSPORT_STACK_API_URL
    if stack_url and "PASTE_" not in stack_url:
        stack_headers = dict(headers)
        if stack_key and "PASTE_" not in stack_key:
            stack_headers["x-api-key"] = stack_key
        try:
            resp = client.get(stack_url, headers=stack_headers)
            resp.raise_for_status()
            if len(resp.content) > 1000 and not resp.content.startswith(b"<!doctype"):
                return resp.content
        except Exception as exc:
            print(f"[WARN] Fallback Transport Stack poll error: {exc}")

    raise RuntimeError("Could not fetch GTFS-Realtime payload from any configured source.")


def parse_vehicle_positions(payload: bytes) -> tuple[int, list[dict[str, Any]]]:
    """Parse FeedMessage into vehicle records."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(payload)

    feed_ts = int(feed.header.timestamp) if feed.header.HasField("timestamp") else 0
    rows: list[dict[str, Any]] = []

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        vehicle = entity.vehicle
        pos = vehicle.position if vehicle.HasField("position") else None

        lat = float(pos.latitude) if (pos and pos.HasField("latitude") and math.isfinite(pos.latitude)) else None
        lon = float(pos.longitude) if (pos and pos.HasField("longitude") and math.isfinite(pos.longitude)) else None
        bearing = float(pos.bearing) if (pos and pos.HasField("bearing") and math.isfinite(pos.bearing)) else None
        speed = float(pos.speed) if (pos and pos.HasField("speed") and math.isfinite(pos.speed)) else None

        vid = vehicle.vehicle.id or vehicle.vehicle.label or entity.id or None
        route_id = vehicle.trip.route_id if (vehicle.HasField("trip") and vehicle.trip.route_id) else None
        trip_id = vehicle.trip.trip_id if (vehicle.HasField("trip") and vehicle.trip.trip_id) else None

        ts = int(vehicle.timestamp) if vehicle.timestamp > 0 else (feed_ts if feed_ts > 0 else 0)
        ist_time = compute_ist_time(ts)

        rows.append({
            "vehicle_id": str(vid) if vid is not None else None,
            "route_id": str(route_id) if route_id is not None else None,
            "trip_id": str(trip_id) if trip_id is not None else None,
            "latitude": lat,
            "longitude": lon,
            "bearing": bearing,
            "speed": speed,
            "timestamp": ts,
            "ist_time": ist_time,
        })

    return feed_ts, rows


def atomic_write_parquet(df: pl.DataFrame, target_path: Path) -> None:
    """Atomically write a DataFrame to Parquet by writing to .tmp first and renaming."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
    df.write_parquet(tmp_path, compression="zstd")
    os.replace(tmp_path, target_path)


def flush_historical_buffer(buffer: list[dict[str, Any]], out_dir: Path) -> Path | None:
    """Flush accumulated unique pings buffer to timestamped Parquet file and clear buffer."""
    if not buffer:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    flush_ts = datetime.now(tz=IST).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"pings_{flush_ts}.parquet"

    df = pl.DataFrame(buffer, schema=VEHICLE_SCHEMA, strict=False).select(COL_ORDER)
    df.write_parquet(out_path, compression="zstd")
    count = len(buffer)
    buffer.clear()
    file_size_kb = out_path.stat().st_size / 1024.0
    print(f"[FLUSH] Wrote {count:,} unique pings to {out_path.name} ({file_size_kb:.1f} KB)")
    return out_path


def run_collector(poll_interval: int = 10, flush_every: int = 30, once: bool = False) -> None:
    """Fault-tolerant GTFS-Realtime polling collector."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()

    otd_url = os.getenv("OTD_RT_URL", "").strip()
    stack_url = os.getenv("TRANSPORT_STACK_API_URL", "").strip()
    stack_key = os.getenv("TRANSPORT_STACK_API_KEY", "").strip()

    if not otd_url and not stack_url:
        print("[ERROR] Neither OTD_RT_URL nor TRANSPORT_STACK_API_URL configured.")
        return

    print("=================================================================")
    print("Starting Delhi GTFS-RT Live Collector")
    print(f"Poll Interval : {poll_interval}s")
    print(f"Flush Every   : {flush_every} polls (~{(poll_interval * flush_every) / 60:.1f} mins)")
    print(f"Single Poll   : {once}")
    print(f"Live Buffer   : {LIVE_BUFFER_DIR / 'latest_vehicles.parquet'}")
    print(f"Historical Dir: {HISTORICAL_DIR}")
    print("=================================================================")

    last_seen_ts: dict[str, int] = {}
    historical_buffer: list[dict[str, Any]] = []
    poll_count = 0

    client = httpx.Client(timeout=20.0, follow_redirects=True)

    try:
        while True:
            poll_count += 1
            now_dt = datetime.now(tz=IST)
            ist_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

            # 1. Fetch & parse payload with try/except
            try:
                payload = fetch_gtfs_rt(client, otd_url, stack_url, stack_key)
                feed_ts, rows = parse_vehicle_positions(payload)
            except Exception as exc:
                print(f"[{ist_str}] [WARN] Poll #{poll_count} failed: {exc}")
                if once:
                    break
                time.sleep(poll_interval)
                continue

            if not rows:
                print(f"[{ist_str}] [WARN] Poll #{poll_count}: Feed contained 0 vehicle entities.")
            else:
                # 2. Convert active vehicles to DataFrame and write to live_buffer atomically
                current_df = pl.DataFrame(rows, schema=VEHICLE_SCHEMA, strict=False).select(COL_ORDER)
                latest_path = LIVE_BUFFER_DIR / "latest_vehicles.parquet"
                atomic_write_parquet(current_df, latest_path)

                # 3. Track new pings per vehicle_id
                new_pings = 0
                for r in rows:
                    vid = r["vehicle_id"]
                    ts = r["timestamp"]
                    if vid and ts > last_seen_ts.get(vid, 0):
                        last_seen_ts[vid] = ts
                        historical_buffer.append(r)
                        new_pings += 1

                print(
                    f"[{ist_str}] Poll #{poll_count}: {current_df.height:,} active buses, "
                    f"{new_pings:,} new pings buffered (total buffered: {len(historical_buffer):,})"
                )

            # 4. Periodic flush check
            if poll_count % flush_every == 0:
                flush_historical_buffer(historical_buffer, HISTORICAL_DIR)

            # 5. Clean exit if --once
            if once:
                if historical_buffer:
                    flush_historical_buffer(historical_buffer, HISTORICAL_DIR)
                print("[INFO] Single poll complete (--once). Exiting cleanly.")
                break

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[INFO] Collector interrupted by user. Flushing buffered pings...")
        flush_historical_buffer(historical_buffer, HISTORICAL_DIR)
        print("[INFO] Collector stopped.")
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Delhi GTFS-Realtime live collector")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll iteration and exit cleanly for testing",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Polling interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=30,
        help="Number of polls before flushing historical buffer (default: 30)",
    )
    args = parser.parse_args()

    run_collector(
        poll_interval=args.poll_interval,
        flush_every=args.flush_every,
        once=args.once,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
