"""FastAPI server for the Delhi Transit Common-Lines RAPTOR Journey Planner.

Features:
- Async lifespan management: loads StaticIndex, TransitGraph, and CommonLinesRaptor once on startup.
- Background live vehicle refresher updating latest_vehicles.parquet every 30 seconds.
- High-performance endpoints:
  - GET /: Interactive MapLibre GL JS web dashboard.
  - GET /journeys: Time-dependent journey planning enriched with road polyline GeoJSON.
  - GET /stops/search: Fast stop autocomplete across 6,812 Delhi bus stops.
  - GET /live/buses: Active vehicle tracking positions for map visualization.
  - GET /demo/presets: One-click judge benchmark comparison scenarios.
  - GET /health: Operational metrics, models status, and live bus counts.
"""

from __future__ import annotations

import asyncio
import os
import sys
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.geometry import extract_leg_geojson_coords
from src.api.schemas import JourneyOptionSchema, JourneyResponseSchema, LegSchema
from src.ingestion.hygiene import find_project_root
from src.matching.shape_indexer import load_static_index
from src.model.historical_baseline import epoch_to_ist_features
from src.model.live_blender import LiveStateTracker, build_precomputed_horizon_table
from src.router.raptor import CommonLinesRaptor, TransitLeg, WalkLeg
from src.router.transit_graph import TransitGraph, load_transit_graph
from src.router.wait_estimator import LiveVehicleIndex, build_live_vehicle_index

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = find_project_root()
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
LATEST_VEHICLES_PATH = PROJECT_ROOT / "data" / "live_buffer" / "latest_vehicles.parquet"
OBSERVED_SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "observed_segments.parquet"

IST_TZ = timezone(timedelta(hours=5, minutes=30))


def _parse_departure_param(depart_str: str | None) -> float:
    """Parse departure time query parameter into Unix epoch seconds in IST.

    Supports:
    - 'now' or empty string -> current Unix epoch
    - Unix timestamp string (e.g. '1794193200')
    - 'HH:MM' time string (e.g. '08:30'), anchored to the current/next weekday in IST
    - ISO-8601 string (e.g. '2026-11-10T08:30:00+05:30')
    """
    if not depart_str or depart_str.strip().lower() in ("now", ""):
        return time.time()

    val = depart_str.strip()

    # Case 1: Numeric Unix epoch
    try:
        epoch = float(val)
        if epoch > 1e8:
            return epoch
    except ValueError:
        pass

    # Case 2: HH:MM format (anchor to current or next weekday in IST)
    if ":" in val and len(val) in (4, 5):
        parts = val.split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            h, m = int(parts[0]), int(parts[1])
            if 0 <= h <= 23 and 0 <= m <= 59:
                now_ist = datetime.now(IST_TZ)
                target = now_ist.replace(hour=h, minute=m, second=0, microsecond=0)
                # Anchor to next weekday if today is Saturday (5) or Sunday (6)
                if target.weekday() >= 5:
                    days_ahead = 7 - target.weekday()
                    target = target + timedelta(days=days_ahead)
                return target.timestamp()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid HH:MM departure time '{depart_str}'. Expected 00:00 to 23:59.",
        )

    # Case 3: ISO-8601 string
    try:
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST_TZ)
        return dt.timestamp()
    except Exception:
        pass

    # Case 4: Standard "YYYY-MM-DD HH:MM:SS"
    try:
        dt = datetime.strptime(val, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST_TZ)
        return dt.timestamp()
    except Exception:
        pass

    raise HTTPException(
        status_code=400,
        detail=f"Invalid departure time format '{depart_str}'. Use ISO-8601, 'HH:MM', Unix epoch, or 'now'.",
    )


def _parse_location_param(param_str: str, param_name: str) -> str | tuple[float, float]:
    """Parse 'from' or 'to' query parameter as stop name, stop_id, or validated (lat, lon) in Delhi."""
    val = param_str.strip()
    if not val:
        raise HTTPException(status_code=400, detail=f"Query parameter '{param_name}' cannot be empty.")

    if "," in val:
        parts = val.split(",")
        if len(parts) == 2:
            try:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
            except ValueError:
                pass
            else:
                if not (28.3 <= lat <= 28.9 and 76.8 <= lon <= 77.5):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Coordinates ({lat}, {lon}) for '{param_name}' are outside the Delhi NCR "
                            "service bounds (lat 28.3..28.9, lon 76.8..77.5)."
                        ),
                    )
                return (lat, lon)

    return val


async def _refresh_live_vehicles_loop(app: FastAPI) -> None:
    """Non-blocking background task refreshing latest_vehicles.parquet every 30 seconds."""
    while True:
        try:
            await asyncio.sleep(30)
            if LATEST_VEHICLES_PATH.exists():
                live_idx = build_live_vehicle_index(
                    app.state.transit_graph,
                    parquet_path=LATEST_VEHICLES_PATH,
                )
                app.state.live_index = live_idx
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[WARN] Background live vehicle refresh error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup & Shutdown lifecycle manager."""
    print("[STARTUP] Initializing Delhi Transit API...")
    t0 = time.perf_counter()

    # Load Static GTFS index & compiled Transit Graph
    static_index = load_static_index()
    transit_graph = load_transit_graph()

    # Initialize Live State Tracker and ingest observed segments
    tracker = LiveStateTracker()
    if OBSERVED_SEGMENTS_PATH.exists():
        try:
            obs_df = pl.read_parquet(OBSERVED_SEGMENTS_PATH)
            n_obs = tracker.update_from_segments(obs_df)
            print(f"[STARTUP] Ingested {n_obs} corridor live observations into LiveStateTracker")
        except Exception as e:
            print(f"[WARN] Could not ingest observed segments: {e}")

    # Build initial live vehicle index
    live_index = build_live_vehicle_index(
        transit_graph,
        parquet_path=LATEST_VEHICLES_PATH,
    )
    print(f"[STARTUP] Snapped {live_index.n_vehicles} live vehicles into LiveVehicleIndex")

    # Initialize CommonLinesRaptor
    raptor = CommonLinesRaptor(graph=transit_graph, tracker=tracker)

    # Pre-build fast stop autocomplete search index
    stop_search_index = []
    for i in range(transit_graph.n_stops):
        s_id = str(transit_graph.stop_ids[i])
        s_name = str(transit_graph.stop_names[i])
        stop_search_index.append(
            {
                "stop_id": s_id,
                "stop_name": s_name,
                "lat": float(transit_graph.stop_lat[i]),
                "lon": float(transit_graph.stop_lon[i]),
                "search_key": f"{s_name.lower()} {s_id}",
            }
        )

    # Load simplified corridor geometries for traffic heatmap
    corridors_simplified_path = PROJECT_ROOT / "data" / "processed" / "corridors_simplified.json"
    corridors_simplified: dict[str, Any] = {}
    if corridors_simplified_path.exists():
        try:
            with open(corridors_simplified_path, "r", encoding="utf-8") as f:
                corridors_simplified = json.load(f)
            print(f"[STARTUP] Loaded {len(corridors_simplified):,} simplified corridor geometries")
        except Exception as e:
            print(f"[WARN] Failed to load corridors_simplified.json: {e}")

    # Attach to app.state
    app.state.static_index = static_index
    app.state.transit_graph = transit_graph
    app.state.tracker = tracker
    app.state.live_index = live_index
    app.state.raptor = raptor
    app.state.stop_search_index = stop_search_index
    app.state.corridors_simplified = corridors_simplified
    app.state.startup_time = time.time()

    # Launch background live vehicle updater
    refresh_task = asyncio.create_task(_refresh_live_vehicles_loop(app))
    app.state.refresh_task = refresh_task

    print(f"[STARTUP] Completed in {(time.perf_counter() - t0):.2f}s! Server ready.")
    yield

    # Shutdown
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass
    print("[SHUTDOWN] Delhi Transit API background tasks stopped.")


app = FastAPI(
    title="Delhi Transit Time-Dependent Common-Lines Router API",
    version="1.0.0",
    description="Real-time multi-modal journey planner with Live Anomaly Blending, Common-Lines RAPTOR, and P85 Arrival Buffers.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static directory if it exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def serve_dashboard() -> FileResponse:
    """Serve the interactive MapLibre GL JS web dashboard."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    return FileResponse(str(INDEX_HTML))


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Operational health check and network metrics."""
    g: TransitGraph = app.state.transit_graph
    live_idx: LiveVehicleIndex = app.state.live_index
    tracker: LiveStateTracker = app.state.tracker

    return {
        "status": "healthy",
        "models_status": "loaded",
        "n_stops": int(g.n_stops),
        "n_routes": int(g.n_routes),
        "n_corridors": len(tracker.corridor_to_idx),
        "active_live_buses_count": int(live_idx.n_vehicles),
        "live_observations_count": len(tracker.observations),
        "uptime_sec": round(time.time() - app.state.startup_time, 1),
    }


@app.get("/stops/search")
async def search_stops(
    q: str = Query(..., min_length=1, description="Stop name substring or ID query"),
    limit: int = Query(10, ge=1, le=50, description="Max candidates to return"),
) -> list[dict[str, Any]]:
    """Fast substring and prefix autocomplete search across all 6,812 Delhi bus stops."""
    query_clean = q.strip().lower()
    items = app.state.stop_search_index

    # 1. Exact or prefix matches
    prefix_matches = [
        item for item in items if item["search_key"].startswith(query_clean)
    ]
    # 2. Substring matches
    substring_matches = [
        item
        for item in items
        if query_clean in item["search_key"] and item not in prefix_matches
    ]

    results = (prefix_matches + substring_matches)[:limit]
    return [
        {
            "stop_id": r["stop_id"],
            "stop_name": r["stop_name"],
            "lat": r["lat"],
            "lon": r["lon"],
        }
        for r in results
    ]


@app.get("/demo/presets")
async def get_demo_presets() -> list[dict[str, Any]]:
    """Return 4 one-click judge benchmark scenarios."""
    return [
        {
            "id": "preset_kg_np",
            "title": "Kashmere Gate ISBT → Nehru Place Terminal",
            "subtitle": "Direct & Common-Lines Corridor (Peak vs Night Congestion)",
            "from": "ISBT Kashmere Gate",
            "to": "Nehru Place Terminal",
            "depart": "08:30",
            "compare_depart": "23:00",
            "description": "Demonstrates ~18 min congestion delay, P85 safe buffer, and 425CLUP / 449UP common lines.",
        },
        {
            "id": "preset_av_cp",
            "title": "Anand Vihar ISBT → Shivaji Stadium / Connaught Place",
            "subtitle": "East-to-Central Arterial Corridor (Trans-Yamuna Commute)",
            "from": "Anand Vihar ISBT",
            "to": "Shivaji Stadium Terminal",
            "depart": "09:00",
            "description": "High-frequency morning commute through Vikas Marg and ITO.",
        },
        {
            "id": "preset_rohini_aiims",
            "title": "Rohini Sector 22 → AIIMS / Safdarjung",
            "subtitle": "North-West to South Delhi Ring Road Express",
            "from": "Rohini Sector 22",
            "to": "AIIMS",
            "depart": "08:45",
            "description": "Long-distance arterial transfer corridor via Outer Ring Road.",
        },
        {
            "id": "preset_dwarka_ito",
            "title": "Dwarka Mor → ITO AGCR",
            "subtitle": "South-West to Central Delhi Commuter Line",
            "from": "Dwarka Mor",
            "to": "ITO AGCR",
            "depart": "09:15",
            "description": "Peak-hour radial corridor connecting West Delhi to the administrative hub.",
        },
    ]


@app.get("/live/buses")
async def get_live_buses(
    route_id: str | None = Query(None, description="Comma-separated route short names or route IDs"),
    bbox: str | None = Query(None, description="Optional bounding box: min_lon,min_lat,max_lon,max_lat"),
    limit: int = Query(300, ge=1, le=1000, description="Max buses to return"),
) -> list[dict[str, Any]]:
    """Return active live vehicles from latest_vehicles.parquet with route short names and ages."""
    g: TransitGraph = app.state.transit_graph

    if not LATEST_VEHICLES_PATH.exists():
        return []

    try:
        df = pl.read_parquet(LATEST_VEHICLES_PATH)
        if df.height == 0:
            return []

        # Parse filter parameters
        filter_routes = None
        if route_id:
            tokens = [r.strip().lower() for r in route_id.split(",") if r.strip()]
            filter_routes = set(tokens)

        min_lon, min_lat, max_lon, max_lat = None, None, None, None
        if bbox:
            b_parts = [float(x.strip()) for x in bbox.split(",")]
            if len(b_parts) == 4:
                min_lon, min_lat, max_lon, max_lat = b_parts

        now_ts = time.time()
        buses = []

        # Iterate rows
        for row in df.iter_rows(named=True):
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            if lat < 28.0 or lat > 29.5 or lon < 76.5 or lon > 78.0:
                continue

            if min_lon is not None and not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                continue

            r_gtfs = str(row["route_id"])
            r_idx = g.route_id_to_idx.get(r_gtfs)
            short_name = str(g.route_short_names[r_idx]) if r_idx is not None else r_gtfs

            if filter_routes:
                if short_name.lower() not in filter_routes and r_gtfs.lower() not in filter_routes:
                    continue

            ts = float(row["timestamp"])
            buses.append(
                {
                    "vehicle_id": str(row["vehicle_id"]),
                    "route_id": r_gtfs,
                    "route_short_name": short_name,
                    "lat": lat,
                    "lon": lon,
                    "bearing": float(row["bearing"]) if row.get("bearing") is not None else None,
                    "speed": float(row["speed"]) if row.get("speed") is not None else 0.0,
                    "timestamp": int(ts),
                    "age_sec": round(max(0.0, now_ts - ts), 1),
                }
            )
            if len(buses) >= limit:
                break

        return buses
    except Exception as e:
        print(f"[WARN] /live/buses query error: {e}")
        return []


@app.get("/live/traffic-corridors")
async def get_traffic_corridors(
    depart: str = Query("now", description="Departure time ('now', 'HH:MM', epoch, or ISO-8601)"),
    min_ratio: float = Query(0.0, ge=0.0, description="Optional minimum congestion ratio filter"),
) -> dict[str, Any]:
    """Return city-wide corridor traffic congestion heatmap with strict data provenance.

    Provenances:
    - LIVE: >= 1 real vehicle observed on corridor within 45 mins of query time.
    - OBSERVED_HIST: historical GPS observations exist for this (corridor, is_weekend) within +-1 hour.
    - PRIOR: diurnal prior extrapolation only.
    """
    tracker: LiveStateTracker = app.state.tracker
    corrs: dict[str, Any] = getattr(app.state, "corridors_simplified", {})
    t_dep = _parse_departure_param(depart)

    ht = build_precomputed_horizon_table(tracker=tracker, query_epoch=int(t_dep))
    feat = epoch_to_ist_features(t_dep)
    b_idx = int(feat["bin_15m"])
    we = int(feat["is_weekend"])
    eb_table = tracker.eb_prior.eb_table

    features = []
    live_c = 0
    hist_c = 0
    prior_c = 0
    ratios = []

    for cid, idx in tracker.corridor_to_idx.items():
        geom = corrs.get(cid)
        if not geom:
            continue
        t_hat = ht.get_segment_time(idx, t_dep)
        ff = float(ht.free_flow_times[idx])
        ratio = float(t_hat / max(1.0, ff))
        if ratio < min_ratio:
            continue
        ratios.append(ratio)

        # Strict data provenance evaluation
        if cid in tracker.observations and abs(t_dep - tracker.observations[cid].t_obs) <= 2700.0:
            prov = "LIVE"
            live_c += 1
        elif any((cid, b, we) in eb_table for b in range(b_idx - 4, b_idx + 5)):
            prov = "OBSERVED_HIST"
            hist_c += 1
        else:
            prov = "PRIOR"
            prior_c += 1

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": geom},
                "properties": {
                    "corridor_id": cid,
                    "ratio": round(ratio, 2),
                    "provenance": prov,
                    "t_hat_sec": round(t_hat, 1),
                    "free_flow_sec": round(ff, 1),
                },
            }
        )

    dep_dt = datetime.fromtimestamp(t_dep, tz=IST_TZ)
    dep_str = dep_dt.strftime("%Y-%m-%d %H:%M:%S IST")

    mean_r = float(np.mean(ratios)) if ratios else 1.0
    p85_r = float(np.percentile(ratios, 85)) if ratios else 1.0

    return {
        "type": "FeatureCollection",
        "summary": {
            "depart_ist": dep_str,
            "query_epoch": int(t_dep),
            "total_corridors": len(features),
            "live_count": live_c,
            "observed_hist_count": hist_c,
            "prior_count": prior_c,
            "mean_ratio": round(mean_r, 3),
            "p85_ratio": round(p85_r, 3),
        },
        "features": features,
    }


@app.get("/journeys", response_model=JourneyResponseSchema)
async def plan_journey(
    from_param: str = Query(..., alias="from", description="Origin stop name, stop_id, or 'lat,lon'"),
    to_param: str = Query(..., alias="to", description="Destination stop name, stop_id, or 'lat,lon'"),
    depart: str = Query("now", description="Departure time ('now', 'HH:MM', epoch, or ISO-8601)"),
) -> JourneyResponseSchema:
    """Plan Pareto-optimal transit journeys across direct and transfer corridors."""
    t0 = time.perf_counter()

    # Parse and validate origin, destination, and departure time
    origin_query = _parse_location_param(from_param, "from")
    dest_query = _parse_location_param(to_param, "to")
    t_dep = _parse_departure_param(depart)

    raptor: CommonLinesRaptor = app.state.raptor
    g: TransitGraph = app.state.transit_graph
    static_index = app.state.static_index
    live_idx: LiveVehicleIndex = app.state.live_index

    try:
        journeys = raptor.plan_journey(
            origin=origin_query,
            destination=dest_query,
            departure_time=t_dep,
            live_index=live_idx,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Routing error: {e}")

    if not journeys:
        raise HTTPException(
            status_code=404,
            detail=f"No transit journeys found from '{from_param}' to '{to_param}'.",
        )

    routing_lat_ms = (time.perf_counter() - t0) * 1000.0

    # Build options
    option_schemas: list[JourneyOptionSchema] = []

    for idx, j_opt in enumerate(journeys):
        leg_schemas: list[LegSchema] = []

        for leg in j_opt.legs:
            if isinstance(leg, WalkLeg):
                mode = "WALK"
                f_sid = str(g.stop_ids[leg.from_idx]) if leg.from_idx >= 0 else None
                f_name = leg.from_name
                if leg.from_idx >= 0:
                    f_lat = float(g.stop_lat[leg.from_idx])
                    f_lon = float(g.stop_lon[leg.from_idx])
                elif isinstance(origin_query, tuple):
                    f_lat, f_lon = float(origin_query[0]), float(origin_query[1])
                elif leg.to_idx >= 0:
                    f_lat = float(g.stop_lat[leg.to_idx])
                    f_lon = float(g.stop_lon[leg.to_idx])
                else:
                    f_lat, f_lon = 28.6139, 77.2090

                t_sid = str(g.stop_ids[leg.to_idx]) if leg.to_idx >= 0 else None
                t_name = leg.to_name
                if leg.to_idx >= 0:
                    t_lat = float(g.stop_lat[leg.to_idx])
                    t_lon = float(g.stop_lon[leg.to_idx])
                elif isinstance(dest_query, tuple):
                    t_lat, t_lon = float(dest_query[0]), float(dest_query[1])
                elif leg.from_idx >= 0:
                    t_lat = float(g.stop_lat[leg.from_idx])
                    t_lon = float(g.stop_lon[leg.from_idx])
                else:
                    t_lat, t_lon = 28.6139, 77.2090

                geom = extract_leg_geojson_coords(
                    static_index, g, None, None, None, f_lat, f_lon, t_lat, t_lon, "WALK"
                )

                leg_schemas.append(
                    LegSchema(
                        mode="WALK",
                        from_stop_id=f_sid,
                        from_stop_name=f_name,
                        from_lat=round(f_lat, 6),
                        from_lon=round(f_lon, 6),
                        to_stop_id=t_sid,
                        to_stop_name=t_name,
                        to_lat=round(t_lat, 6),
                        to_lon=round(t_lon, 6),
                        routes=[],
                        route_display=f"Walk {leg.dist_m:.0f}m ({leg.walk_sec / 60.0:.1f} mins)",
                        is_common_line=False,
                        wait_time_mins=0.0,
                        in_vehicle_mins=0.0,
                        free_flow_in_vehicle_mins=0.0,
                        walk_distance_m=round(leg.dist_m, 1),
                        walk_duration_mins=round(leg.walk_sec / 60.0, 1),
                        stops_count=0,
                        live_bus_used=False,
                        geometry=geom,
                    )
                )

            elif isinstance(leg, TransitLeg):
                mode = "BUS"
                f_sid = str(g.stop_ids[leg.from_idx])
                f_name = leg.from_name
                f_lat = float(g.stop_lat[leg.from_idx])
                f_lon = float(g.stop_lon[leg.from_idx])

                t_sid = str(g.stop_ids[leg.to_idx])
                t_name = leg.to_name
                t_lat = float(g.stop_lat[leg.to_idx])
                t_lon = float(g.stop_lon[leg.to_idx])

                # Count stops along route
                s_off = int(g.route_stop_offsets[leg.route_idx])
                e_off = int(g.route_stop_offsets[leg.route_idx + 1])
                r_stops = g.route_stops[s_off:e_off]
                pos_u = -1
                pos_v = -1
                for p, st in enumerate(r_stops):
                    if st == leg.from_idx and pos_u < 0:
                        pos_u = p
                    elif st == leg.to_idx and pos_u >= 0:
                        pos_v = p
                        break
                stops_cnt = max(1, pos_v - pos_u) if pos_u >= 0 and pos_v > pos_u else 1

                geom = extract_leg_geojson_coords(
                    static_index,
                    g,
                    leg.route_idx,
                    f_sid,
                    t_sid,
                    f_lat,
                    f_lon,
                    t_lat,
                    t_lon,
                    "BUS",
                )

                live_used = "Live ETA" in leg.route_label or "Live bus" in leg.route_label

                leg_schemas.append(
                    LegSchema(
                        mode="BUS",
                        from_stop_id=f_sid,
                        from_stop_name=f_name,
                        from_lat=round(f_lat, 6),
                        from_lon=round(f_lon, 6),
                        to_stop_id=t_sid,
                        to_stop_name=t_name,
                        to_lat=round(t_lat, 6),
                        to_lon=round(t_lon, 6),
                        routes=leg.candidate_routes,
                        route_display=leg.route_label,
                        is_common_line=leg.is_common_lines,
                        wait_time_mins=round(leg.expected_wait_sec / 60.0, 1),
                        in_vehicle_mins=round(leg.expected_ride_sec / 60.0, 1),
                        free_flow_in_vehicle_mins=round(leg.free_flow_ride_sec / 60.0, 1),
                        walk_distance_m=0.0,
                        walk_duration_mins=0.0,
                        stops_count=stops_cnt,
                        live_bus_used=live_used,
                        geometry=geom,
                    )
                )

        dep_str = datetime.fromtimestamp(j_opt.departure_epoch, tz=IST_TZ).strftime("%H:%M")
        arr_str = datetime.fromtimestamp(j_opt.arrival_epoch, tz=IST_TZ).strftime("%H:%M")
        p85_str = datetime.fromtimestamp(j_opt.p85_arrival_epoch, tz=IST_TZ).strftime("%H:%M")

        option_schemas.append(
            JourneyOptionSchema(
                option_id=idx,
                profile_label=getattr(j_opt, "profile_label", "Fastest"),
                summary=j_opt.summary_str(),
                departure_time_ist=dep_str,
                arrival_time_ist=arr_str,
                p85_arrival_time_ist=p85_str,
                total_duration_mins=round(j_opt.total_duration_sec / 60.0, 1),
                p85_total_duration_mins=round(j_opt.p85_duration_sec / 60.0, 1),
                p85_buffer_mins=round(j_opt.p85_buffer_sec / 60.0, 1),
                free_flow_duration_mins=round(j_opt.free_flow_duration_sec / 60.0, 1),
                traffic_delay_mins=round(j_opt.traffic_delay_sec / 60.0, 1),
                transfers=j_opt.num_transfers,
                total_walk_mins=round(j_opt.total_walk_sec / 60.0, 1),
                total_wait_mins=round(j_opt.total_wait_sec / 60.0, 1),
                total_ride_mins=round(j_opt.total_ride_sec / 60.0, 1),
                legs=leg_schemas,
            )
        )

    return JourneyResponseSchema(
        query={
            "from": from_param,
            "to": to_param,
            "depart": depart,
            "departure_epoch": t_dep,
        },
        routing_latency_ms=round(routing_lat_ms, 2),
        active_live_buses_count=int(live_idx.n_vehicles),
        options=option_schemas,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8550, reload=False)

