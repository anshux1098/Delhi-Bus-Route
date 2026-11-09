"""Sprint 4 Verification & Hardening Suite (`tests/test_sprint4_api.py`).

Tests the FastAPI server (`src/api/app.py`), Pydantic contracts (`src/api/schemas.py`),
GTFS polyline slicing (`src/api/geometry.py`), and MapLibre GL JS dashboard (`src/api/static/index.html`)
using `fastapi.testclient.TestClient` without blocking uvicorn processes.

Enforces the 5 required Sprint 4 audit checks:
1. Flexible `from`, `to`, and `depart` Parsing in `GET /journeys`:
   - Stop name string: `?from=Kashmere Gate ISBT&to=Nehru Place Terminal`
   - Coordinate string: `?from=28.6675,77.2285&to=28.5494,77.2519`
   - Stop ID string: `?from=4696&to=601`
   - `depart` formats: ISO-8601 (`2026-11-10T08:30:00+05:30`), `"08:30"`, Unix epoch (`1794193200`), `"now"`
   - Clean HTTP 400/404 JSON error responses (never HTTP 500 tracebacks) for invalid stop names or coordinates outside Delhi.
2. Accurate Polyline Slicing (`src/api/geometry.py`):
   - `extract_leg_geojson_coords` slices route shape vertices between `d_stop_u` and `d_stop_v` in correct travel order and returns valid `[lon, lat]` pairs within Delhi bounds (`76.8 <= lon <= 77.5`, `28.3 <= lat <= 28.9`).
3. Diurnal API Contract Verification:
   - `GET /journeys?from=ISBT Kashmere Gate&to=Nehru Place Terminal&depart=2026-11-10T08:30:00+05:30` vs `depart=2026-11-10T23:00:00+05:30`
   - Asserts HTTP 200, non-empty `options`, `p85_total_duration_mins > total_duration_mins`, `traffic_delay_mins > 0`, and that the 08:30 AM trip takes significantly longer than the 23:00 PM trip.
4. UI Asset & Endpoint Wiring:
   - `GET /` returns HTTP 200 HTML containing MapLibre GL JS script and calls `/journeys`, `/stops/search`, and `/live/buses`.
5. Additional Endpoint Contract Checks:
   - `/stops/search`, `/live/buses`, `/demo/presets`, and `/health`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.geometry import extract_leg_geojson_coords

IST_TZ = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a module-scoped TestClient that runs the FastAPI lifespan once."""
    with TestClient(app) as tc:
        yield tc


# ==============================================================================
# Check 1: Flexible `from`, `to`, and `depart` Parsing in `GET /journeys`
# ==============================================================================


def test_1a_journey_input_formats_stop_name_coord_and_stop_id(client: TestClient) -> None:
    """Verify `from` and `to` work seamlessly for stop names, coordinate strings, and stop IDs."""
    # a) Stop name string
    r_name = client.get(
        "/journeys",
        params={
            "from": "Kashmere Gate ISBT",
            "to": "Nehru Place Terminal",
            "depart": "2026-11-10T08:30:00+05:30",
        },
    )
    assert r_name.status_code == 200, r_name.text
    d_name = r_name.json()
    assert len(d_name["options"]) >= 1

    # b) Coordinate string ("lat,lon")
    r_coord = client.get(
        "/journeys",
        params={
            "from": "28.6675,77.2285",
            "to": "28.5494,77.2519",
            "depart": "2026-11-10T08:30:00+05:30",
        },
    )
    assert r_coord.status_code == 200, r_coord.text
    d_coord = r_coord.json()
    assert len(d_coord["options"]) >= 1

    # c) Stop ID / Stop Index string ("4696" -> "601")
    r_id = client.get(
        "/journeys",
        params={
            "from": "4696",
            "to": "601",
            "depart": "2026-11-10T08:30:00+05:30",
        },
    )
    assert r_id.status_code == 200, r_id.text
    d_id = r_id.json()
    assert len(d_id["options"]) >= 1


def test_1b_depart_time_formats_and_weekday_anchoring(client: TestClient) -> None:
    """Verify `depart` parses ISO-8601, '08:30' (anchored to weekday in IST), Unix epoch, and 'now'."""
    # 1. ISO-8601 string
    r_iso = client.get(
        "/journeys",
        params={
            "from": "ISBT Kashmere Gate",
            "to": "Nehru Place Terminal",
            "depart": "2026-11-10T08:30:00+05:30",
        },
    )
    assert r_iso.status_code == 200
    assert r_iso.json()["options"][0]["departure_time_ist"] == "08:30"

    # 2. "08:30" time string -> anchored to weekday in IST
    r_hhmm = client.get(
        "/journeys",
        params={
            "from": "ISBT Kashmere Gate",
            "to": "Nehru Place Terminal",
            "depart": "08:30",
        },
    )
    assert r_hhmm.status_code == 200
    d_hhmm = r_hhmm.json()
    assert d_hhmm["options"][0]["departure_time_ist"] == "08:30"
    dep_dt = datetime.fromtimestamp(d_hhmm["query"]["departure_epoch"], tz=IST_TZ)
    assert dep_dt.weekday() < 5, f"Expected weekday (0..4) for '08:30', got weekday={dep_dt.weekday()}"

    # 3. Unix timestamp ("1794193200")
    r_epoch = client.get(
        "/journeys",
        params={
            "from": "ISBT Kashmere Gate",
            "to": "Nehru Place Terminal",
            "depart": "1794193200",
        },
    )
    assert r_epoch.status_code == 200
    assert abs(r_epoch.json()["query"]["departure_epoch"] - 1794193200.0) < 1.0

    # 4. "now"
    r_now = client.get(
        "/journeys",
        params={
            "from": "ISBT Kashmere Gate",
            "to": "Nehru Place Terminal",
            "depart": "now",
        },
    )
    assert r_now.status_code == 200
    assert len(r_now.json()["options"]) >= 1


def test_1c_clean_error_responses_for_invalid_inputs(client: TestClient) -> None:
    """Verify invalid stop names or out-of-Delhi coordinates return HTTP 400/404 JSON (never HTTP 500)."""
    # Non-existent stop name outside Delhi
    r_bad_stop = client.get(
        "/journeys",
        params={
            "from": "Chhatrapati Shivaji Maharaj Terminus Mumbai XYZ999",
            "to": "Nehru Place Terminal",
            "depart": "08:30",
        },
    )
    assert r_bad_stop.status_code in (400, 404)
    assert "detail" in r_bad_stop.json()

    # Coordinates outside Delhi NCR bounds (e.g., Mumbai 19.0760, 72.8777)
    r_bad_coord = client.get(
        "/journeys",
        params={
            "from": "19.0760,72.8777",
            "to": "28.5494,77.2519",
            "depart": "08:30",
        },
    )
    assert r_bad_coord.status_code in (400, 404)
    assert "detail" in r_bad_coord.json()

    # Invalid departure string
    r_bad_depart = client.get(
        "/journeys",
        params={
            "from": "ISBT Kashmere Gate",
            "to": "Nehru Place Terminal",
            "depart": "not-a-timestamp",
        },
    )
    assert r_bad_depart.status_code == 400
    assert "detail" in r_bad_depart.json()


# ==============================================================================
# Check 2: Accurate Polyline Slicing (`src/api/geometry.py`)
# ==============================================================================


def test_2_accurate_polyline_slicing_and_delhi_bounds(client: TestClient) -> None:
    """Verify `extract_leg_geojson_coords` slices route shape vertices in travel order within Delhi bounds."""
    static_index = app.state.static_index
    g = app.state.transit_graph

    # Pick route 425CLUP or a multi-stop route with a rich shape
    r_idx = 0
    for cand_r in range(g.n_routes):
        s_off = int(g.route_stop_offsets[cand_r])
        e_off = int(g.route_stop_offsets[cand_r + 1])
        if (e_off - s_off) >= 15:
            r_idx = cand_r
            break

    s_off = int(g.route_stop_offsets[r_idx])
    u_idx = int(g.route_stops[s_off + 2])
    v_idx = int(g.route_stops[s_off + 12])

    u_sid = str(g.stop_ids[u_idx])
    v_sid = str(g.stop_ids[v_idx])
    u_lat, u_lon = float(g.stop_lat[u_idx]), float(g.stop_lon[u_idx])
    v_lat, v_lon = float(g.stop_lat[v_idx]), float(g.stop_lon[v_idx])

    coords = extract_leg_geojson_coords(
        static_index,
        g,
        r_idx,
        u_sid,
        v_sid,
        u_lat,
        u_lon,
        v_lat,
        v_lon,
        "BUS",
    )

    assert isinstance(coords, list)
    assert len(coords) >= 2

    # Verify all coordinates are valid [lon, lat] within Delhi NCR bounds
    for lon, lat in coords:
        assert 76.8 <= lon <= 77.5, f"Longitude {lon} outside Delhi bounds [76.8, 77.5]"
        assert 28.3 <= lat <= 28.9, f"Latitude {lat} outside Delhi bounds [28.3, 28.9]"

    # Verify travel order: start of sliced polyline is close to stop u, end is close to stop v
    start_lon, start_lat = coords[0]
    end_lon, end_lat = coords[-1]
    dist_start_to_u = np.hypot(start_lon - u_lon, start_lat - u_lat)
    dist_end_to_v = np.hypot(end_lon - v_lon, end_lat - v_lat)
    dist_start_to_v = np.hypot(start_lon - v_lon, start_lat - v_lat)

    assert dist_start_to_u < 0.01, f"Start of polyline not near origin stop u: {dist_start_to_u}"
    assert dist_end_to_v < 0.01, f"End of polyline not near destination stop v: {dist_end_to_v}"
    assert dist_start_to_u < dist_start_to_v, "Polyline sliced in reversed order!"


# ==============================================================================
# Check 3: Diurnal API Contract Verification
# ==============================================================================


def test_3_diurnal_api_contract_peak_vs_night(client: TestClient) -> None:
    """Verify 08:30 AM vs 23:00 PM journeys on Kashmere Gate ISBT -> Nehru Place Terminal."""
    r_peak = client.get(
        "/journeys",
        params={
            "from": "ISBT Kashmere Gate",
            "to": "Nehru Place Terminal",
            "depart": "2026-11-10T08:30:00+05:30",
        },
    )
    r_night = client.get(
        "/journeys",
        params={
            "from": "ISBT Kashmere Gate",
            "to": "Nehru Place Terminal",
            "depart": "2026-11-10T23:00:00+05:30",
        },
    )

    assert r_peak.status_code == 200
    assert r_night.status_code == 200

    peak_data = r_peak.json()
    night_data = r_night.json()

    assert len(peak_data["options"]) >= 1
    assert len(night_data["options"]) >= 1

    peak_opt = peak_data["options"][0]
    night_opt = night_data["options"][0]

    # Assert P85 safe duration > expected duration and positive traffic delay
    assert peak_opt["p85_total_duration_mins"] > peak_opt["total_duration_mins"]
    assert peak_opt["p85_buffer_mins"] > 0
    assert peak_opt["traffic_delay_mins"] > 0

    # Assert 08:30 AM peak trip takes significantly longer than 23:00 PM night trip
    assert peak_opt["total_duration_mins"] > night_opt["total_duration_mins"] + 8.0, (
        f"Expected 08:30 AM duration ({peak_opt['total_duration_mins']}m) to be significantly "
        f"longer than 23:00 PM duration ({night_opt['total_duration_mins']}m)"
    )

    # Assert all legs have valid GeoJSON coordinates within Delhi bounds
    for opt in peak_data["options"]:
        for leg in opt["legs"]:
            coords = leg["geometry"]
            assert len(coords) >= 2
            for lon, lat in coords:
                assert 76.8 <= lon <= 77.5
                assert 28.3 <= lat <= 28.9


# ==============================================================================
# Check 4: UI Asset & Endpoint Wiring
# ==============================================================================


def test_4_ui_dashboard_and_supporting_endpoints(client: TestClient) -> None:
    """Verify GET / serves the MapLibre GL JS HTML dashboard and calls all backend endpoints."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    html = res.text

    # Verify MapLibre GL JS script and API endpoint calls exist in index.html
    assert "maplibre-gl" in html
    assert "/journeys" in html
    assert "/stops/search" in html
    assert "/live/buses" in html

    # Verify supporting endpoints return HTTP 200 and valid payloads
    r_stops = client.get("/stops/search", params={"q": "Kashmere", "limit": 10})
    assert r_stops.status_code == 200
    stops = r_stops.json()
    assert len(stops) >= 1
    assert "Kashmere" in stops[0]["stop_name"]

    r_buses = client.get("/live/buses", params={"limit": 50})
    assert r_buses.status_code == 200
    buses = r_buses.json()
    assert isinstance(buses, list)
    assert len(buses) >= 1
    assert "vehicle_id" in buses[0]
    assert "route_short_name" in buses[0]

    r_presets = client.get("/demo/presets")
    assert r_presets.status_code == 200
    assert len(r_presets.json()) == 4

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"
    assert r_health.json()["n_stops"] == 6812
    assert r_health.json()["n_routes"] == 2554
