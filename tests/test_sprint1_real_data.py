"""Sprint 1 Real-Data & Edge-Case Verification Suite.

Tests:
1. Static GTFS Indexer coverage across all 2,554 Delhi routes and sub-0.5s load time.
2. Clean fallback to stop-to-stop polylines when a route has missing shape_id in trips.txt.
3. Hygiene pipeline: string ID casting, bbox filtering, frozen run collapsing, and teleport removal.
4. Numba 1D Map-Matcher: 3-strike off-route rejection (>75m) and global re-acquisition.
5. Segment Extractor: 100m terminal layover truncation and IsotonicRegression interpolation.
6. Multi-poll real historical data end-to-end verification (monotonic d_m and realistic pace).
"""

from __future__ import annotations

import time
import numpy as np
import polars as pl
import pytest

from src.ingestion.hygiene import (
    LAT0,
    LON0,
    M_PER_DEG_LAT,
    M_PER_DEG_LON,
    load_and_clean_pings,
)
from src.matching.map_matcher import (
    extract_stop_to_stop_segments,
    match_pings,
    match_vehicle_trace,
)
from src.matching.shape_indexer import (
    PROJECT_ROOT,
    RouteShapeInfo,
    ShapePolyline,
    StaticIndex,
    build_static_index,
    load_static_index,
)


@pytest.fixture(scope="module")
def static_index() -> StaticIndex:
    """Load the compiled Delhi StaticIndex once for the test module."""
    t0 = time.time()
    idx = load_static_index()
    elapsed = time.time() - t0
    assert elapsed < 0.5, f"StaticIndex load took {elapsed:.3f}s (expected < 0.5s)"
    return idx


def test_all_2554_delhi_routes_mapped_and_monotonic(static_index: StaticIndex) -> None:
    """Verify all 2,554 routes in routes.txt map to valid shapes with monotonic stop distances."""
    routes_path = PROJECT_ROOT / "data" / "static" / "routes.txt"
    routes_df = pl.read_csv(routes_path, schema_overrides={"route_id": pl.Utf8})
    assert routes_df.height == 2554

    route_ids = routes_df["route_id"].to_list()
    mapped_count = 0
    for rid in route_ids:
        cand_shapes = static_index.get_candidate_shapes(rid)
        assert len(cand_shapes) >= 1, f"Route {rid} has no candidate shapes"
        for sid in cand_shapes:
            poly = static_index.get_shape(sid)
            assert poly is not None, f"Shape {sid} missing from index"
            assert len(poly.x_m) >= 2
            assert poly.total_length_m > 0.0
            # Cumulative arc-length must be non-decreasing
            assert np.all(np.diff(poly.s_m) >= 0.0)

            rs_info = static_index.get_route_shape(rid, sid)
            assert rs_info is not None, f"RouteShapeInfo missing for ({rid}, {sid})"
            assert len(rs_info.stop_ids) >= 2
            # Projected stop positions must be strictly monotonic increasing
            stop_diffs = np.diff(rs_info.stop_dists_m)
            assert np.all(stop_diffs > 0.0), f"Non-monotonic stops on ({rid}, {sid})"
        mapped_count += 1

    assert mapped_count == 2554


def test_missing_shape_id_fallback_to_stop_polyline() -> None:
    """Verify routes with null/missing shape_id in trips.txt fall back to stop-to-stop polylines."""
    routes_df = pl.DataFrame({"route_id": ["R_FALLBACK"]})
    trips_df = pl.DataFrame({
        "route_id": ["R_FALLBACK"],
        "trip_id": ["T_1"],
        "shape_id": [None],
    })
    stops_df = pl.DataFrame({
        "stop_id": ["S1", "S2", "S3"],
        "stop_name": ["Origin", "Mid", "Dest"],
        "stop_lat": [LAT0, LAT0 + 0.005, LAT0 + 0.010],
        "stop_lon": [LON0, LON0 + 0.005, LON0 + 0.010],
    })
    stop_times_df = pl.DataFrame({
        "trip_id": ["T_1", "T_1", "T_1"],
        "stop_id": ["S1", "S2", "S3"],
        "stop_sequence": [0, 1, 2],
    })
    shapes_df = pl.DataFrame(
        schema={
            "shape_id": pl.Utf8,
            "shape_pt_lat": pl.Float64,
            "shape_pt_lon": pl.Float64,
            "shape_pt_sequence": pl.Int64,
        }
    )

    idx = build_static_index(
        routes_df=routes_df,
        trips_df=trips_df,
        stops_df=stops_df,
        stop_times_df=stop_times_df,
        shapes_df=shapes_df,
    )

    cands = idx.get_candidate_shapes("R_FALLBACK")
    assert cands == ["fallback_R_FALLBACK"]

    poly = idx.get_shape("fallback_R_FALLBACK")
    assert poly is not None
    assert len(poly.x_m) == 3
    assert poly.total_length_m > 1000.0

    rs_info = idx.get_route_shape("R_FALLBACK", "fallback_R_FALLBACK")
    assert rs_info is not None
    assert rs_info.stop_ids == ["S1", "S2", "S3"]
    assert np.all(np.diff(rs_info.stop_dists_m) > 0.0)


def test_hygiene_string_id_casting_frozen_collapse_and_teleport_filter(
    static_index: StaticIndex,
) -> None:
    """Verify integer route_id casting, stationary run collapse, and teleport filtering."""
    # Pick a real route_id that looks numeric (e.g., 2204)
    raw_df = pl.DataFrame({
        "vehicle_id": ["V1", "V1", "V1", "V1", "V1", "V1", "V1", "V1"],
        "route_id": [2204, 2204, 2204, 2204, 2204, 2204, 2204, 2204],  # int64!
        "latitude": [
            28.613900,  # t=1000: initial
            28.613900,  # t=1000: duplicate timestamp
            28.613901,  # t=1010: stationary (<1m) - first of stationary run
            28.613902,  # t=1020: stationary (<1m) - middle (should be collapsed!)
            28.613901,  # t=1030: stationary (<1m) - last of stationary run
            28.614300,  # t=1040: valid move (~44m in 10s = 4.4 m/s)
            28.650000,  # t=1045: impossible teleport (~4000m in 5s = 800 m/s -> drop!)
            12.971600,  # t=1050: Bangalore out-of-bbox -> drop!
        ],
        "longitude": [LON0] * 8,
        "timestamp": [1000, 1000, 1010, 1020, 1030, 1040, 1045, 1050],
    })

    cleaned = load_and_clean_pings(raw_df)

    # Check string casting
    assert cleaned.schema["vehicle_id"] == pl.Utf8
    assert cleaned.schema["route_id"] == pl.Utf8
    assert cleaned["route_id"][0] == "2204"

    # Expected kept timestamps:
    # 1000 (initial), 1010 (first stationary), 1030 (last stationary), 1040 (valid move)
    # 1020 (middle stationary), 1045 (teleport), and 1050 (out of bbox) must be removed
    kept_ts = cleaned["timestamp"].to_list()
    assert kept_ts == [1000, 1010, 1030, 1040]

    # Verify unit movement vector on the ~44m step at t=1040
    last_row = cleaned.row(-1, named=True)
    assert last_row["step_dist_m"] >= 20.0
    assert bool(last_row["has_bearing"]) is True
    assert np.isclose(last_row["vx"] ** 2 + last_row["vy"] ** 2, 1.0, atol=1e-5)

    # Verify string route_id matches static_index cleanly
    matched = match_pings(cleaned, static_index)
    assert matched.height == 4
    assert matched["shape_id"][0] is not None


def test_off_route_rejection_and_reacquisition() -> None:
    """Verify >75m for 3 consecutive pings triggers is_off_route=True and global re-acquisition."""
    # Straight east-west polyline of length 2000m along y=0
    poly_x = np.array([0.0, 1000.0, 2000.0], dtype=np.float64)
    poly_y = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    poly_s = np.array([0.0, 1000.0, 2000.0], dtype=np.float64)
    tx = np.array([1.0, 1.0], dtype=np.float64)
    ty = np.array([0.0, 0.0], dtype=np.float64)

    # Trace:
    # Ping 0: on-route at (100, 5)
    # Ping 1: off-route strike 1 at (200, 120) -> perp_dist = 120m > 75m
    # Ping 2: off-route strike 2 at (300, 130) -> perp_dist = 130m > 75m
    # Ping 3: off-route strike 3 at (400, 140) -> perp_dist = 140m > 75m -> is_off_route = True!
    # Ping 4: returns on-route far ahead at (1600, 4) -> global re-acquisition!
    p_x = np.array([100.0, 200.0, 300.0, 400.0, 1600.0], dtype=np.float64)
    p_y = np.array([5.0, 120.0, 130.0, 140.0, 4.0], dtype=np.float64)
    p_vx = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    p_vy = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    p_hb = np.array([True, True, True, True, True], dtype=np.bool_)
    p_dt = np.array([0.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)

    d_arr, perp_arr, _, off_arr = match_vehicle_trace(
        p_x, p_y, p_vx, p_vy, p_hb, p_dt, poly_x, poly_y, poly_s, tx, ty
    )

    assert list(off_arr) == [False, False, False, True, False]
    assert np.isclose(perp_arr[3], 140.0)
    # Ping 4 at x=1600m is >1000m ahead of Ping 2 (x=300m), beyond a normal 10s local window,
    # proving global re-acquisition [0, L] was triggered after Ping 3 went off-route.
    assert np.isclose(d_arr[4], 1600.0)


def test_terminal_layover_stripping_and_isotonic_segments() -> None:
    """Verify 100m terminal layover stripping at first/last stops and monotonic interpolation."""
    poly = ShapePolyline(
        shape_id="SHP_SYN",
        x_m=np.array([0.0, 500.0, 1000.0], dtype=np.float64),
        y_m=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        s_m=np.array([0.0, 500.0, 1000.0], dtype=np.float64),
        tx=np.array([1.0, 1.0], dtype=np.float64),
        ty=np.array([0.0, 0.0], dtype=np.float64),
        total_length_m=1000.0,
    )
    rs_info = RouteShapeInfo(
        route_id="R_SYN",
        shape_id="SHP_SYN",
        canonical_trip_id="T_SYN",
        stop_ids=["STOP_A", "STOP_B", "STOP_C"],
        stop_dists_m=np.array([0.0, 500.0, 1000.0], dtype=np.float64),
    )
    syn_index = StaticIndex(
        route_to_shapes={"R_SYN": ["SHP_SYN"]},
        shapes={"SHP_SYN": poly},
        route_shape_info={("R_SYN", "SHP_SYN"): rs_info},
    )

    # Vehicle idles near STOP_A (d=10m..30m) from t=100 to t=600 (500s layover),
    # departs at t=600, has a slight backward GPS jitter at t=630 (260m -> 250m),
    # passes STOP_B (500m) at t=650, enters within 100m of STOP_C (950m) at t=700,
    # and idles at destination terminal from t=700 to t=1500.
    matched_df = pl.DataFrame({
        "vehicle_id": ["BUS_1"] * 8,
        "route_id": ["R_SYN"] * 8,
        "shape_id": ["SHP_SYN"] * 8,
        "timestamp": [100, 350, 600, 625, 630, 650, 700, 1500],
        "d_m": [10.0, 20.0, 30.0, 260.0, 250.0, 500.0, 950.0, 990.0],
        "perp_dist_m": [2.0] * 8,
        "bearing_dot": [1.0] * 8,
        "is_off_route": [False] * 8,
    })

    seg_df = extract_stop_to_stop_segments(matched_df, syn_index)
    assert seg_df.height == 2

    rows = seg_df.to_dicts()
    # Segment 1: STOP_A -> STOP_B (500m in ~50s, stripped of the 500s origin layover!)
    assert rows[0]["stop_u"] == "STOP_A"
    assert rows[0]["stop_v"] == "STOP_B"
    assert rows[0]["corridor_segment_id"] == "STOP_A_STOP_B_500"
    assert np.isclose(rows[0]["dep_time_u"], 600.0, atol=1.0)
    assert np.isclose(rows[0]["dep_time_v"], 650.0, atol=1.0)
    assert np.isclose(rows[0]["travel_time_sec"], 50.0, atol=1.0)
    assert np.isclose(rows[0]["pace_sec_per_m"], 0.10, atol=0.01)

    # Segment 2: STOP_B -> STOP_C (500m in ~50s, stripped of the 800s destination layover!)
    assert rows[1]["stop_u"] == "STOP_B"
    assert rows[1]["stop_v"] == "STOP_C"
    assert rows[1]["corridor_segment_id"] == "STOP_B_STOP_C_500"
    assert np.isclose(rows[1]["travel_time_sec"], 50.0, atol=1.0)
    assert np.isclose(rows[1]["pace_sec_per_m"], 0.10, atol=0.01)


def test_real_multi_poll_historical_pipeline(static_index: StaticIndex) -> None:
    """Verify full hygiene + map_matcher + segment extraction on real Delhi historical data."""
    cleaned_df = load_and_clean_pings()
    assert cleaned_df.height > 15000
    assert cleaned_df.schema["vehicle_id"] == pl.Utf8
    assert cleaned_df.schema["route_id"] == pl.Utf8

    matched_df = match_pings(cleaned_df, static_index)
    assert matched_df.height == cleaned_df.height

    on_route_ratio = matched_df.filter(~pl.col("is_off_route")).height / matched_df.height
    assert on_route_ratio > 0.75, f"Expected >75% on-route ratio, got {on_route_ratio:.2%}"

    seg_df = extract_stop_to_stop_segments(matched_df, static_index)
    assert seg_df.height > 1000, f"Expected >1000 real segment traversals, got {seg_df.height}"

    expected_cols = [
        "trip_run_id",
        "vehicle_id",
        "route_id",
        "shape_id",
        "stop_u",
        "stop_v",
        "corridor_segment_id",
        "dep_time_u",
        "dep_time_v",
        "travel_time_sec",
        "segment_length_m",
        "pace_sec_per_m",
    ]
    for col in expected_cols:
        assert col in seg_df.columns

    # Verify realistic pace bounds (0.03 to 0.60 s/m) and positive segment lengths
    min_pace = seg_df["pace_sec_per_m"].min()
    max_pace = seg_df["pace_sec_per_m"].max()
    assert min_pace >= 0.03
    assert max_pace <= 0.60
    assert seg_df["segment_length_m"].min() > 10.0
    assert (seg_df["dep_time_v"] > seg_df["dep_time_u"]).all()
