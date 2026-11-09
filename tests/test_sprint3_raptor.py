"""Sprint 3 Mathematical & Algorithmic Verification Suite for Transit Graph, Wait Estimator, and RAPTOR.

Strictly verifies the 6 audit checks:
1. Subpath Correlated Variance Formula (Ruling Q1b) with rho = 0.600 & Lognormal q85_J > mu_J.
2. Common-Lines Wait Reduction (E[W_A] < min E[W_r]) & 0-Transfer Through-Riding Preservation (Ruling Q2).
3. Live Upstream Bus Gating (Delta_star = 240s vs >1500s fallback) & FIFO Monotonic Departure (Ruling Q3).
4. Strict Ban on Consecutive Walk Legs (WALK -> WALK) across all returned journeys (Audit Fix 4.6).
5. Problem Statement Section 2.1 & 2.2 Diurnal Benchmark (Kashmere Gate ISBT -> Nehru Place Terminal,
   08:30 AM Tuesday vs 11:00 PM Tuesday vs 05:30 AM Free-Flow).
6. End-to-End In-Memory Routing Latency < 100 ms on CPU.
"""

from __future__ import annotations

import copy
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.model.lgbm_trainer import load_travel_time_models
from src.model.live_blender import build_precomputed_horizon_table
from src.router.raptor import CommonLinesRaptor, TransitLeg, WalkLeg
from src.router.transit_graph import TransitGraph, load_transit_graph
from src.router.wait_estimator import (
    RHO_EMPIRICAL,
    LiveVehicleIndex,
    RoutePrefixArrays,
    build_route_prefix_arrays,
    compute_leg_wait_and_ride,
)

IST_TZ = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture(scope="module")
def transit_graph() -> TransitGraph:
    return load_transit_graph()


@pytest.fixture(scope="module")
def raptor_router(transit_graph: TransitGraph) -> CommonLinesRaptor:
    return CommonLinesRaptor(graph=transit_graph)


def test_1_subpath_correlated_variance_and_lognormal_p85(
    transit_graph: TransitGraph,
    raptor_router: CommonLinesRaptor,
) -> None:
    """Check 1 (Ruling Q1b): V(i -> j) = (1 - rho)*(S2[j] - S2[i]) + rho*(S1[j] - S1[i])^2
    with rho = 0.600 from Sprint 2, and q85_J > mu_J on every returned journey.
    """
    artifacts = load_travel_time_models()
    rho_model = float(artifacts.get("rho", 0.600))
    assert abs(rho_model - 0.600) < 1e-3
    assert abs(RHO_EMPIRICAL - 0.600) < 1e-3

    t_q = int(datetime(2026, 11, 10, 8, 30, tzinfo=IST_TZ).timestamp())
    ht = build_precomputed_horizon_table(tracker=raptor_router.tracker, query_epoch=t_q)
    pref = build_route_prefix_arrays(transit_graph, horizon_table=ht, query_time=t_q, rho=0.600)

    # Verify exact subpath variance on Route 2452 (425CLUP) across multiple subpaths
    r_idx = 2452
    off = int(transit_graph.route_stop_offsets[r_idx])
    for i, j in [(0, 10), (5, 20), (0, 28)]:
        mu_leg, ff_leg, v_leg = pref.get_leg_stats(r_idx, i, j)
        s1_diff = float(pref.S1[off + j] - pref.S1[off + i])
        s2_diff = float(pref.S2[off + j] - pref.S2[off + i])
        expected_v = (1.0 - 0.600) * s2_diff + 0.600 * (s1_diff**2)

        assert mu_leg > ff_leg > 0.0
        assert abs(v_leg - expected_v) < 1e-2, f"Variance mismatch on ({i}->{j}): {v_leg} != {expected_v}"

    # Verify lognormal P85 formula and q85_J > mu_J on all returned journeys
    journeys = raptor_router.plan_journey(
        origin="ISBT Kashmere Gate",
        destination="Nehru Place Terminal",
        departure_time=t_q,
        horizon_table=ht,
        prefix_arrays=pref,
    )
    assert len(journeys) >= 2
    for j_opt in journeys:
        mu_J = j_opt.total_duration_sec
        V_J = j_opt.journey_variance
        s2 = math.log(1.0 + V_J / (mu_J**2))
        expected_q85 = max(mu_J * 1.03, mu_J * math.exp(-0.5 * s2 + 1.036 * math.sqrt(s2)))
        assert abs(j_opt.p85_duration_sec - expected_q85) < 1e-3
        assert j_opt.p85_duration_sec > mu_J, "Lognormal q85_J must be strictly greater than mu_J"
        assert j_opt.p85_buffer_sec > 0.0


def test_2_common_lines_wait_reduction_and_through_riding(
    transit_graph: TransitGraph,
    raptor_router: CommonLinesRaptor,
) -> None:
    """Check 2 (Ruling Q2):
    - On shared corridor (u, v) with m >= 2 routes in A(u, v), E[W_A] < min_{r in A} E[W_r].
    - A passenger staying on route r past v to w (where other routes in A(u, v) diverged)
      uses A(u, w) with 0 transfers rather than paying a transfer penalty at v.
    """
    t_q = int(datetime(2026, 11, 10, 8, 30, tzinfo=IST_TZ).timestamp())
    ht = build_precomputed_horizon_table(tracker=raptor_router.tracker, query_epoch=t_q)
    pref = build_route_prefix_arrays(transit_graph, horizon_table=ht, query_time=t_q)
    bucket = 1

    # Real corridor triple on Route 1 (102AUP):
    # u = 4696 (Ashok Vihar Crossing)
    # v = 601  (Wazirpur DTC Depot / Netaji Subhash Palace) -> 13 common lines in A(u, v)
    # w = 55   (Madhuban Chowk) -> 11 routes diverge at v, leaving 2 routes in A(u, w)
    u, v, w = 4696, 601, 55
    cl_uv = transit_graph.lookup_common_lines(u, v, bucket)
    cl_uw = transit_graph.lookup_common_lines(u, w, bucket)

    assert cl_uv is not None and len(cl_uv["routes"]) >= 2
    assert cl_uw is not None and len(cl_uw["routes"]) < len(cl_uv["routes"])

    # Verify E[W_A] < min_{r in A} E[W_r] on both (u, v) and (u, w) and (4716, 1943)
    for stop_a, stop_b in [(u, v), (u, w), (4716, 1943)]:
        cl = transit_graph.lookup_common_lines(stop_a, stop_b, bucket)
        assert cl is not None and len(cl["routes"]) >= 2
        e_w_combined, _, _, _, _, lbl = compute_leg_wait_and_ride(
            transit_graph,
            pref,
            None,
            stop_a,
            stop_b,
            float(t_q),
            float(t_q),
            bucket,
            candidate_routes=list(cl["routes"]),
            attractive_set_info=cl,
        )
        min_single_wait = min(float(transit_graph.EW_sec[r, bucket]) for r in cl["routes"])
        assert e_w_combined < min_single_wait, (
            f"Combined wait {e_w_combined:.1f}s must be strictly less than min single wait {min_single_wait:.1f}s"
        )
        assert "Take any of:" in lbl

    # Verify Through-Riding Preservation from u (4696) past v (601) to w (55):
    # Must arrive with 0 transfers using A(u, w) directly!
    journeys_uw = raptor_router.plan_journey(
        origin=u,
        destination=w,
        departure_time=t_q,
        horizon_table=ht,
        prefix_arrays=pref,
    )
    assert len(journeys_uw) >= 1
    best_uw = journeys_uw[0]
    assert best_uw.num_transfers == 0, (
        f"Through-riding passenger from {u} to {w} must have 0 transfers, got {best_uw.num_transfers}"
    )
    bus_legs = [l for l in best_uw.legs if isinstance(l, TransitLeg)]
    assert len(bus_legs) == 1
    assert bus_legs[0].from_idx == u and bus_legs[0].to_idx == w
    assert set(bus_legs[0].candidate_routes) == {
        str(transit_graph.route_short_names[r]) for r in cl_uw["routes"]
    }


def test_3_live_upstream_bus_gating_and_monotonic_departure(
    transit_graph: TransitGraph,
    raptor_router: CommonLinesRaptor,
) -> None:
    """Check 3 (Ruling Q3):
    - Synthetic live upstream bus 4 minutes away (Delta_star = 240s) drops E[W_A] toward ~240s
      when coverage is high, and falls back smoothly to headway prior when ETA_b - t_q > 1500s.
    - Arrival-order monotonicity: arriving at t_a + 60s never produces an earlier expected
      departure t_a + E[W] than arriving at t_a.
    """
    t_q = 1794279600.0  # 08:30 AM Tuesday
    ht = build_precomputed_horizon_table(tracker=raptor_router.tracker, query_epoch=int(t_q))
    pref = build_route_prefix_arrays(transit_graph, horizon_table=ht, query_time=t_q)
    bucket = 1

    # Use Route 2452 (425CLUP): stop position 10 (u) -> stop position 28 (v = 1944)
    r_idx = 2452
    s = int(transit_graph.route_stop_offsets[r_idx])
    pos_u = 10
    pos_v = 28
    u_idx = int(transit_graph.route_stops[s + pos_u])
    v_idx = int(transit_graph.route_stops[s + pos_v])

    # Find upstream distance d_bus along route 2452 so that mu(d_bus -> pos_u) == 240.0s (4 mins)
    tcum_u = float(pref.Tcum[s + pos_u])
    target_tcum_bus = tcum_u - 240.0
    assert target_tcum_bus > 0.0

    # Interpolate exact d_bus where Tcum == target_tcum_bus
    tcum_slice = pref.Tcum[s : s + pos_u + 1]
    d_slice = transit_graph.route_stop_dists_m[s : s + pos_u + 1]
    d_bus_240 = float(np.interp(target_tcum_bus, tcum_slice, d_slice))

    mu_check, _, _ = pref.get_stats_from_distance(r_idx, d_bus_240, pos_u)
    assert abs(mu_check - 240.0) < 2.0, f"Expected ~240s upstream ETA, got {mu_check:.1f}s"

    # High coverage graph copy (cov_r = 0.90, EH = 1800s) so untracked rate Lambda is very small
    graph_hi_cov = copy.copy(transit_graph)
    cov_copy = transit_graph.cov_r.copy()
    cov_copy[r_idx] = 0.90
    graph_hi_cov.cov_r = cov_copy

    live_d = [np.empty(0, dtype=np.float32) for _ in range(transit_graph.n_routes)]
    live_t = [np.empty(0, dtype=np.float64) for _ in range(transit_graph.n_routes)]
    live_v = [[] for _ in range(transit_graph.n_routes)]
    live_d[r_idx] = np.array([d_bus_240], dtype=np.float32)
    live_t[r_idx] = np.array([t_q], dtype=np.float64)
    live_v[r_idx] = ["DL1PC9999"]

    live_idx_4m = LiveVehicleIndex(
        live_d_m=live_d,
        live_t_obs=live_t,
        live_veh_id=live_v,
        n_vehicles=1,
        is_fresh=True,
        max_t_obs=t_q,
    )

    # 1. Evaluate with live bus 4 mins away (Delta_star = 240s)
    ew_live, var_w_live, _, _, _, lbl_live = compute_leg_wait_and_ride(
        graph_hi_cov,
        pref,
        live_idx_4m,
        u_idx,
        v_idx,
        t_a=t_q,
        t_q=t_q,
        bucket=bucket,
        candidate_routes=[r_idx],
    )
    ew_prior = float(graph_hi_cov.EW_sec[r_idx, bucket])
    assert 220.0 <= ew_live <= 240.5, f"With live bus at 240s and high cov, E[W] ({ew_live:.1f}s) must be ~240s"
    assert ew_live < 0.5 * ew_prior
    assert "Live ETA" in lbl_live

    # 2. Evaluate when bus ETA_b - t_q > 1500s (e.g. t_obs set in future or far upstream so ETA - t_q = 1800s)
    live_t_far = [np.empty(0, dtype=np.float64) for _ in range(transit_graph.n_routes)]
    live_t_far[r_idx] = np.array([t_q + 1600.0], dtype=np.float64)  # ETA_b - t_q = 1840s > 1500s gate
    live_idx_far = LiveVehicleIndex(
        live_d_m=live_d,
        live_t_obs=live_t_far,
        live_veh_id=live_v,
        n_vehicles=1,
        is_fresh=True,
        max_t_obs=t_q,
    )
    ew_fallback, _, _, _, _, lbl_fallback = compute_leg_wait_and_ride(
        graph_hi_cov,
        pref,
        live_idx_far,
        u_idx,
        v_idx,
        t_a=t_q,
        t_q=t_q,
        bucket=bucket,
        candidate_routes=[r_idx],
    )
    assert abs(ew_fallback - ew_prior) < 1e-3, (
        f"When ETA_b - t_q > 1500s, wait must fall back to headway prior {ew_prior:.1f}s, got {ew_fallback:.1f}s"
    )
    assert "Sched wait" in lbl_fallback

    # 3. Verify arrival-order (FIFO) monotonicity: t_a vs t_a + 60s
    ew_ta0, _, _, _, _, _ = compute_leg_wait_and_ride(
        graph_hi_cov,
        pref,
        live_idx_4m,
        u_idx,
        v_idx,
        t_a=t_q,
        t_q=t_q,
        bucket=bucket,
        candidate_routes=[r_idx],
    )
    ew_ta60, _, _, _, _, _ = compute_leg_wait_and_ride(
        graph_hi_cov,
        pref,
        live_idx_4m,
        u_idx,
        v_idx,
        t_a=t_q + 60.0,
        t_q=t_q,
        bucket=bucket,
        candidate_routes=[r_idx],
    )
    dep_ta0 = t_q + ew_ta0
    dep_ta60 = (t_q + 60.0) + ew_ta60
    assert dep_ta60 >= dep_ta0 - 1e-6, (
        f"FIFO violation: arriving 60s later gave earlier expected departure ({dep_ta60} < {dep_ta0})"
    )


def test_4_ban_on_consecutive_walk_legs(raptor_router: CommonLinesRaptor) -> None:
    """Check 4 (Audit Fix 4.6): No returned journey option ever contains WALK -> WALK."""
    t_q = int(datetime(2026, 11, 10, 8, 30, tzinfo=IST_TZ).timestamp())
    od_pairs = [
        ("ISBT Kashmere Gate", "Nehru Place Terminal"),
        ("Connaught Place", "Hauz Khas"),
        ((28.6675, 77.2285), (28.5494, 77.2519)),
    ]

    for orig, dest in od_pairs:
        journeys = raptor_router.plan_journey(origin=orig, destination=dest, departure_time=t_q)
        assert len(journeys) >= 1, f"Expected at least 1 journey for {orig} -> {dest}"
        for j_opt in journeys:
            for idx in range(len(j_opt.legs) - 1):
                leg_a = j_opt.legs[idx]
                leg_b = j_opt.legs[idx + 1]
                assert not (isinstance(leg_a, WalkLeg) and isinstance(leg_b, WalkLeg)), (
                    f"Consecutive WALK -> WALK legs found in {orig} -> {dest}: {leg_a} -> {leg_b}"
                )


def test_5_kashmere_gate_to_nehru_place_diurnal_benchmark(
    raptor_router: CommonLinesRaptor,
) -> None:
    """Check 5 (Problem Statement Section 2.1 & 2.2):
    - Query Kashmere Gate ISBT -> Nehru Place Terminal at 08:30 AM Tuesday vs 11:00 PM Tuesday.
    - 08:30 AM in-vehicle travel time > 11:00 PM in-vehicle travel time and > 05:30 AM Free-Flow.
    - Returns 2-3 sensible, distinct journey options with complete leg breakdowns.
    """
    t_morn = datetime(2026, 11, 10, 8, 30, tzinfo=IST_TZ).timestamp()
    t_night = datetime(2026, 11, 10, 23, 0, tzinfo=IST_TZ).timestamp()

    j_morn = raptor_router.plan_journey(
        origin="ISBT Kashmere Gate",
        destination="Nehru Place Terminal",
        departure_time=t_morn,
    )
    j_night = raptor_router.plan_journey(
        origin="ISBT Kashmere Gate",
        destination="Nehru Place Terminal",
        departure_time=t_night,
    )

    assert 2 <= len(j_morn) <= 3, f"Expected 2-3 distinct options at 08:30 AM, got {len(j_morn)}"
    assert 2 <= len(j_night) <= 3, f"Expected 2-3 distinct options at 11:00 PM, got {len(j_night)}"

    # Compare direct Route 425CLUP (Option 1 in both scenarios)
    morn_opt1 = j_morn[0]
    night_opt1 = j_night[0]

    assert morn_opt1.total_ride_sec > night_opt1.total_ride_sec, (
        f"08:30 AM in-vehicle ride time ({morn_opt1.total_ride_sec/60:.1f}m) must exceed "
        f"11:00 PM in-vehicle ride time ({night_opt1.total_ride_sec/60:.1f}m)"
    )
    assert morn_opt1.traffic_delay_sec > 0.0, "Morning peak traffic delay over 05:30 AM free-flow must be > 0"
    assert morn_opt1.traffic_delay_sec > night_opt1.traffic_delay_sec

    # Verify complete leg breakdown fields on every returned option
    for j_opt in j_morn + j_night:
        assert j_opt.total_duration_sec > 0.0
        assert j_opt.p85_buffer_sec > 0.0
        assert j_opt.free_flow_duration_sec > 0.0
        bus_legs = [l for l in j_opt.legs if isinstance(l, TransitLeg)]
        assert len(bus_legs) >= 1
        for bl in bus_legs:
            assert bl.from_name and bl.to_name
            assert bl.route_label
            assert bl.expected_wait_sec > 0.0
            assert bl.expected_ride_sec > 0.0
            assert bl.free_flow_ride_sec > 0.0


def test_6_end_to_end_routing_latency_under_100ms(
    transit_graph: TransitGraph,
    raptor_router: CommonLinesRaptor,
) -> None:
    """Check 6: Once transit_graph.pkl and prefix arrays are in memory, a full plan_journey
    call completes in < 100 ms on CPU.
    """
    t_q = int(datetime(2026, 11, 10, 8, 30, tzinfo=IST_TZ).timestamp())
    ht = build_precomputed_horizon_table(tracker=raptor_router.tracker, query_epoch=t_q)
    pref = build_route_prefix_arrays(transit_graph, horizon_table=ht, query_time=t_q)

    # Warm call
    _ = raptor_router.plan_journey(
        origin="ISBT Kashmere Gate",
        destination="Nehru Place Terminal",
        departure_time=t_q,
        horizon_table=ht,
        prefix_arrays=pref,
    )

    # Timed benchmark over 3 runs
    latencies_ms = []
    for _ in range(3):
        t0 = time.perf_counter()
        res = raptor_router.plan_journey(
            origin="ISBT Kashmere Gate",
            destination="Nehru Place Terminal",
            departure_time=t_q,
            horizon_table=ht,
            prefix_arrays=pref,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        assert len(res) >= 2

    median_lat_ms = float(np.median(latencies_ms))
    print(f"\n[BENCHMARK] Median in-memory plan_journey latency: {median_lat_ms:.2f} ms (runs: {latencies_ms})")
    assert median_lat_ms < 100.0, f"Routing latency {median_lat_ms:.2f} ms exceeded 100 ms budget"
