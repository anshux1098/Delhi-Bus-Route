"""Time-Dependent Common-Lines RAPTOR Router for Delhi Bus Transit.

Implements Rulings Q1, Q2, and Q3:
1. Origin/Destination Snapping:
   - Resolves stop_id, stop_name substring, or (lat, lon) coordinates within 800m.
   - Initializes Round 0 access walk times and relaxes transfer footpaths.
2. Rounds k = 1, 2, 3 (representing 0, 1, and 2 transfers):
   - Numba-accelerated O(L_r) linear route scan preserving through-riding continuity.
   - Evaluates Chriqui-Robillard Common-Lines attractive sets and closed-form live waits.
   - Enforces strict footpath transfer relaxation banning WALK -> WALK chains.
3. Journey Reconstruction & Lognormal P85 Buffer (Ruling Q1b):
   - Exact journey-level uncertainty aggregation with intra-trip correlation rho = 0.600.
   - s^2 = ln(1 + V_J / mu_J^2), q85_J = mu_J * exp(-0.5 * s^2 + 1.036 * sqrt(s^2)).
   - Diurnal traffic delay comparison against 05:30 AM free-flow travel time.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Ensure generic CPU architecture for Numba on Windows ARM64 / Snapdragon X
os.environ.setdefault("NUMBA_CPU_NAME", "generic")

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from src.ingestion.hygiene import find_project_root
from src.model.historical_baseline import epoch_to_ist_features
from src.model.live_blender import (
    LiveStateTracker,
    PrecomputedHorizonTable,
    build_precomputed_horizon_table,
)
from src.router.transit_graph import (
    LAT0,
    LON0,
    M_PER_DEG_LAT,
    M_PER_DEG_LON,
    N_BUCKETS,
    TransitGraph,
    load_transit_graph,
)
from src.router.wait_estimator import (
    LiveVehicleIndex,
    RoutePrefixArrays,
    build_live_vehicle_index,
    build_route_prefix_arrays,
    compute_leg_wait_and_ride,
)

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = find_project_root()
WALK_SPEED_MPS = 1.25  # 4.5 km/h
WALK_DETOUR_FACTOR = 1.3
IST_TZ = timezone(timedelta(hours=5, minutes=30))


@njit(fastmath=True)
def _bsearch_cl_jit(cl_keys: np.ndarray, key: np.int64) -> int:
    """Binary search in sorted cl_keys for (u * n_stops + v) * 4 + bucket."""
    lo = 0
    hi = len(cl_keys) - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        v = cl_keys[mid]
        if v == key:
            return mid
        elif v < key:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


@njit(fastmath=True)
def _raptor_scan_jit(
    tau: np.ndarray,
    marked_init: np.ndarray,
    dest_mask: np.ndarray,
    n_rounds: int,
    n_stops: int,
    n_routes: int,
    bucket: int,
    route_stop_offsets: np.ndarray,
    route_stops: np.ndarray,
    Tcum: np.ndarray,
    EW_sec: np.ndarray,
    footpath_offsets: np.ndarray,
    footpath_target_stops: np.ndarray,
    footpath_walk_sec: np.ndarray,
    footpath_penalty_sec: np.ndarray,
    cl_keys: np.ndarray,
    cl_wait_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Execute Rounds k = 1..n_rounds of Time-Dependent Common-Lines RAPTOR in < 10ms.

    Each round k computes the earliest k-bus-leg arrival time tau[k, v] at every stop v,
    preserving 0-transfer through-riding continuity (Ruling Q2) and strictly banning
    WALK -> WALK chains (Audit Fix 4.6).
    """
    bus_parent_u = np.full((n_rounds + 1, n_stops), -1, dtype=np.int32)
    bus_parent_r = np.full((n_rounds + 1, n_stops), -1, dtype=np.int32)
    walk_parent_u = np.full((n_rounds + 1, n_stops), -1, dtype=np.int32)
    walk_parent_edge = np.full((n_rounds + 1, n_stops), -1, dtype=np.int32)

    # Record distinct direct (Round 1) hits at destination candidates: (r, board_u, dest_v, arr_t)
    max_hits = 64
    direct_hits = np.zeros((max_hits, 4), dtype=np.float64)
    n_direct_hits = 0

    best_overall = tau[0].copy()
    marked = marked_init.copy()
    best_dest_arr = 1e18

    for k in range(1, n_rounds + 1):
        bus_updated = np.zeros(n_stops, dtype=np.bool_)
        prune_cutoff = best_dest_arr + 1500.0

        for r in range(n_routes):
            s = route_stop_offsets[r]
            e = route_stop_offsets[r + 1]
            L = e - s
            if L <= 1:
                continue

            # Quick check if any stop on route r is marked in round k - 1
            has_marked = False
            for pos in range(L):
                if marked[route_stops[s + pos]]:
                    has_marked = True
                    break
            if not has_marked:
                continue

            ew_r = float(EW_sec[r, bucket])
            best_pos = -1
            best_st = -1
            best_adj_t = 1e18

            latest_pos = -1
            latest_st = -1

            for pos in range(L):
                st = route_stops[s + pos]
                tcum_pos = float(Tcum[s + pos])

                if best_pos >= 0 and pos > best_pos:
                    # Candidate 1: best prefix-adjusted boarding stop (preserves through-riding)
                    mu_1 = tcum_pos - float(Tcum[s + best_pos])
                    t_prev_1 = tau[k - 1, best_st]
                    optimistic_1 = t_prev_1 + mu_1

                    if optimistic_1 < prune_cutoff and (optimistic_1 < tau[k, st] or (k == 1 and dest_mask[st])):
                        arr_1 = optimistic_1 + ew_r
                        chosen_u = best_st

                        # Check Common-Lines wait reduction if optimistic bound can improve
                        if optimistic_1 < best_overall[st] or dest_mask[st]:
                            key1 = (np.int64(best_st) * np.int64(n_stops) + np.int64(st)) * 4 + bucket
                            cl_idx = _bsearch_cl_jit(cl_keys, key1)
                            if cl_idx >= 0:
                                cl_w = float(cl_wait_prior[cl_idx])
                                cand_cl = optimistic_1 + cl_w
                                if cand_cl < arr_1:
                                    arr_1 = cand_cl

                        # Candidate 2: latest marked boarding stop (captures mid-route common-line hubs)
                        if latest_pos >= 0 and latest_pos != best_pos and pos > latest_pos:
                            mu_2 = tcum_pos - float(Tcum[s + latest_pos])
                            t_prev_2 = tau[k - 1, latest_st]
                            optimistic_2 = t_prev_2 + mu_2
                            if optimistic_2 < arr_1:
                                arr_2 = optimistic_2 + ew_r
                                key2 = (np.int64(latest_st) * np.int64(n_stops) + np.int64(st)) * 4 + bucket
                                cl_idx2 = _bsearch_cl_jit(cl_keys, key2)
                                if cl_idx2 >= 0:
                                    cl_w2 = float(cl_wait_prior[cl_idx2])
                                    cand_cl2 = optimistic_2 + cl_w2
                                    if cand_cl2 < arr_2:
                                        arr_2 = cand_cl2
                                if arr_2 < arr_1:
                                    arr_1 = arr_2
                                    chosen_u = latest_st

                        # Record Round 1 direct hit to destination candidate
                        if k == 1 and dest_mask[st] and n_direct_hits < max_hits:
                            direct_hits[n_direct_hits, 0] = float(r)
                            direct_hits[n_direct_hits, 1] = float(chosen_u)
                            direct_hits[n_direct_hits, 2] = float(st)
                            direct_hits[n_direct_hits, 3] = arr_1
                            n_direct_hits += 1

                        if arr_1 < tau[k, st]:
                            tau[k, st] = arr_1
                            bus_parent_u[k, st] = chosen_u
                            bus_parent_r[k, st] = r
                            bus_updated[st] = True
                            if arr_1 < best_overall[st]:
                                best_overall[st] = arr_1
                            if dest_mask[st] and arr_1 < best_dest_arr:
                                best_dest_arr = arr_1

                # Update candidate boarding stops along route r
                if marked[st]:
                    t_prev = tau[k - 1, st]
                    if t_prev < prune_cutoff:
                        prev_r = bus_parent_r[k - 1, st] if walk_parent_u[k - 1, st] < 0 else -1
                        if k == 1 or prev_r != r:
                            adj_t = t_prev - tcum_pos
                            if adj_t < best_adj_t:
                                best_adj_t = adj_t
                                best_pos = pos
                                best_st = st
                            latest_pos = pos
                            latest_st = st

        # Footpath Transfer Relaxation (Strictly banning WALK -> WALK)
        # Only relax footpaths out of stops updated by a BUS leg in Round k
        next_marked = np.zeros(n_stops, dtype=np.bool_)
        prune_cutoff = best_dest_arr + 1500.0
        for u in range(n_stops):
            if not bus_updated[u]:
                continue
            t_u = tau[k, u]
            if t_u >= prune_cutoff:
                continue
            if tau[k, u] <= best_overall[u]:
                next_marked[u] = True
            s_fp = footpath_offsets[u]
            e_fp = footpath_offsets[u + 1]
            for edge in range(s_fp, e_fp):
                v = footpath_target_stops[edge]
                if v == u:
                    continue
                w_sec = float(footpath_walk_sec[edge] + footpath_penalty_sec[edge])
                cand_arr = t_u + w_sec
                if cand_arr < tau[k, v] and cand_arr < prune_cutoff:
                    tau[k, v] = cand_arr
                    walk_parent_u[k, v] = u
                    walk_parent_edge[k, v] = edge
                    if cand_arr < best_overall[v]:
                        best_overall[v] = cand_arr
                        next_marked[v] = True
                    if dest_mask[v] and cand_arr < best_dest_arr:
                        best_dest_arr = cand_arr

        marked = next_marked

    return bus_parent_u, bus_parent_r, walk_parent_u, walk_parent_edge, direct_hits, n_direct_hits


@dataclass
class WalkLeg:
    """Pedestrian walking segment (access, transfer, or egress)."""

    from_name: str
    to_name: str
    from_idx: int
    to_idx: int
    walk_sec: float
    dist_m: float
    leg_type: str = "WALK"  # ACCESS, TRANSFER, EGRESS

    def __repr__(self) -> str:
        mins = self.walk_sec / 60.0
        return f"[Walk {mins:.1f}m ({self.dist_m:.0f}m): {self.from_name} -> {self.to_name}]"


@dataclass
class TransitLeg:
    """Scheduled / real-time bus transit segment."""

    from_name: str
    to_name: str
    from_idx: int
    to_idx: int
    route_idx: int
    route_short_name: str
    candidate_routes: list[str]
    board_epoch: float
    alight_epoch: float
    expected_wait_sec: float
    wait_variance: float
    expected_ride_sec: float
    ride_variance: float
    free_flow_ride_sec: float
    route_label: str
    is_common_lines: bool
    leg_type: str = "BUS"

    def __repr__(self) -> str:
        w_min = self.expected_wait_sec / 60.0
        r_min = self.expected_ride_sec / 60.0
        return (
            f"[Bus {self.route_label} | Wait: {w_min:.1f}m, Ride: {r_min:.1f}m | "
            f"{self.from_name} -> {self.to_name}]"
        )


@dataclass
class JourneyOption:
    """Complete door-to-door transit itinerary."""

    round_k: int
    departure_epoch: float
    arrival_epoch: float
    total_duration_sec: float
    total_walk_sec: float
    total_wait_sec: float
    total_ride_sec: float
    free_flow_duration_sec: float
    traffic_delay_sec: float
    journey_variance: float
    p85_duration_sec: float
    p85_arrival_epoch: float
    p85_buffer_sec: float
    legs: list[WalkLeg | TransitLeg] = field(default_factory=list)

    @property
    def num_transfers(self) -> int:
        bus_legs = sum(1 for leg in self.legs if isinstance(leg, TransitLeg))
        return max(0, bus_legs - 1)

    def summary_str(self) -> str:
        dur_m = self.total_duration_sec / 60.0
        p85_m = self.p85_duration_sec / 60.0
        buf_m = self.p85_buffer_sec / 60.0
        del_m = self.traffic_delay_sec / 60.0
        ff_m = self.free_flow_duration_sec / 60.0
        xfer = self.num_transfers
        return (
            f"Expected: {dur_m:.1f} mins | P85 Buffer: +{buf_m:.1f} mins (P85 total: {p85_m:.1f} mins) | "
            f"Transfers: {xfer} | Congestion Delay: +{del_m:.1f} mins (Free-flow: {ff_m:.1f} mins)"
        )


class CommonLinesRaptor:
    """Time-Dependent Common-Lines RAPTOR router with empirical uncertainty buffers."""

    def __init__(
        self,
        graph: TransitGraph | None = None,
        tracker: LiveStateTracker | None = None,
    ) -> None:
        self.graph = graph or load_transit_graph()
        self.tracker = tracker or LiveStateTracker()
        self.n_stops = int(self.graph.n_stops)
        self.n_routes = int(self.graph.n_routes)

        # Build inverted index: stop_idx -> list of route_idx serving that stop
        self.stop_to_routes: list[list[int]] = [[] for _ in range(self.n_stops)]
        for r in range(self.n_routes):
            s = int(self.graph.route_stop_offsets[r])
            e = int(self.graph.route_stop_offsets[r + 1])
            for st in self.graph.route_stops[s:e]:
                self.stop_to_routes[st].append(r)

        # Stop KDTree for coordinate snapping
        self.stop_coords_xy = np.column_stack([self.graph.stop_x_m, self.graph.stop_y_m])
        self.kdtree = cKDTree(self.stop_coords_xy)

        # Lowercase stop names for substring matching
        self.stop_names_lower = [str(n).lower() for n in self.graph.stop_names]

        # Cache of (PrecomputedHorizonTable, RoutePrefixArrays) by minute epoch
        self._prefix_cache: dict[int, tuple[PrecomputedHorizonTable, RoutePrefixArrays]] = {}

        # Warm up Numba JIT kernels once at initialization so queries execute in < 25ms
        self._warmup_jit()

    def _warmup_jit(self) -> None:
        """Warm up Numba JIT functions using a lightweight dummy 1-round scan."""
        ht = build_precomputed_horizon_table(tracker=self.tracker, query_epoch=1794193200)
        pref = build_route_prefix_arrays(self.graph, horizon_table=ht, query_time=1794193200)
        self._prefix_cache[1794193200 // 60] = (ht, pref)

        tau_dummy = np.full((2, self.n_stops), np.inf, dtype=np.float64)
        marked_dummy = np.zeros(self.n_stops, dtype=np.bool_)
        dest_dummy = np.zeros(self.n_stops, dtype=np.bool_)
        tau_dummy[0, 0] = 1794193200.0
        marked_dummy[0] = True
        dest_dummy[1] = True

        _raptor_scan_jit(
            tau_dummy,
            marked_dummy,
            dest_dummy,
            1,
            self.n_stops,
            self.n_routes,
            1,
            self.graph.route_stop_offsets,
            self.graph.route_stops,
            pref.Tcum,
            self.graph.EW_sec,
            self.graph.footpath_offsets,
            self.graph.footpath_target_stops,
            self.graph.footpath_walk_sec,
            self.graph.footpath_penalty_sec,
            self.graph.cl_keys,
            self.graph.cl_wait_prior,
        )

    def _resolve_candidates(
        self,
        query: str | int | tuple[float, float],
        max_candidates: int = 8,
        max_walk_m: float = 800.0,
    ) -> list[tuple[int, float, float, float, float]]:
        """Resolve query to candidate (stop_idx, walk_sec, dist_m, orig_x, orig_y)."""
        # Case 1: integer stop index or stop_id
        if isinstance(query, (int, np.integer)):
            idx = int(query)
            if 0 <= idx < self.n_stops:
                return [(idx, 0.0, 0.0, float(self.graph.stop_x_m[idx]), float(self.graph.stop_y_m[idx]))]
            s_idx = self.graph.stop_id_to_idx.get(str(idx))
            if s_idx is not None:
                return [(s_idx, 0.0, 0.0, float(self.graph.stop_x_m[s_idx]), float(self.graph.stop_y_m[s_idx]))]

        # Case 2: tuple of (lat, lon)
        if isinstance(query, (tuple, list)) and len(query) == 2:
            lat, lon = float(query[0]), float(query[1])
            x_m = (lon - LON0) * M_PER_DEG_LON
            y_m = (lat - LAT0) * M_PER_DEG_LAT
            idxs = self.kdtree.query_ball_point([x_m, y_m], r=max_walk_m)
            if not idxs:
                _, nearest = self.kdtree.query([x_m, y_m])
                idxs = [int(nearest)]
            candidates = []
            for idx in idxs:
                dist = float(np.hypot(self.graph.stop_x_m[idx] - x_m, self.graph.stop_y_m[idx] - y_m))
                walk_sec = (dist * WALK_DETOUR_FACTOR) / WALK_SPEED_MPS
                candidates.append((int(idx), walk_sec, dist, x_m, y_m))
            candidates.sort(key=lambda c: c[2])
            return candidates[:max_candidates]

        # Case 3: string (stop_id, numeric stop_idx, or stop_name substring)
        q_str = str(query).strip()
        if q_str.isdigit():
            cands: list[tuple[int, float, float, float, float]] = []
            idx_val = int(q_str)
            if 0 <= idx_val < self.n_stops:
                cands.append(
                    (idx_val, 0.0, 0.0, float(self.graph.stop_x_m[idx_val]), float(self.graph.stop_y_m[idx_val]))
                )
            if q_str in self.graph.stop_id_to_idx:
                s_idx = self.graph.stop_id_to_idx[q_str]
                if s_idx != idx_val:
                    cands.append(
                        (s_idx, 0.0, 0.0, float(self.graph.stop_x_m[s_idx]), float(self.graph.stop_y_m[s_idx]))
                    )
            if cands:
                return cands

        if q_str in self.graph.stop_id_to_idx:
            s_idx = self.graph.stop_id_to_idx[q_str]
            return [(s_idx, 0.0, 0.0, float(self.graph.stop_x_m[s_idx]), float(self.graph.stop_y_m[s_idx]))]

        # Substring search in stop names
        q_lower = q_str.lower()
        exact_matches = [i for i, name in enumerate(self.stop_names_lower) if q_lower == name]
        starts_matches = [i for i, name in enumerate(self.stop_names_lower) if name.startswith(q_lower)]
        contains_matches = [i for i, name in enumerate(self.stop_names_lower) if q_lower in name]

        all_matches: list[int] = []
        seen: set[int] = set()
        for idx_list in (exact_matches, starts_matches, contains_matches):
            for i in idx_list:
                if i not in seen:
                    all_matches.append(i)
                    seen.add(i)

        if not all_matches:
            # Strip common generic hub suffixes like 'isbt' or 'terminal' if needed
            tokens = [t for t in q_lower.replace("(", " ").replace(")", " ").split() if t not in ("isbt", "bus", "terminal")]
            if not tokens:
                tokens = q_lower.split()
            token_matches = [
                i for i, name in enumerate(self.stop_names_lower)
                if all(tok in name for tok in tokens)
            ]
            for i in token_matches:
                if i not in seen:
                    all_matches.append(i)
                    seen.add(i)

        if not all_matches:
            raise ValueError(f"No transit stop found matching '{query}'")

        primary_idx = all_matches[0]
        primary_x = float(self.graph.stop_x_m[primary_idx])
        primary_y = float(self.graph.stop_y_m[primary_idx])

        # Also include nearby hub platform stops within 350m of the primary matched stop
        nearby_idxs = self.kdtree.query_ball_point([primary_x, primary_y], r=350.0)
        for ni in nearby_idxs:
            if int(ni) not in seen:
                all_matches.append(int(ni))
                seen.add(int(ni))

        candidates = []
        for i in all_matches:
            dist = float(np.hypot(self.graph.stop_x_m[i] - primary_x, self.graph.stop_y_m[i] - primary_y))
            if dist <= max_walk_m:
                walk_sec = (dist * WALK_DETOUR_FACTOR) / WALK_SPEED_MPS if i != primary_idx else 0.0
                candidates.append((i, walk_sec, dist, primary_x, primary_y))

        candidates.sort(key=lambda c: c[2])
        return candidates[:max_candidates]

    def _build_journey_from_bus_legs(
        self,
        bus_legs_spec: list[tuple[int, int, int]],
        round_k: int,
        t_dep: float,
        bucket: int,
        origin_label: str,
        dest_label: str,
        orig_x: float,
        orig_y: float,
        dest_x: float,
        dest_y: float,
        is_exact_stop_origin: bool,
        is_exact_stop_dest: bool,
        prefix_arrays: RoutePrefixArrays,
        live_index: LiveVehicleIndex | None,
    ) -> JourneyOption | None:
        """Construct a JourneyOption from a sequence of (u_board, v_alight, r_idx) bus legs.

        Strictly enforces the ban on WALK -> WALK chains:
        - At most ONE ACCESS WalkLeg before the first bus leg.
        - At most ONE TRANSFER WalkLeg between consecutive bus legs.
        - At most ONE EGRESS WalkLeg after the final bus leg.
        """
        if not bus_legs_spec:
            return None

        legs: list[WalkLeg | TransitLeg] = []
        curr_t = t_dep

        # 1. Optional single ACCESS walk to first boarding stop
        first_board = bus_legs_spec[0][0]
        if not is_exact_stop_origin:
            acc_dist = float(
                np.hypot(self.graph.stop_x_m[first_board] - orig_x, self.graph.stop_y_m[first_board] - orig_y)
            )
            if acc_dist > 5.0:
                acc_sec = (acc_dist * WALK_DETOUR_FACTOR) / WALK_SPEED_MPS
                legs.append(
                    WalkLeg(
                        from_name=origin_label,
                        to_name=str(self.graph.stop_names[first_board]),
                        from_idx=-1,
                        to_idx=first_board,
                        walk_sec=acc_sec,
                        dist_m=acc_dist,
                        leg_type="ACCESS",
                    )
                )
                curr_t += acc_sec

        # 2. Bus legs + optional single TRANSFER walk between bus legs
        for leg_i, (u_st, v_st, r_idx) in enumerate(bus_legs_spec):
            if leg_i > 0:
                prev_alight = bus_legs_spec[leg_i - 1][1]
                if prev_alight != u_st:
                    xfer_dist = float(
                        np.hypot(
                            self.graph.stop_x_m[u_st] - self.graph.stop_x_m[prev_alight],
                            self.graph.stop_y_m[u_st] - self.graph.stop_y_m[prev_alight],
                        )
                    )
                    xfer_sec = max(60.0, (xfer_dist * WALK_DETOUR_FACTOR) / WALK_SPEED_MPS)
                    legs.append(
                        WalkLeg(
                            from_name=str(self.graph.stop_names[prev_alight]),
                            to_name=str(self.graph.stop_names[u_st]),
                            from_idx=prev_alight,
                            to_idx=u_st,
                            walk_sec=xfer_sec,
                            dist_m=max(20.0, xfer_dist),
                            leg_type="TRANSFER",
                        )
                    )
                    curr_t += xfer_sec

            # Look up Common-Lines attractive set A(u_st, v_st)
            cl_info = self.graph.lookup_common_lines(u_st, v_st, bucket)
            if cl_info is not None and r_idx in cl_info["routes"]:
                cand_routes = list(cl_info["routes"])
                att_info = cl_info
            else:
                cand_routes = [r_idx]
                att_info = None

            e_w, v_w, mu_r, v_r, ff_r, lbl = compute_leg_wait_and_ride(
                self.graph,
                prefix_arrays,
                live_index,
                u_st,
                v_st,
                curr_t,
                t_dep,
                bucket,
                candidate_routes=cand_routes,
                attractive_set_info=att_info,
            )
            if mu_r >= 90000.0:
                return None

            cand_names = [str(self.graph.route_short_names[cr]) for cr in cand_routes]
            r_name = str(self.graph.route_short_names[r_idx])
            board_ep = curr_t + e_w
            alight_ep = board_ep + mu_r

            legs.append(
                TransitLeg(
                    from_name=str(self.graph.stop_names[u_st]),
                    to_name=str(self.graph.stop_names[v_st]),
                    from_idx=u_st,
                    to_idx=v_st,
                    route_idx=r_idx,
                    route_short_name=r_name,
                    candidate_routes=cand_names,
                    board_epoch=float(board_ep),
                    alight_epoch=float(alight_ep),
                    expected_wait_sec=float(e_w),
                    wait_variance=float(v_w),
                    expected_ride_sec=float(mu_r),
                    ride_variance=float(v_r),
                    free_flow_ride_sec=float(ff_r),
                    route_label=lbl,
                    is_common_lines=(len(cand_routes) > 1),
                )
            )
            curr_t = alight_ep

        # 3. Optional single EGRESS walk from final alighting stop to destination
        last_alight = bus_legs_spec[-1][1]
        if not is_exact_stop_dest:
            egr_dist = float(
                np.hypot(self.graph.stop_x_m[last_alight] - dest_x, self.graph.stop_y_m[last_alight] - dest_y)
            )
            if egr_dist > 5.0:
                egr_sec = (egr_dist * WALK_DETOUR_FACTOR) / WALK_SPEED_MPS
                legs.append(
                    WalkLeg(
                        from_name=str(self.graph.stop_names[last_alight]),
                        to_name=dest_label,
                        from_idx=last_alight,
                        to_idx=-1,
                        walk_sec=egr_sec,
                        dist_m=egr_dist,
                        leg_type="EGRESS",
                    )
                )
                curr_t += egr_sec

        # Aggregate journey statistics
        tot_walk = sum(l.walk_sec for l in legs if isinstance(l, WalkLeg))
        tot_wait = sum(l.expected_wait_sec for l in legs if isinstance(l, TransitLeg))
        tot_ride = sum(l.expected_ride_sec for l in legs if isinstance(l, TransitLeg))
        tot_ff = sum(l.free_flow_ride_sec for l in legs if isinstance(l, TransitLeg))

        mu_J = float(tot_walk + tot_wait + tot_ride)
        V_J = float(max(100.0, sum(l.wait_variance + l.ride_variance for l in legs if isinstance(l, TransitLeg))))

        # Lognormal P85 arrival buffer (Ruling Q1b)
        s2 = math.log(1.0 + V_J / (mu_J**2))
        q85_raw = mu_J * math.exp(-0.5 * s2 + 1.036 * math.sqrt(s2))
        q85_J = max(mu_J * 1.03, q85_raw)
        p85_buf = q85_J - mu_J

        ff_dur = tot_walk + tot_wait + tot_ff
        traf_delay = max(0.0, mu_J - ff_dur)

        return JourneyOption(
            round_k=round_k,
            departure_epoch=t_dep,
            arrival_epoch=t_dep + mu_J,
            total_duration_sec=mu_J,
            total_walk_sec=tot_walk,
            total_wait_sec=tot_wait,
            total_ride_sec=tot_ride,
            free_flow_duration_sec=ff_dur,
            traffic_delay_sec=traf_delay,
            journey_variance=V_J,
            p85_duration_sec=q85_J,
            p85_arrival_epoch=t_dep + q85_J,
            p85_buffer_sec=p85_buf,
            legs=legs,
        )

    def route(
        self,
        origin: str | int | tuple[float, float],
        destination: str | int | tuple[float, float],
        departure_time: float | int | str | datetime | None = None,
        max_transfers: int = 2,
        horizon_table: PrecomputedHorizonTable | None = None,
        prefix_arrays: RoutePrefixArrays | None = None,
        live_index: LiveVehicleIndex | None = None,
    ) -> list[JourneyOption]:
        """Find Pareto-optimal transit journeys across Rounds k = 1, 2, 3 in < 100ms."""
        # Parse departure timestamp
        if departure_time is None:
            t_dep = time.time()
        elif isinstance(departure_time, (int, float)):
            t_dep = float(departure_time)
        elif isinstance(departure_time, datetime):
            t_dep = departure_time.timestamp()
        elif isinstance(departure_time, str):
            try:
                dt = datetime.fromisoformat(departure_time)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=IST_TZ)
                t_dep = dt.timestamp()
            except Exception:
                try:
                    dt = datetime.strptime(departure_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST_TZ)
                    t_dep = dt.timestamp()
                except Exception:
                    t_dep = time.time()
        else:
            t_dep = time.time()

        # Determine IST diurnal bucket (0..3)
        ist_feat = epoch_to_ist_features(t_dep)
        h_float = float(ist_feat["hour_float"])
        if h_float < 6.0:
            bucket = 0
        elif h_float < 11.0:
            bucket = 1
        elif h_float < 17.0:
            bucket = 2
        else:
            bucket = 3

        # Retrieve or build Horizon Table & Route Prefix Arrays
        min_key = int(t_dep) // 60
        if prefix_arrays is None:
            if horizon_table is not None:
                prefix_arrays = build_route_prefix_arrays(
                    self.graph, horizon_table=horizon_table, query_time=t_dep
                )
            elif min_key in self._prefix_cache:
                horizon_table, prefix_arrays = self._prefix_cache[min_key]
            else:
                horizon_table = build_precomputed_horizon_table(
                    tracker=self.tracker, query_epoch=int(t_dep)
                )
                prefix_arrays = build_route_prefix_arrays(
                    self.graph, horizon_table=horizon_table, query_time=t_dep
                )
                self._prefix_cache[min_key] = (horizon_table, prefix_arrays)

        # Resolve origin & destination candidate stops
        orig_cands = self._resolve_candidates(origin)
        dest_cands = self._resolve_candidates(destination)

        orig_x, orig_y = orig_cands[0][3], orig_cands[0][4]
        dest_x, dest_y = dest_cands[0][3], dest_cands[0][4]
        is_exact_orig = isinstance(origin, (int, np.integer)) or (
            isinstance(origin, str)
            and (origin.strip().isdigit() or origin.strip() in self.graph.stop_id_to_idx)
        )
        is_exact_dest = isinstance(destination, (int, np.integer)) or (
            isinstance(destination, str)
            and (destination.strip().isdigit() or destination.strip() in self.graph.stop_id_to_idx)
        )
        orig_label = (
            str(self.graph.stop_names[orig_cands[0][0]])
            if is_exact_orig
            else (str(origin) if isinstance(origin, str) else "Origin")
        )
        dest_label = (
            str(self.graph.stop_names[dest_cands[0][0]])
            if is_exact_dest
            else (str(destination) if isinstance(destination, str) else "Destination")
        )

        # Initialize Round 0 arrival times
        n_rounds = max(1, min(3, int(max_transfers) + 1))
        tau = np.full((n_rounds + 1, self.n_stops), np.inf, dtype=np.float64)
        marked_init = np.zeros(self.n_stops, dtype=np.bool_)
        dest_mask = np.zeros(self.n_stops, dtype=np.bool_)

        for dc_idx, _, _, _, _ in dest_cands:
            dest_mask[dc_idx] = True

        for u, w_sec, _, _, _ in orig_cands:
            t_arr = t_dep + w_sec
            if t_arr < tau[0, u]:
                tau[0, u] = t_arr
                marked_init[u] = True

        # Relax initial footpaths from primary origin stops
        for u, _, _, _, _ in orig_cands[:3]:
            t_u = tau[0, u]
            s_fp = int(self.graph.footpath_offsets[u])
            e_fp = int(self.graph.footpath_offsets[u + 1])
            for edge in range(s_fp, e_fp):
                v = int(self.graph.footpath_target_stops[edge])
                if v == u:
                    continue
                w_sec = float(self.graph.footpath_walk_sec[edge])
                if t_u + w_sec < tau[0, v]:
                    tau[0, v] = t_u + w_sec
                    marked_init[v] = True

        # Run compiled Numba RAPTOR scan
        b_u, b_r, w_u, w_e, direct_hits, n_direct = _raptor_scan_jit(
            tau,
            marked_init,
            dest_mask,
            n_rounds,
            self.n_stops,
            self.n_routes,
            bucket,
            self.graph.route_stop_offsets,
            self.graph.route_stops,
            prefix_arrays.Tcum,
            self.graph.EW_sec,
            self.graph.footpath_offsets,
            self.graph.footpath_target_stops,
            self.graph.footpath_walk_sec,
            self.graph.footpath_penalty_sec,
            self.graph.cl_keys,
            self.graph.cl_wait_prior,
        )

        # Collect candidate bus-leg specifications: (round_k, est_arr, [(u, v, r), ...])
        cand_specs: list[tuple[int, float, list[tuple[int, int, int]]]] = []

        # 1. Direct hits from Round 1
        for h in range(n_direct):
            r_idx = int(direct_hits[h, 0])
            u_st = int(direct_hits[h, 1])
            v_st = int(direct_hits[h, 2])
            arr_t = float(direct_hits[h, 3])
            egr_d = float(np.hypot(self.graph.stop_x_m[v_st] - dest_x, self.graph.stop_y_m[v_st] - dest_y))
            egr_s = 0.0 if is_exact_dest else (egr_d * WALK_DETOUR_FACTOR) / WALK_SPEED_MPS
            cand_specs.append((1, arr_t + egr_s, [(u_st, v_st, r_idx)]))

        # 2. Backtrack best k-leg journeys from destination candidates for k = 1..n_rounds
        for k in range(1, n_rounds + 1):
            for dest_idx, egr_sec, _, _, _ in dest_cands:
                if not np.isfinite(tau[k, dest_idx]):
                    continue
                curr = dest_idx
                curr_k = k
                spec: list[tuple[int, int, int]] = []
                valid = True
                while curr_k > 0:
                    if w_u[curr_k, curr] >= 0:
                        curr = int(w_u[curr_k, curr])
                    u_board = int(b_u[curr_k, curr])
                    r_used = int(b_r[curr_k, curr])
                    if u_board < 0 or r_used < 0:
                        valid = False
                        break
                    spec.append((u_board, curr, r_used))
                    curr = u_board
                    curr_k -= 1
                if valid and len(spec) == k:
                    spec.reverse()
                    cand_specs.append((k, float(tau[k, dest_idx]) + egr_sec, spec))

        # Sort candidates by estimated arrival time within each round, then reconstruct top distinct options
        cand_specs.sort(key=lambda x: (x[0], x[1]))

        journeys: list[JourneyOption] = []
        seen_corridors: set[str] = set()
        direct_count = 0

        for k, _, spec in cand_specs:
            if k == 1 and direct_count >= 2 and any(cs[0] >= 2 for cs in cand_specs):
                continue

            # Build corridor signature based on route/attractive-set labels
            corridor_parts = []
            for u_st, v_st, r_idx in spec:
                cl = self.graph.lookup_common_lines(u_st, v_st, bucket)
                if cl is not None and r_idx in cl["routes"]:
                    c_key = "/".join(str(self.graph.route_short_names[cr]) for cr in cl["routes"][:3])
                else:
                    c_key = str(self.graph.route_short_names[r_idx])
                corridor_parts.append(c_key)
            sig = " -> ".join(corridor_parts)
            if sig in seen_corridors:
                continue

            j_opt = self._build_journey_from_bus_legs(
                spec,
                k,
                t_dep,
                bucket,
                orig_label,
                dest_label,
                orig_x,
                orig_y,
                dest_x,
                dest_y,
                is_exact_orig,
                is_exact_dest,
                prefix_arrays,
                live_index,
            )
            if j_opt is not None:
                seen_corridors.add(sig)
                journeys.append(j_opt)
                if k == 1:
                    direct_count += 1
                if len(journeys) >= 3:
                    break

        journeys.sort(key=lambda j: (j.num_transfers, j.total_duration_sec))
        return journeys

    def plan_journey(
        self,
        origin: str | int | tuple[float, float],
        destination: str | int | tuple[float, float],
        departure_time: float | int | str | datetime | None = None,
        max_transfers: int = 2,
        horizon_table: PrecomputedHorizonTable | None = None,
        prefix_arrays: RoutePrefixArrays | None = None,
        live_index: LiveVehicleIndex | None = None,
    ) -> list[JourneyOption]:
        """Alias for route() matching standard planner interface."""
        return self.route(
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            max_transfers=max_transfers,
            horizon_table=horizon_table,
            prefix_arrays=prefix_arrays,
            live_index=live_index,
        )


def run_demonstration() -> None:
    """CLI demonstration comparing Kashmere Gate ISBT to Nehru Place Terminal at 08:30 AM vs 11:00 PM."""
    print("=" * 80)
    print("DELHI TRANSIT PLANNER — TIME-DEPENDENT COMMON-LINES RAPTOR")
    print("=" * 80)

    t0_load = time.perf_counter()
    graph = load_transit_graph()
    raptor = CommonLinesRaptor(graph=graph)
    t1_load = time.perf_counter()
    print(f"[OK] Loaded TransitGraph & CommonLinesRaptor (with Numba warmup) in {(t1_load - t0_load)*1000:.2f}ms\n")

    origin = "ISBT Kashmere Gate"
    dest = "Nehru Place Terminal"

    # Reference weekday: Tuesday, 2026-11-10
    morn_dt = datetime(2026, 11, 10, 8, 30, tzinfo=IST_TZ)
    night_dt = datetime(2026, 11, 10, 23, 0, tzinfo=IST_TZ)

    scenarios = [
        ("MORNING PEAK (08:30 AM IST, Tuesday Weekday)", morn_dt),
        ("LATE NIGHT (11:00 PM IST, Tuesday Night)", night_dt),
    ]

    for title, dt in scenarios:
        epoch = dt.timestamp()
        print("-" * 80)
        print(f"SCENARIO: {title}")
        print(f"Origin     : {origin}")
        print(f"Destination: {dest}")
        print(f"Query Time : {dt.strftime('%Y-%m-%d %H:%M:%S IST')} (Epoch {int(epoch)})")
        print("-" * 80)

        # Precompute prefix arrays once to measure pure in-memory routing latency
        ht = build_precomputed_horizon_table(tracker=raptor.tracker, query_epoch=int(epoch))
        pref = build_route_prefix_arrays(graph, horizon_table=ht, query_time=epoch)

        t_start = time.perf_counter()
        journeys = raptor.plan_journey(
            origin=origin,
            destination=dest,
            departure_time=epoch,
            horizon_table=ht,
            prefix_arrays=pref,
        )
        t_lat_ms = (time.perf_counter() - t_start) * 1000.0

        print(f"[OK] Routing completed in {t_lat_ms:.2f} ms (Target: < 100 ms) | Found {len(journeys)} option(s)\n")

        for idx, j in enumerate(journeys[:3], 1):
            dep_str = datetime.fromtimestamp(j.departure_epoch, tz=IST_TZ).strftime("%H:%M")
            arr_str = datetime.fromtimestamp(j.arrival_epoch, tz=IST_TZ).strftime("%H:%M")
            p85_str = datetime.fromtimestamp(j.p85_arrival_epoch, tz=IST_TZ).strftime("%H:%M")

            print(f"Option {idx}: [{dep_str} -> {arr_str} (P85: {p85_str})] — {j.summary_str()}")
            for leg in j.legs:
                if isinstance(leg, WalkLeg):
                    print(f"   Walk ({leg.leg_type}): {leg.from_name} -> {leg.to_name} ({leg.dist_m:.0f}m, {leg.walk_sec/60:.1f} mins)")
                elif isinstance(leg, TransitLeg):
                    cl_tag = "[COMMON LINES] " if leg.is_common_lines else ""
                    print(
                        f"   Bus : {cl_tag}{leg.route_label}\n"
                        f"         From: {leg.from_name} -> To: {leg.to_name}\n"
                        f"         Wait: {leg.expected_wait_sec/60:.1f}m | Ride: {leg.expected_ride_sec/60:.1f}m "
                        f"(Free-flow ride: {leg.free_flow_ride_sec/60:.1f}m)"
                    )
            print()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Delhi Transit Time-Dependent Common-Lines RAPTOR")
    parser.add_argument("--origin", default=None, help="Origin stop name, ID, or lat,lon")
    parser.add_argument("--destination", default=None, help="Destination stop name, ID, or lat,lon")
    parser.add_argument("--time", default=None, help="Departure time (ISO format or epoch)")
    args = parser.parse_args()

    if args.origin and args.destination:
        graph = load_transit_graph()
        raptor = CommonLinesRaptor(graph=graph)
        journeys = raptor.plan_journey(
            origin=args.origin,
            destination=args.destination,
            departure_time=args.time,
        )
        print(f"Found {len(journeys)} journey options:")
        for i, j in enumerate(journeys, 1):
            print(f"Option {i}: {j.summary_str()}")
            for leg in j.legs:
                print(f"  {leg}")
        return 0

    run_demonstration()
    return 0


if __name__ == "__main__":
    sys.exit(main())
