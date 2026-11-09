"""Prefix Array & Live-ETA Wait Estimator for Delhi Bus Transit.

Implements Rulings Q1, Q2, and Q3:
1. Route Prefix Arrays (build_route_prefix_arrays):
   - Flat float32 prefix arrays (Tcum, S1, S2, FFcum) across all route stops.
   - O(1) exact expected travel time, free-flow time, and intra-trip correlated
     variance:
     V_leg = (1.0 - rho) * (S2[j] - S2[i]) + rho * (S1[j] - S1[i])**2
     (with rho = 0.600 from Sprint 2).
2. Live Upstream Vehicle Index (build_live_vehicle_index):
   - Snaps real-time vehicle positions from data/live_buffer/latest_vehicles.parquet
     onto their route shape, keeping sorted (d_m, t_obs) arrays per route.
3. Closed-Form Wait & Ride Estimator (compute_leg_wait_and_ride):
   - Combines gated live buses (Delta_star = min(ETA_b - t_a)) with untracked/prior
     arrival rate Lambda.
   - Gating rules: 0 <= ETA_b - t_q <= 1500s, sigma_b <= 0.5 * EH_m, catch_prob >= 0.5.
   - Closed-form exponential wait expectation, second moment, and blended ride stats.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure generic CPU architecture for Numba on Windows ARM64 / Snapdragon X
os.environ.setdefault("NUMBA_CPU_NAME", "generic")

import numpy as np
import polars as pl
from numba import njit

from src.ingestion.hygiene import find_project_root, load_and_clean_pings
from src.matching.map_matcher import match_pings
from src.matching.shape_indexer import StaticIndex, load_static_index
from src.model.live_blender import PrecomputedHorizonTable, build_precomputed_horizon_table
from src.router.transit_graph import TransitGraph, load_transit_graph

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = find_project_root()
LATEST_VEHICLES_PATH = PROJECT_ROOT / "data" / "live_buffer" / "latest_vehicles.parquet"
RHO_EMPIRICAL = 0.600  # From Sprint 2 calibration


@njit(fastmath=True)
def _compute_prefix_arrays_jit(
    offsets: np.ndarray,
    corridor_idxs: np.ndarray,
    mat: np.ndarray,
    sig: np.ndarray,
    ff: np.ndarray,
    n_routes: int,
    n_pos: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute flat prefix arrays over all route stop positions in < 2ms."""
    Tcum = np.zeros(n_pos, dtype=np.float32)
    S1 = np.zeros(n_pos, dtype=np.float32)
    S2 = np.zeros(n_pos, dtype=np.float32)
    FFcum = np.zeros(n_pos, dtype=np.float32)

    for r in range(n_routes):
        s = offsets[r]
        e = offsets[r + 1]
        t = np.float32(0.0)
        s1 = np.float32(0.0)
        s2 = np.float32(0.0)
        f = np.float32(0.0)
        for k in range(s, e - 1):
            c = corridor_idxs[k]
            if c >= 0:
                mu_e = mat[c]
                sig_e = sig[c]
                ff_e = ff[c]
            else:
                mu_e = np.float32(60.0)
                sig_e = np.float32(10.0)
                ff_e = np.float32(60.0)
            t += mu_e
            s1 += sig_e
            s2 += sig_e * sig_e
            f += ff_e
            Tcum[k + 1] = t
            S1[k + 1] = s1
            S2[k + 1] = s2
            FFcum[k + 1] = f

    return Tcum, S1, S2, FFcum


@dataclass
class RoutePrefixArrays:
    """Precomputed flat prefix arrays for all route stop sequences."""

    Tcum: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    FFcum: np.ndarray
    offsets: np.ndarray
    stop_dists_m: np.ndarray
    rho: float = RHO_EMPIRICAL

    def get_leg_stats(
        self,
        route_idx: int,
        stop_pos_i: int,
        stop_pos_j: int,
    ) -> tuple[float, float, float]:
        """O(1) exact subpath statistics between stop positions i and j (i < j).

        Returns:
            (mu_leg, ff_leg, V_leg)
        """
        off = int(self.offsets[route_idx])
        idx_i = off + int(stop_pos_i)
        idx_j = off + int(stop_pos_j)

        mu_leg = float(max(0.0, self.Tcum[idx_j] - self.Tcum[idx_i]))
        ff_leg = float(max(0.0, self.FFcum[idx_j] - self.FFcum[idx_i]))
        s1_diff = float(max(0.0, self.S1[idx_j] - self.S1[idx_i]))
        s2_diff = float(max(0.0, self.S2[idx_j] - self.S2[idx_i]))

        V_leg = float(max(1.0, (1.0 - self.rho) * s2_diff + self.rho * (s1_diff**2)))
        return mu_leg, ff_leg, V_leg

    def get_stats_from_distance(
        self,
        route_idx: int,
        d_bus: float,
        stop_pos_u: int,
    ) -> tuple[float, float, float]:
        """Linearly interpolate prefix statistics from a bus at d_bus to stop_pos_u.

        Returns:
            (mu_to_u, ff_to_u, V_to_u)
        """
        off = int(self.offsets[route_idx])
        idx_u = off + int(stop_pos_u)
        d_u = float(self.stop_dists_m[idx_u])

        if d_bus >= d_u:
            return 0.0, 0.0, 1.0

        # Scan for segment k such that d_k <= d_bus <= d_{k+1}
        # bounded up to stop_pos_u
        d_slice = self.stop_dists_m[off : idx_u + 1]
        k = int(np.searchsorted(d_slice, d_bus, side="right")) - 1
        k = max(0, min(int(stop_pos_u) - 1, k))

        idx_k = off + k
        idx_k1 = off + k + 1
        d_k = float(self.stop_dists_m[idx_k])
        d_k1 = float(self.stop_dists_m[idx_k1])
        denom = max(1e-3, d_k1 - d_k)
        frac = float(np.clip((d_bus - d_k) / denom, 0.0, 1.0))

        t_bus = float(self.Tcum[idx_k] + frac * (self.Tcum[idx_k1] - self.Tcum[idx_k]))
        s1_bus = float(self.S1[idx_k] + frac * (self.S1[idx_k1] - self.S1[idx_k]))
        s2_bus = float(self.S2[idx_k] + frac * (self.S2[idx_k1] - self.S2[idx_k]))
        ff_bus = float(self.FFcum[idx_k] + frac * (self.FFcum[idx_k1] - self.FFcum[idx_k]))

        mu = float(max(0.0, self.Tcum[idx_u] - t_bus))
        ff = float(max(0.0, self.FFcum[idx_u] - ff_bus))
        ds1 = float(max(0.0, self.S1[idx_u] - s1_bus))
        ds2 = float(max(0.0, self.S2[idx_u] - s2_bus))
        V = float(max(1.0, (1.0 - self.rho) * ds2 + self.rho * (ds1**2)))

        return mu, ff, V


def build_route_prefix_arrays(
    graph: TransitGraph,
    horizon_table: PrecomputedHorizonTable | None = None,
    query_time: float | int | None = None,
    rho: float = RHO_EMPIRICAL,
) -> RoutePrefixArrays:
    """Build flat prefix arrays for all route stop sequences.

    Evaluated against the first bin (or query_time) of the horizon table.
    """
    if horizon_table is None:
        q_epoch = int(query_time if query_time is not None else time.time())
        horizon_table = build_precomputed_horizon_table(query_epoch=q_epoch)

    mat = np.ascontiguousarray(horizon_table.matrix[:, 0], dtype=np.float32)
    sig = np.ascontiguousarray(horizon_table.sigma_matrix[:, 0], dtype=np.float32)
    ff = np.ascontiguousarray(horizon_table.free_flow_times, dtype=np.float32)
    offsets = np.ascontiguousarray(graph.route_stop_offsets, dtype=np.int32)
    corridor_idxs = np.ascontiguousarray(graph.route_corridor_idxs, dtype=np.int32)

    Tcum, S1, S2, FFcum = _compute_prefix_arrays_jit(
        offsets,
        corridor_idxs,
        mat,
        sig,
        ff,
        int(graph.n_routes),
        len(graph.route_stops),
    )

    return RoutePrefixArrays(
        Tcum=Tcum,
        S1=S1,
        S2=S2,
        FFcum=FFcum,
        offsets=offsets,
        stop_dists_m=graph.route_stop_dists_m,
        rho=rho,
    )


@dataclass
class LiveVehicleIndex:
    """Active live bus locations grouped and sorted per route_idx."""

    live_d_m: list[np.ndarray]
    live_t_obs: list[np.ndarray]
    live_veh_id: list[list[str]]
    n_vehicles: int
    is_fresh: bool
    max_t_obs: float


def build_live_vehicle_index(
    graph: TransitGraph,
    parquet_path: Path | None = None,
    query_time: float | int | None = None,
    max_age_sec: float = 900.0,  # 15 minutes
    static_index: StaticIndex | None = None,
) -> LiveVehicleIndex:
    """Snap latest vehicles to route shapes and index them for upstream queries."""
    p = parquet_path or LATEST_VEHICLES_PATH
    empty_index = LiveVehicleIndex(
        live_d_m=[np.empty(0, dtype=np.float32) for _ in range(graph.n_routes)],
        live_t_obs=[np.empty(0, dtype=np.float64) for _ in range(graph.n_routes)],
        live_veh_id=[[] for _ in range(graph.n_routes)],
        n_vehicles=0,
        is_fresh=False,
        max_t_obs=0.0,
    )

    if not p.exists():
        return empty_index

    try:
        df = pl.read_parquet(p)
        if df.height == 0:
            return empty_index

        # Clean and filter hygiene
        cleaned = load_and_clean_pings(df)
        if cleaned.height == 0:
            return empty_index

        max_ts = float(cleaned["timestamp"].max())
        q_time = float(query_time if query_time is not None else max_ts)

        # Check freshness: age relative to query timestamp
        is_fresh = abs(q_time - max_ts) <= max_age_sec

        s_index = static_index or load_static_index()
        matched = match_pings(cleaned, s_index)

        # Filter out off-route pings
        valid = matched.filter(
            (~pl.col("is_off_route")) & pl.col("d_m").is_not_null()
        )

        n_routes = graph.n_routes
        d_lists: list[list[float]] = [[] for _ in range(n_routes)]
        t_lists: list[list[float]] = [[] for _ in range(n_routes)]
        v_lists: list[list[str]] = [[] for _ in range(n_routes)]

        for row in valid.select("vehicle_id", "route_id", "timestamp", "d_m").iter_rows(named=True):
            rid = str(row["route_id"])
            ridx = graph.route_id_to_idx.get(rid)
            if ridx is not None and 0 <= ridx < n_routes:
                d_lists[ridx].append(float(row["d_m"]))
                t_lists[ridx].append(float(row["timestamp"]))
                v_lists[ridx].append(str(row["vehicle_id"]))

        # Sort each route's buses by d_m ascending
        final_d: list[np.ndarray] = []
        final_t: list[np.ndarray] = []
        final_v: list[list[str]] = []
        total_veh = 0

        for r in range(n_routes):
            if d_lists[r]:
                order = np.argsort(d_lists[r])
                arr_d = np.array(d_lists[r], dtype=np.float32)[order]
                arr_t = np.array(t_lists[r], dtype=np.float64)[order]
                v_sorted = [v_lists[r][i] for i in order]
                final_d.append(arr_d)
                final_t.append(arr_t)
                final_v.append(v_sorted)
                total_veh += len(v_sorted)
            else:
                final_d.append(np.empty(0, dtype=np.float32))
                final_t.append(np.empty(0, dtype=np.float64))
                final_v.append([])

        return LiveVehicleIndex(
            live_d_m=final_d,
            live_t_obs=final_t,
            live_veh_id=final_v,
            n_vehicles=total_veh,
            is_fresh=is_fresh,
            max_t_obs=max_ts,
        )
    except Exception as e:
        print(f"[WARN] Live vehicle indexing error: {e}")
        return empty_index


def norm_cdf(z: float) -> float:
    """Standard normal cumulative distribution function Phi(z)."""
    return 0.5 * (1.0 + math.erf(z / 1.4142135623730951))


def compute_leg_wait_and_ride(
    graph: TransitGraph,
    prefix_arrays: RoutePrefixArrays,
    live_index: LiveVehicleIndex | None,
    u_idx: int,
    v_idx: int,
    t_a: float,
    t_q: float,
    bucket: int,
    candidate_routes: list[int] | np.ndarray | None = None,
    attractive_set_info: dict[str, Any] | None = None,
) -> tuple[float, float, float, float, float, str]:
    """Compute wait time and ride time under common lines and live vehicle tracking.

    Follows Ruling Q3 closed-form formula:
    - Checks upstream live buses for each candidate route.
    - Gates live buses by ETA window (0..1500s), uncertainty (<= 0.5 EH), and catch prob (>= 0.5).
    - Blends nearest gated live bus Delta_star with prior arrival rate Lambda.

    Returns:
        (E_W, Var_W, mu_ride, V_ride, ff_ride, route_label)
    """
    # Determine candidate route set A
    if attractive_set_info is not None and "routes" in attractive_set_info:
        cand_r = list(attractive_set_info["routes"])
    elif candidate_routes is not None:
        cand_r = list(candidate_routes)
    else:
        # Fallback: look up in common lines table
        cl = graph.lookup_common_lines(u_idx, v_idx, bucket)
        if cl is not None:
            cand_r = list(cl["routes"])
        else:
            cand_r = []

    if not cand_r:
        return 99999.0, 99999.0, 99999.0, 99999.0, 99999.0, "No route"

    # Evaluate each candidate route's leg stats and live bus status
    active_routes: list[int] = []
    mu_list: list[float] = []
    ff_list: list[float] = []
    v_list: list[float] = []
    ew_list: list[float] = []

    gated_live_routes: list[int] = []
    gated_deltas: list[float] = []
    gated_sigmas: list[float] = []

    use_live = live_index is not None and live_index.is_fresh

    for r in cand_r:
        s = int(graph.route_stop_offsets[r])
        e = int(graph.route_stop_offsets[r + 1])
        stops = graph.route_stops[s:e]

        # Find position of u and v
        pos_u = -1
        pos_v = -1
        for p, st in enumerate(stops):
            if st == u_idx and pos_u < 0:
                pos_u = p
            elif st == v_idx and pos_u >= 0:
                pos_v = p
                break

        if pos_u < 0 or pos_v <= pos_u:
            continue

        mu_m, ff_m, V_m = prefix_arrays.get_leg_stats(r, pos_u, pos_v)
        eh_m = float(graph.EH_sec[r, bucket])
        ew_m = float(graph.EW_sec[r, bucket])
        cov_m = float(graph.cov_r[r])

        active_routes.append(r)
        mu_list.append(mu_m)
        ff_list.append(ff_m)
        v_list.append(V_m)
        ew_list.append(max(60.0, ew_m))

        has_gated_bus = False
        if use_live:
            d_buses = live_index.live_d_m[r]
            t_buses = live_index.live_t_obs[r]
            d_stop_u = float(graph.route_stop_dists_m[s + pos_u])

            if d_buses.size > 0:
                # Scan upstream buses from closest (largest d_bus < d_stop_u) backwards
                pos_b = int(np.searchsorted(d_buses, d_stop_u, side="left")) - 1
                for idx_b in range(pos_b, -1, -1):
                    d_b = float(d_buses[idx_b])
                    t_obs_b = float(t_buses[idx_b])
                    mu_b_u, _, V_b_u = prefix_arrays.get_stats_from_distance(r, d_b, pos_u)
                    eta_b = t_obs_b + mu_b_u
                    age_sec = max(0.0, t_q - t_obs_b)
                    sigma_b = math.sqrt(V_b_u + (0.15 * age_sec) ** 2)

                    # Gate rules:
                    # 1) 0 <= ETA_b - t_q <= 1500.0 (25 mins)
                    # 2) sigma_b <= 0.5 * EH_m
                    # 3) catch prob: 0.9 * norm_cdf((ETA_b - t_a) / max(15.0, sigma_b)) >= 0.5
                    gate1 = 0.0 <= (eta_b - t_q) <= 1500.0
                    gate2 = sigma_b <= (0.5 * eh_m)
                    z_catch = (eta_b - t_a) / max(15.0, sigma_b)
                    catch_prob = 0.9 * norm_cdf(z_catch)
                    gate3 = catch_prob >= 0.5

                    if gate1 and gate2 and gate3:
                        has_gated_bus = True
                        delta_b = max(0.0, eta_b - t_a)
                        gated_live_routes.append(r)
                        gated_deltas.append(delta_b)
                        gated_sigmas.append(sigma_b)
                        break

    if not active_routes:
        return 99999.0, 99999.0, 99999.0, 99999.0, 99999.0, "No valid route"

    # Compute arrival rate Lambda
    Lambda = 0.0
    for idx, r in enumerate(active_routes):
        eh_m = float(graph.EH_sec[r, bucket])
        cov_m = float(graph.cov_r[r])
        ew_m = ew_list[idx]
        if r in gated_live_routes:
            Lambda += (1.0 - cov_m) / max(60.0, eh_m)
        else:
            Lambda += 1.0 / max(60.0, ew_m)

    Lambda = max(1e-4, Lambda)

    # Prior route weights and average ride statistics
    inv_ew = [1.0 / max(60.0, ew) for ew in ew_list]
    sum_inv_ew = sum(inv_ew)
    weights = [w / sum_inv_ew for w in inv_ew]

    T_bar_A = sum(w * mu for w, mu in zip(weights, mu_list))
    V_bar_A = sum(w * v for w, v in zip(weights, v_list))
    FF_bar_A = sum(w * ff for w, ff in zip(weights, ff_list))

    # Case 1: No gated live bus exists
    if not gated_live_routes:
        if len(active_routes) == 1:
            r0 = active_routes[0]
            ew0 = ew_list[0]
            eh0 = float(graph.EH_sec[r0, bucket])
            cv2_0 = float(graph.CV2[r0, bucket])
            E_W = ew0
            Var_W = max(100.0, ((0.5 * eh0) ** 2) * (0.33 + cv2_0))
            mu_ride = mu_list[0]
            V_ride = v_list[0]
            ff_ride = ff_list[0]
            sname = str(graph.route_short_names[r0])
            route_label = f"Route {sname} (Sched wait: {E_W/60.0:.1f}m)"
        else:
            E_W = 1.0 / Lambda
            Var_W = max(100.0, 1.0 / (Lambda**2))
            mu_ride = T_bar_A
            V_ride = V_bar_A
            ff_ride = FF_bar_A
            names = " / ".join(str(graph.route_short_names[r]) for r in active_routes[:4])
            if len(active_routes) > 4:
                names += f" +{len(active_routes) - 4}"
            route_label = f"Take any of: {names} (Combined wait: {E_W/60.0:.1f}m)"

        return E_W, Var_W, mu_ride, V_ride, ff_ride, route_label

    # Case 2: Gated live bus exists
    best_idx = int(np.argmin(gated_deltas))
    r_star = gated_live_routes[best_idx]
    Delta_star = gated_deltas[best_idx]
    sigma_b_star = gated_sigmas[best_idx]

    # Find r_star's individual ride stats
    r_star_idx = active_routes.index(r_star)
    T_r_star = mu_list[r_star_idx]
    V_r_star = v_list[r_star_idx]
    FF_r_star = ff_list[r_star_idx]

    x = Lambda * Delta_star
    if x < 1e-5:
        exp_term = 1.0 - x
        E_W = Delta_star * (1.0 - 0.5 * x)
        E_W2 = (Delta_star**2) * (1.0 - (2.0 / 3.0) * x)
    else:
        exp_term = math.exp(-x)
        E_W = (1.0 - exp_term) / Lambda
        E_W2 = (2.0 / (Lambda**2)) * (1.0 - exp_term * (1.0 + x))

    Var_W = max(900.0, E_W2 - (E_W**2) + exp_term * (sigma_b_star**2))
    mu_ride = exp_term * T_r_star + (1.0 - exp_term) * T_bar_A
    V_ride = exp_term * V_r_star + (1.0 - exp_term) * V_bar_A
    ff_ride = exp_term * FF_r_star + (1.0 - exp_term) * FF_bar_A

    star_name = str(graph.route_short_names[r_star])
    if len(active_routes) == 1:
        route_label = (
            f"Route {star_name} (Live ETA: in {Delta_star/60.0:.1f}m, expected wait: {E_W/60.0:.1f}m)"
        )
    else:
        names = " / ".join(str(graph.route_short_names[r]) for r in active_routes[:3])
        route_label = (
            f"Take any of: {names} (Live bus on {star_name} in {Delta_star/60.0:.1f}m, combined wait: {E_W/60.0:.1f}m)"
        )

    return E_W, Var_W, mu_ride, V_ride, ff_ride, route_label
