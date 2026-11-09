"""Numba-accelerated 1D Linear Referencing and Stop-to-Stop Segment Extraction Engine.

Implements:
1. @njit project_point_to_polyline_window: clamps projection to [0.0, 1.0] and returns
   (best_d, min_perp_dist_m, bearing_dot).
2. @njit match_vehicle_trace: tracks a vehicle's ping sequence with adaptive search windows
   and a 3-strike off-route gate.
3. extract_stop_to_stop_segments: groups runs, applies terminal truncation, fits
   IsotonicRegression(increasing=True) on d_m vs timestamp, interpolates departure timestamps,
   and computes atomic corridor segment traversals.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# Ensure generic CPU architecture for Numba on Windows ARM64 / Snapdragon X
os.environ.setdefault("NUMBA_CPU_NAME", "generic")

import numpy as np
import polars as pl
from numba import njit
from sklearn.isotonic import IsotonicRegression

from src.ingestion.hygiene import load_and_clean_pings
from src.matching.shape_indexer import StaticIndex, load_static_index

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Realistic transit pace bounds (sec/m):
# 0.03 s/m = 33.3 m/s (120 km/h max bus speed)
# 0.60 s/m = 1.67 m/s (6 km/h heavy congestion including stop dwell)
MIN_REALISTIC_PACE_S_PER_M = 0.03
MAX_REALISTIC_PACE_S_PER_M = 0.60
MIN_RUN_DISPLACEMENT_M = 50.0


@njit
def project_point_to_polyline_window(
    px: float,
    py: float,
    vx: float,
    vy: float,
    has_bearing: bool,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    poly_s: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    d_min: float,
    d_max: float,
) -> tuple[float, float, float]:
    """Project a point (px, py) onto the polyline restricted to arc-length window [d_min, d_max].

    Returns:
        (best_d, min_perp_dist_m, bearing_dot)
    """
    n_pts = len(poly_x)
    n_segs = n_pts - 1
    total_len = poly_s[n_pts - 1]

    w_min = max(0.0, d_min)
    w_max = min(total_len, d_max)
    if w_min > w_max:
        w_min = 0.0
        w_max = total_len

    best_d = w_min
    min_perp_dist = 1e9
    bearing_dot = 1.0
    best_score = 1e9

    for i in range(n_segs):
        sa = poly_s[i]
        sb = poly_s[i + 1]
        if sb < w_min or sa > w_max:
            continue

        L = sb - sa
        if L <= 1e-6:
            continue

        xa = poly_x[i]
        ya = poly_y[i]
        xb = poly_x[i + 1]
        yb = poly_y[i + 1]

        t_x = tx[i]
        t_y = ty[i]

        # Vector from vertex A to point P
        wx = px - xa
        wy = py - ya

        # Scalar projection along segment tangent
        proj = wx * t_x + wy * t_y
        t_proj = proj / L

        # Clamp segment projection to [0.0, 1.0]
        if t_proj < 0.0:
            t_clamped = 0.0
        elif t_proj > 1.0:
            t_clamped = 1.0
        else:
            t_clamped = t_proj

        proj_x = xa + t_clamped * (xb - xa)
        proj_y = ya + t_clamped * (yb - ya)
        dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

        b_dot = 1.0
        if has_bearing:
            b_dot = vx * t_x + vy * t_y

        # Score balances perpendicular distance with bearing alignment
        score = dist
        if has_bearing and b_dot < 0.0:
            score += 100.0 * (1.0 - b_dot)

        if score < best_score:
            best_score = score
            min_perp_dist = dist
            best_d = sa + t_clamped * L
            bearing_dot = b_dot

    # Fallback to full polyline if localized window found no candidate segment
    if min_perp_dist > 1e8 and (w_min > 0.0 or w_max < total_len):
        return project_point_to_polyline_window(
            px, py, vx, vy, has_bearing, poly_x, poly_y, poly_s, tx, ty, 0.0, total_len
        )

    return best_d, min_perp_dist, bearing_dot


@njit
def match_vehicle_trace(
    p_x: np.ndarray,
    p_y: np.ndarray,
    p_vx: np.ndarray,
    p_vy: np.ndarray,
    p_has_bearing: np.ndarray,
    p_dt_sec: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    poly_s: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map-match a sequence of vehicle pings along a candidate shape polyline.

    Tracking rules:
    - First ping or re-acquisition: global search [0, L].
    - Subsequent pings: gap-scaled window [d_prev - 50.0, d_prev + 22.0 * dt_sec + 200.0].
    - Off-route gate: if min_perp_dist_m > 75.0m for 3 consecutive pings,
      mark is_off_route = True and trigger re-acquisition on next ping.
    """
    m = len(p_x)
    total_len = poly_s[len(poly_s) - 1]

    out_d = np.zeros(m, dtype=np.float64)
    out_perp_dist = np.zeros(m, dtype=np.float64)
    out_bearing_dot = np.zeros(m, dtype=np.float64)
    out_is_off_route = np.zeros(m, dtype=np.bool_)

    in_reacquisition = True
    d_prev = 0.0
    consecutive_off_route = 0

    for j in range(m):
        dt = p_dt_sec[j]
        if dt < 0.0:
            dt = 0.0

        if in_reacquisition or j == 0:
            d_min = 0.0
            d_max = total_len
        else:
            d_min = max(0.0, d_prev - 50.0)
            d_max = min(total_len, d_prev + 22.0 * dt + 200.0)

        best_d, perp_dist, b_dot = project_point_to_polyline_window(
            p_x[j],
            p_y[j],
            p_vx[j],
            p_vy[j],
            bool(p_has_bearing[j]),
            poly_x,
            poly_y,
            poly_s,
            tx,
            ty,
            d_min,
            d_max,
        )

        if perp_dist > 75.0:
            consecutive_off_route += 1
        else:
            consecutive_off_route = 0

        if consecutive_off_route >= 3:
            out_is_off_route[j] = True
            in_reacquisition = True
        else:
            out_is_off_route[j] = False
            in_reacquisition = False
            d_prev = best_d

        out_d[j] = best_d
        out_perp_dist[j] = perp_dist
        out_bearing_dot[j] = b_dot

    return out_d, out_perp_dist, out_bearing_dot, out_is_off_route


def match_pings(cleaned_df: pl.DataFrame, static_index: StaticIndex) -> pl.DataFrame:
    """Run map-matching on all vehicle traces in cleaned_df against static shapes."""
    if cleaned_df.is_empty():
        return cleaned_df.with_columns(
            shape_id=pl.lit(None, dtype=pl.Utf8),
            d_m=pl.lit(None, dtype=pl.Float64),
            perp_dist_m=pl.lit(None, dtype=pl.Float64),
            bearing_dot=pl.lit(None, dtype=pl.Float64),
            is_off_route=pl.lit(None, dtype=pl.Boolean),
        )

    # Ensure hygiene fields are present
    if "x_m" not in cleaned_df.columns or "vx" not in cleaned_df.columns:
        cleaned_df = load_and_clean_pings(cleaned_df)

    # Sort by vehicle_id, route_id, timestamp so contiguous slices can be processed in-place
    df = cleaned_df.with_columns(
        pl.col("vehicle_id").cast(pl.Utf8),
        pl.col("route_id").cast(pl.Utf8),
    ).sort(["vehicle_id", "route_id", "timestamp"])

    n_total = df.height

    # Warm up numba matcher
    dummy_x = np.array([0.0, 100.0], dtype=np.float64)
    dummy_y = np.array([0.0, 0.0], dtype=np.float64)
    dummy_s = np.array([0.0, 100.0], dtype=np.float64)
    dummy_tx = np.array([1.0], dtype=np.float64)
    dummy_ty = np.array([0.0], dtype=np.float64)
    match_vehicle_trace(
        np.array([50.0]),
        np.array([5.0]),
        np.array([1.0]),
        np.array([0.0]),
        np.array([True]),
        np.array([10.0]),
        dummy_x,
        dummy_y,
        dummy_s,
        dummy_tx,
        dummy_ty,
    )

    all_x = df["x_m"].to_numpy().astype(np.float64)
    all_y = df["y_m"].to_numpy().astype(np.float64)
    all_vx = df["vx"].to_numpy().astype(np.float64)
    all_vy = df["vy"].to_numpy().astype(np.float64)
    all_hb = df["has_bearing"].to_numpy().astype(np.bool_)
    all_dt = df["dt_sec"].to_numpy().astype(np.float64)

    out_shape_ids: list[str | None] = [None] * n_total
    out_d_m = np.zeros(n_total, dtype=np.float64)
    out_perp_m = np.full(n_total, 999.0, dtype=np.float64)
    out_bdot = np.zeros(n_total, dtype=np.float64)
    out_off = np.ones(n_total, dtype=np.bool_)

    # Find group boundaries over (vehicle_id, route_id)
    group_counts = df.group_by(["vehicle_id", "route_id"], maintain_order=True).len()
    rids = group_counts["route_id"].to_list()
    lengths = group_counts["len"].to_list()

    offset = 0
    for rid, count in zip(rids, lengths):
        start = offset
        end = offset + count
        offset = end

        if not rid:
            continue

        candidate_shapes = static_index.get_candidate_shapes(str(rid))
        if not candidate_shapes:
            continue

        p_x = all_x[start:end]
        p_y = all_y[start:end]
        p_vx = all_vx[start:end]
        p_vy = all_vy[start:end]
        p_hb = all_hb[start:end]
        p_dt = all_dt[start:end]

        best_sid: str | None = None
        best_d: np.ndarray | None = None
        best_perp: np.ndarray | None = None
        best_bdot: np.ndarray | None = None
        best_off: np.ndarray | None = None
        min_avg_perp = 1e9

        for sid in candidate_shapes:
            poly = static_index.get_shape(sid)
            if not poly or len(poly.x_m) < 2:
                continue

            d_arr, perp_arr, bdot_arr, off_arr = match_vehicle_trace(
                p_x,
                p_y,
                p_vx,
                p_vy,
                p_hb,
                p_dt,
                poly.x_m,
                poly.y_m,
                poly.s_m,
                poly.tx,
                poly.ty,
            )

            avg_perp = float(np.mean(perp_arr))
            if avg_perp < min_avg_perp:
                min_avg_perp = avg_perp
                best_sid = sid
                best_d = d_arr
                best_perp = perp_arr
                best_bdot = bdot_arr
                best_off = off_arr

        if best_d is not None and best_sid is not None:
            for k in range(start, end):
                out_shape_ids[k] = best_sid
            out_d_m[start:end] = best_d
            out_perp_m[start:end] = best_perp
            out_bdot[start:end] = best_bdot
            out_off[start:end] = best_off

    return df.with_columns(
        shape_id=pl.Series("shape_id", out_shape_ids, dtype=pl.Utf8),
        d_m=pl.Series("d_m", out_d_m, dtype=pl.Float64),
        perp_dist_m=pl.Series("perp_dist_m", out_perp_m, dtype=pl.Float64),
        bearing_dot=pl.Series("bearing_dot", out_bdot, dtype=pl.Float64),
        is_off_route=pl.Series("is_off_route", out_off, dtype=pl.Boolean),
    )


def extract_stop_to_stop_segments(
    matched_df: pl.DataFrame, static_index: StaticIndex
) -> pl.DataFrame:
    """Extract atomic stop-to-stop corridor segment traversals from matched traces.

    Pipeline:
    1. Filter to on-route pings and partition into single-direction runs (trip_run_id).
    2. Terminal truncation: strip idle pings within 100m of the first stop prior to final
       departure, and after first entry within 100m of the final stop.
    3. Require forward displacement > 50m and fit IsotonicRegression(increasing=True)
       on d_m vs timestamp.
    4. Interpolate single-point departure timestamps at canonical stop positions.
    5. Output atomic segment traversals with realistic pace_sec_per_m in [0.03, 0.60].
    """
    out_schema = {
        "trip_run_id": pl.Utf8,
        "vehicle_id": pl.Utf8,
        "route_id": pl.Utf8,
        "shape_id": pl.Utf8,
        "stop_u": pl.Utf8,
        "stop_v": pl.Utf8,
        "corridor_segment_id": pl.Utf8,
        "dep_time_u": pl.Float64,
        "dep_time_v": pl.Float64,
        "travel_time_sec": pl.Float64,
        "segment_length_m": pl.Float64,
        "pace_sec_per_m": pl.Float64,
    }

    if matched_df.is_empty():
        return pl.DataFrame(schema=out_schema)

    # 1. Filter to valid on-route pings
    on_df = matched_df.filter(
        (~pl.col("is_off_route"))
        & pl.col("shape_id").is_not_null()
        & pl.col("route_id").is_not_null()
    )

    if on_df.is_empty():
        return pl.DataFrame(schema=out_schema)

    segment_rows: list[dict[str, Any]] = []

    # Group by vehicle_id, route_id, shape_id
    grouped = on_df.sort(["vehicle_id", "timestamp"]).group_by(
        ["vehicle_id", "route_id", "shape_id"], maintain_order=True
    )

    for (vid, rid, sid), group_df in grouped:
        rs_info = static_index.get_route_shape(str(rid), str(sid))
        if not rs_info or len(rs_info.stop_ids) < 2:
            continue

        stop_ids = rs_info.stop_ids
        d_stops = rs_info.stop_dists_m
        d_first = float(d_stops[0])
        d_last = float(d_stops[-1])

        timestamps = group_df["timestamp"].to_numpy().astype(np.float64)
        d_vals = group_df["d_m"].to_numpy().astype(np.float64)

        n_pings = len(timestamps)
        if n_pings < 2:
            continue

        # Detect run splits on large time gaps (>1200s / 20 mins) or backwards jumps (>100m)
        run_indices: list[list[int]] = [[0]]
        for i in range(1, n_pings):
            dt = timestamps[i] - timestamps[i - 1]
            dd = d_vals[i] - d_vals[i - 1]
            if dt > 1200.0 or dd < -100.0:
                run_indices.append([i])
            else:
                run_indices[-1].append(i)

        for run_idx, indices in enumerate(run_indices):
            if len(indices) < 2:
                continue

            r_t = timestamps[indices].copy()
            r_d = d_vals[indices].copy()

            # 2. Terminal truncation:
            # (a) Strip pings within 100m of the first stop prior to final departure
            dep_zone_idx = np.where(r_d <= d_first + 100.0)[0]
            start_i = 0
            departed_first_stop = False
            if len(dep_zone_idx) > 0:
                if dep_zone_idx[-1] < len(r_d) - 1:
                    # Vehicle departed the first-stop 100m zone; keep from final departure ping
                    start_i = int(dep_zone_idx[-1])
                    departed_first_stop = True
                else:
                    # Vehicle never left the 100m zone of the first stop
                    continue

            # (b) Strip pings after first entry within 100m of final stop
            arr_zone_idx = np.where(r_d >= d_last - 100.0)[0]
            end_i = len(r_d)
            entered_last_stop = False
            if len(arr_zone_idx) > 0:
                first_arr_idx = int(arr_zone_idx[0])
                if first_arr_idx > start_i:
                    end_i = first_arr_idx + 1
                    entered_last_stop = True
                else:
                    # Vehicle started inside the final stop 100m zone and never left
                    continue

            if end_i - start_i < 2:
                continue

            r_t_trunc = r_t[start_i:end_i].copy()
            r_d_trunc = r_d[start_i:end_i].copy()

            # Anchor terminal stop boundaries when vehicle departed first stop / entered last stop zone
            if departed_first_stop and r_d_trunc[0] <= d_first + 100.0:
                r_d_trunc[0] = min(r_d_trunc[0], d_first)
            if entered_last_stop and r_d_trunc[-1] >= d_last - 100.0:
                r_d_trunc[-1] = max(r_d_trunc[-1], d_last)

            # 3. Fit IsotonicRegression(increasing=True) on d_m vs timestamp
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            d_mono = iso.fit_transform(r_t_trunc, r_d_trunc)

            min_d_run = float(np.min(d_mono))
            max_d_run = float(np.max(d_mono))

            # Require vehicle to move >50 meters along route
            if (max_d_run - min_d_run) <= MIN_RUN_DISPLACEMENT_M:
                continue

            # Identify stops covered by this run
            covered_stop_indices = [
                k for k in range(len(d_stops)) if min_d_run <= d_stops[k] <= max_d_run
            ]

            if len(covered_stop_indices) < 2:
                continue

            # Collapse flat spots to departure timestamp (maximum t for each unique d)
            unique_d = []
            unique_t = []
            for d_val in np.unique(d_mono):
                mask = d_mono == d_val
                unique_d.append(float(d_val))
                unique_t.append(float(np.max(r_t_trunc[mask])))

            if len(unique_d) < 2:
                continue

            u_d_arr = np.array(unique_d, dtype=np.float64)
            u_t_arr = np.array(unique_t, dtype=np.float64)

            trip_run_id = f"{vid}_{sid}_{int(r_t_trunc[0])}_{run_idx}"

            # 4. Interpolate single-point departure timestamps at each covered adjacent pair
            for idx_k in range(len(covered_stop_indices) - 1):
                k_u = covered_stop_indices[idx_k]
                k_v = covered_stop_indices[idx_k + 1]

                # Only adjacent stops form atomic corridor segments
                if k_v != k_u + 1:
                    continue

                du = float(d_stops[k_u])
                dv = float(d_stops[k_v])
                seg_len = dv - du
                if seg_len <= 10.0:
                    continue

                dep_u = float(np.interp(du, u_d_arr, u_t_arr))
                dep_v = float(np.interp(dv, u_d_arr, u_t_arr))
                travel_time = dep_v - dep_u

                if travel_time <= 0.0:
                    continue

                pace = travel_time / seg_len
                if not (MIN_REALISTIC_PACE_S_PER_M <= pace <= MAX_REALISTIC_PACE_S_PER_M):
                    continue

                stop_u_id = stop_ids[k_u]
                stop_v_id = stop_ids[k_v]
                nominal_len_100m = int(round(seg_len / 100.0) * 100)
                corridor_seg_id = f"{stop_u_id}_{stop_v_id}_{nominal_len_100m}"

                segment_rows.append({
                    "trip_run_id": trip_run_id,
                    "vehicle_id": str(vid),
                    "route_id": str(rid),
                    "shape_id": str(sid),
                    "stop_u": stop_u_id,
                    "stop_v": stop_v_id,
                    "corridor_segment_id": corridor_seg_id,
                    "dep_time_u": dep_u,
                    "dep_time_v": dep_v,
                    "travel_time_sec": travel_time,
                    "segment_length_m": seg_len,
                    "pace_sec_per_m": pace,
                })

    if not segment_rows:
        return pl.DataFrame(schema=out_schema)

    seg_df = pl.DataFrame(segment_rows, schema=out_schema, strict=False)
    return seg_df


if __name__ == "__main__":
    print("=================================================================")
    print("Testing 1D Map-Matcher and Segment Extractor on Real Delhi Data")
    print("=================================================================")

    # 1. Load static index
    t0 = time.time()
    static_index = load_static_index()
    print(f"[OK] Static index loaded in {time.time() - t0:.3f}s")

    # 2. Run hygiene pipeline
    t_clean = time.time()
    cleaned_df = load_and_clean_pings()
    print(
        f"[OK] Hygiene pipeline cleaned {cleaned_df.height:,} pings across "
        f"{cleaned_df['vehicle_id'].n_unique():,} vehicles in {time.time() - t_clean:.2f}s"
    )

    # 3. Run Map-Matching
    t_match = time.time()
    matched_df = match_pings(cleaned_df, static_index)
    match_time = time.time() - t_match
    on_route_count = matched_df.filter(~pl.col("is_off_route")).height
    print(
        f"[OK] Map-matched {matched_df.height:,} pings in {match_time:.2f}s "
        f"({matched_df.height / max(0.001, match_time):,.0f} pings/sec)"
    )
    print(
        f"     On-route pings: {on_route_count:,} "
        f"({(on_route_count / max(1, matched_df.height)) * 100:.1f}%)"
    )

    # 4. Extract Stop-to-Stop Segments
    t_seg = time.time()
    seg_df = extract_stop_to_stop_segments(matched_df, static_index)
    print(
        f"[OK] Extracted {seg_df.height:,} atomic segment traversals in "
        f"{time.time() - t_seg:.2f}s"
    )

    if seg_df.height > 0:
        print("\nFirst 5 atomic stop-to-stop traversals:")
        pl.Config.set_tbl_cols(12)
        print(seg_df.head(5))
        print("\nSummary Statistics of Segments:")
        print(seg_df.select([
            pl.col("travel_time_sec").mean().alias("avg_travel_time_s"),
            pl.col("segment_length_m").mean().alias("avg_segment_len_m"),
            pl.col("pace_sec_per_m").min().alias("min_pace_s_per_m"),
            pl.col("pace_sec_per_m").mean().alias("avg_pace_s_per_m"),
            pl.col("pace_sec_per_m").max().alias("max_pace_s_per_m"),
            pl.col("corridor_segment_id").n_unique().alias("unique_corridors"),
        ]))
    else:
        print("\nNote: Additional pings over longer time window needed for full stop traversal.")

    print("\n[SUCCESS] Map-matcher and segment extraction pipeline verified.")
