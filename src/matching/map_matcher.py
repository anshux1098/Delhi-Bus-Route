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

from bisect import bisect_right
import os
import sys
import time
from typing import Any

# Ensure generic CPU architecture for Numba on Windows ARM64 / Snapdragon X
os.environ.setdefault("NUMBA_CPU_NAME", "generic")

import numpy as np
import polars as pl
from numba import njit

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


@njit(cache=True)
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

    for attempt in range(2):
        best_d = w_min
        min_perp_dist = 1e9
        bearing_dot = 1.0
        best_score = 1e9

        # Binary search for first segment i whose end sb = poly_s[i + 1] >= w_min
        if w_min <= 0.0:
            i_start = 0
        else:
            lo = 0
            hi = n_segs - 1
            while lo < hi:
                mid = (lo + hi) >> 1
                if poly_s[mid + 1] < w_min:
                    lo = mid + 1
                else:
                    hi = mid
            i_start = lo

        for i in range(i_start, n_segs):
            sa = poly_s[i]
            if sa > w_max:
                break
            sb = poly_s[i + 1]
            if sb < w_min:
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
            dx_p = px - proj_x
            dy_p = py - proj_y
            dist = np.sqrt(dx_p * dx_p + dy_p * dy_p)

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

        if min_perp_dist <= 1e8 or (w_min <= 0.0 and w_max >= total_len) or attempt == 1:
            return best_d, min_perp_dist, bearing_dot

        # Fallback to full polyline if localized window found no candidate segment
        w_min = 0.0
        w_max = total_len

    return 0.0, 1e9, 1.0


@njit(cache=True)
def _pava_isotonic_increasing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact Pool Adjacent Violators Algorithm (PAVA) for non-decreasing isotonic fit.

    Matches sklearn.isotonic.IsotonicRegression(increasing=True) on 1D arrays,
    averaging y over any tied x values before pooling adjacent decreasing blocks.
    """
    n = len(y)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return y.astype(np.float64)

    # Sort check (x is expected non-decreasing; fallback handles arbitrary order safely)
    is_sorted = True
    for i in range(1, n):
        if x[i] < x[i - 1]:
            is_sorted = False
            break

    if not is_sorted:
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ys = y[order]
    else:
        xs = x
        ys = y

    block_val = np.empty(n, dtype=np.float64)
    block_w = np.empty(n, dtype=np.float64)
    block_count = np.empty(n, dtype=np.int64)
    nb = 0

    # Group tied x values first, matching sklearn tie-handling
    i = 0
    while i < n:
        x_cur = xs[i]
        sum_y = ys[i]
        cnt = 1
        j = i + 1
        while j < n and xs[j] == x_cur:
            sum_y += ys[j]
            cnt += 1
            j += 1

        block_val[nb] = sum_y / cnt
        block_w[nb] = float(cnt)
        block_count[nb] = cnt
        nb += 1

        while nb >= 2 and block_val[nb - 2] > block_val[nb - 1]:
            w_tot = block_w[nb - 2] + block_w[nb - 1]
            block_val[nb - 2] = (
                block_val[nb - 2] * block_w[nb - 2] + block_val[nb - 1] * block_w[nb - 1]
            ) / w_tot
            block_w[nb - 2] = w_tot
            block_count[nb - 2] += block_count[nb - 1]
            nb -= 1

        i = j

    out_sorted = np.empty(n, dtype=np.float64)
    pos = 0
    for b in range(nb):
        v = block_val[b]
        c = block_count[b]
        for _ in range(c):
            out_sorted[pos] = v
            pos += 1

    if is_sorted:
        return out_sorted

    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        out[order[k]] = out_sorted[k]
    return out


@njit(cache=True)
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

    all_x = np.asarray(df["x_m"].to_numpy(), dtype=np.float64)
    all_y = np.asarray(df["y_m"].to_numpy(), dtype=np.float64)
    all_vx = np.asarray(df["vx"].to_numpy(), dtype=np.float64)
    all_vy = np.asarray(df["vy"].to_numpy(), dtype=np.float64)
    all_hb = np.asarray(df["has_bearing"].to_numpy(), dtype=np.bool_)
    all_dt = np.asarray(df["dt_sec"].to_numpy(), dtype=np.float64)

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
            out_shape_ids[start:end] = [best_sid] * (end - start)
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
    """Extract atomic stop-to-stop corridor segment traversals from matched traces."""
    seg_df, _ = extract_segments_and_spans(matched_df, static_index)
    return seg_df


def extract_segments_and_spans(
    matched_df: pl.DataFrame,
    static_index: StaticIndex,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Extract stop-to-stop segments with span_weight and extract poll-to-poll spans.

    For each contiguous run:
    - Identifies real matched ping-pairs (t_a, d_a) -> (t_b, d_b) with dt >= 15s and dd >= 80m.
    - Maps each interpolated stop-to-stop segment to its containing span.
    - Assigns span_weight = 1.0 / n_seg_in_span so one ping-pair counts exactly once.
    - Emits observed_spans with overlap fractions across traversed corridor segments.
    """
    from bisect import bisect_left

    seg_schema = {
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
        "span_id": pl.Utf8,
        "n_seg_in_span": pl.Int64,
        "span_weight": pl.Float64,
    }

    span_schema = {
        "span_id": pl.Utf8,
        "trip_run_id": pl.Utf8,
        "vehicle_id": pl.Utf8,
        "route_id": pl.Utf8,
        "shape_id": pl.Utf8,
        "t_a": pl.Float64,
        "t_b": pl.Float64,
        "travel_time_sec": pl.Float64,
        "d_a": pl.Float64,
        "d_b": pl.Float64,
        "span_length_m": pl.Float64,
        "span_pace_sec_per_m": pl.Float64,
        "corridor_ids": pl.Utf8,
        "overlap_fractions": pl.Utf8,
        "overlap_meters": pl.Utf8,
        "n_corridors": pl.Int64,
        "dep_time_u": pl.Float64,
    }

    if matched_df.is_empty():
        return pl.DataFrame(schema=seg_schema), pl.DataFrame(schema=span_schema)

    # 1. Filter to valid on-route pings and sort once so groups are contiguous & chronological
    on_df = matched_df.filter(
        (~pl.col("is_off_route"))
        & pl.col("shape_id").is_not_null()
        & pl.col("route_id").is_not_null()
    )

    if on_df.is_empty():
        return pl.DataFrame(schema=seg_schema), pl.DataFrame(schema=span_schema)

    sorted_df = on_df.sort(["vehicle_id", "route_id", "shape_id", "timestamp"])
    all_ts = np.asarray(sorted_df["timestamp"].to_numpy(), dtype=np.float64)
    all_d = np.asarray(sorted_df["d_m"].to_numpy(), dtype=np.float64)

    group_counts = sorted_df.group_by(
        ["vehicle_id", "route_id", "shape_id"], maintain_order=True
    ).len()
    vids = group_counts["vehicle_id"].to_list()
    rids = group_counts["route_id"].to_list()
    sids = group_counts["shape_id"].to_list()
    counts = group_counts["len"].to_list()

    # Columnar accumulators for zero-dict DataFrame construction
    seg_trip_run_id: list[str] = []
    seg_vid: list[str] = []
    seg_rid: list[str] = []
    seg_sid: list[str] = []
    seg_stop_u: list[str] = []
    seg_stop_v: list[str] = []
    seg_corridor_id: list[str] = []
    seg_dep_u: list[float] = []
    seg_dep_v: list[float] = []
    seg_tt: list[float] = []
    seg_len_col: list[float] = []
    seg_pace: list[float] = []
    seg_span_id: list[str] = []
    seg_n_span: list[int] = []
    seg_weight: list[float] = []

    sp_id_col: list[str] = []
    sp_trip_run_id: list[str] = []
    sp_vid: list[str] = []
    sp_rid: list[str] = []
    sp_sid: list[str] = []
    sp_ta: list[float] = []
    sp_tb: list[float] = []
    sp_tt: list[float] = []
    sp_da: list[float] = []
    sp_db: list[float] = []
    sp_len: list[float] = []
    sp_pace: list[float] = []
    sp_cids: list[str] = []
    sp_ofracs: list[str] = []
    sp_ometers: list[str] = []
    sp_ncorr: list[int] = []
    sp_dep_u: list[float] = []

    offset = 0
    for vid, rid, sid, count in zip(vids, rids, sids, counts):
        start_g = offset
        end_g = offset + count
        offset = end_g

        if count < 2:
            continue

        vid_str = str(vid)
        rid_str = str(rid)
        sid_str = str(sid)
        rs_info = static_index.get_route_shape(rid_str, sid_str)
        if not rs_info or len(rs_info.stop_ids) < 2:
            continue

        stop_ids = rs_info.stop_ids
        d_stops = rs_info.stop_dists_m
        d_first = float(d_stops[0])
        d_last = float(d_stops[-1])

        timestamps = all_ts[start_g:end_g]
        d_vals = all_d[start_g:end_g]

        # Split into runs on large time gaps (>1200s / 20 mins) or backwards jumps (>100m)
        splits = np.flatnonzero(
            (np.diff(timestamps) > 1200.0) | (np.diff(d_vals) < -100.0)
        ) + 1
        n_runs = len(splits) + 1
        run_starts = [0] + splits.tolist()
        run_ends = splits.tolist() + [count]

        for run_idx in range(n_runs):
            rs = run_starts[run_idx]
            re_ = run_ends[run_idx]
            if re_ - rs < 2:
                continue

            r_t = timestamps[rs:re_]
            r_d = d_vals[rs:re_]

            # 2. Terminal truncation:
            dep_zone_idx = np.flatnonzero(r_d <= d_first + 100.0)
            start_i = 0
            departed_first_stop = False
            if len(dep_zone_idx) > 0:
                last_dep = int(dep_zone_idx[-1])
                if last_dep < len(r_d) - 1:
                    start_i = last_dep
                    departed_first_stop = True
                else:
                    continue

            arr_zone_idx = np.flatnonzero(r_d >= d_last - 100.0)
            end_i = len(r_d)
            entered_last_stop = False
            if len(arr_zone_idx) > 0:
                first_arr_idx = int(arr_zone_idx[0])
                if first_arr_idx > start_i:
                    end_i = first_arr_idx + 1
                    entered_last_stop = True
                else:
                    continue

            if end_i - start_i < 2:
                continue

            r_t_trunc = r_t[start_i:end_i]
            r_d_trunc = r_d[start_i:end_i]

            need_anchor_start = departed_first_stop and r_d_trunc[0] <= d_first + 100.0
            need_anchor_end = entered_last_stop and r_d_trunc[-1] >= d_last - 100.0
            if need_anchor_start or need_anchor_end:
                r_d_trunc = r_d_trunc.copy()
                if need_anchor_start:
                    r_d_trunc[0] = min(r_d_trunc[0], d_first)
                if need_anchor_end:
                    r_d_trunc[-1] = max(r_d_trunc[-1], d_last)

            # 3. Fast Numba PAVA isotonic regression on d_m vs timestamp
            d_mono = _pava_isotonic_increasing(r_t_trunc, r_d_trunc)

            min_d_run = float(np.min(d_mono))
            max_d_run = float(np.max(d_mono))

            if (max_d_run - min_d_run) <= MIN_RUN_DISPLACEMENT_M:
                continue

            k_lo = int(np.searchsorted(d_stops, min_d_run, side="left"))
            k_hi = int(np.searchsorted(d_stops, max_d_run, side="right"))
            if k_hi - k_lo < 2:
                continue

            # Collapse flat spots: since d_mono is monotonically non-decreasing, max t is at the end of each plateau
            n_mono = len(d_mono)
            diff_mask = np.empty(n_mono, dtype=np.bool_)
            diff_mask[:-1] = d_mono[:-1] != d_mono[1:]
            diff_mask[-1] = True
            u_d_arr = d_mono[diff_mask]
            u_t_arr = r_t_trunc[diff_mask]
            if len(u_d_arr) < 2:
                continue

            trip_run_id = f"{vid_str}_{sid_str}_{int(r_t_trunc[0])}_{run_idx}"

            # Build ping-to-ping spans (T_ab = t_b - t_a >= 15s, d_b - d_a >= 80m)
            span_dt_arr = np.diff(r_t_trunc)
            span_dd_arr = np.diff(d_mono)
            valid_p = np.flatnonzero((span_dt_arr >= 15.0) & (span_dd_arr >= 80.0))

            run_span_ids: list[str] = []
            run_span_ta: list[float] = []
            run_span_tb: list[float] = []
            run_span_dt: list[float] = []
            run_span_da: list[float] = []
            run_span_db: list[float] = []
            run_span_dd: list[float] = []
            for p_idx in valid_p.tolist():
                t_a = float(r_t_trunc[p_idx])
                t_b = float(r_t_trunc[p_idx + 1])
                d_a = float(d_mono[p_idx])
                d_b = float(d_mono[p_idx + 1])
                run_span_ids.append(f"{trip_run_id}_s{p_idx}")
                run_span_ta.append(t_a)
                run_span_tb.append(t_b)
                run_span_dt.append(t_b - t_a)
                run_span_da.append(d_a)
                run_span_db.append(d_b)
                run_span_dd.append(d_b - d_a)

            n_spans = len(run_span_ids)

            # Vectorized interpolation of stop departure times
            dep_stops = np.interp(d_stops[k_lo:k_hi], u_d_arr, u_t_arr)

            run_seg_u_ids: list[str] = []
            run_seg_v_ids: list[str] = []
            run_seg_cids: list[str] = []
            run_seg_dep_u: list[float] = []
            run_seg_dep_v: list[float] = []
            run_seg_tt: list[float] = []
            run_seg_len: list[float] = []
            run_seg_pace: list[float] = []
            run_seg_span_id: list[str] = []
            run_seg_du: list[float] = []
            run_seg_dv: list[float] = []

            for local_i in range(k_hi - k_lo - 1):
                k_u = k_lo + local_i
                k_v = k_u + 1
                du = float(d_stops[k_u])
                dv = float(d_stops[k_v])
                seg_len = dv - du
                if seg_len <= 10.0:
                    continue

                dep_u = float(dep_stops[local_i])
                dep_v = float(dep_stops[local_i + 1])
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

                # Map to containing span in O(log n_spans)
                if n_spans == 0:
                    best_span_id = f"{trip_run_id}_s0"
                else:
                    d_mid = 0.5 * (du + dv)
                    sp_i = bisect_left(run_span_db, d_mid)
                    if sp_i < n_spans and run_span_da[sp_i] <= d_mid <= run_span_db[sp_i]:
                        best_span_id = run_span_ids[sp_i]
                    else:
                        # Nearest span endpoint (matches argmin tie-breaking by smaller index)
                        if sp_i == 0:
                            best_span_id = run_span_ids[0]
                        elif sp_i >= n_spans:
                            best_span_id = run_span_ids[n_spans - 1]
                        else:
                            dist_prev = d_mid - run_span_db[sp_i - 1]
                            dist_next = run_span_da[sp_i] - d_mid
                            best_span_id = (
                                run_span_ids[sp_i - 1]
                                if dist_prev <= dist_next
                                else run_span_ids[sp_i]
                            )

                run_seg_u_ids.append(stop_u_id)
                run_seg_v_ids.append(stop_v_id)
                run_seg_cids.append(corridor_seg_id)
                run_seg_dep_u.append(dep_u)
                run_seg_dep_v.append(dep_v)
                run_seg_tt.append(travel_time)
                run_seg_len.append(seg_len)
                run_seg_pace.append(pace)
                run_seg_span_id.append(best_span_id)
                run_seg_du.append(du)
                run_seg_dv.append(dv)

            # Compute span-to-segment overlaps via monotonic bisect in O(n_spans + n_segs)
            n_rsegs = len(run_seg_cids)
            for s_i in range(n_spans):
                s_da = run_span_da[s_i]
                s_db = run_span_db[s_i]
                c_ids: list[str] = []
                c_fracs: list[str] = []
                c_meters: list[str] = []

                j_start = bisect_right(run_seg_dv, s_da)
                for j in range(j_start, n_rsegs):
                    seg_u = run_seg_du[j]
                    if seg_u >= s_db:
                        break
                    seg_v = run_seg_dv[j]
                    ov = min(s_db, seg_v) - max(s_da, seg_u)
                    if ov >= 5.0:
                        c_ids.append(run_seg_cids[j])
                        c_fracs.append(str(round(ov / (seg_v - seg_u), 4)))
                        c_meters.append(str(round(ov, 1)))

                sp_id_col.append(run_span_ids[s_i])
                sp_trip_run_id.append(trip_run_id)
                sp_vid.append(vid_str)
                sp_rid.append(rid_str)
                sp_sid.append(sid_str)
                sp_ta.append(run_span_ta[s_i])
                sp_tb.append(run_span_tb[s_i])
                sp_tt.append(run_span_dt[s_i])
                sp_da.append(s_da)
                sp_db.append(s_db)
                sp_len.append(run_span_dd[s_i])
                sp_pace.append(run_span_dt[s_i] / run_span_dd[s_i])
                sp_cids.append(",".join(c_ids))
                sp_ofracs.append(",".join(c_fracs))
                sp_ometers.append(",".join(c_meters))
                sp_ncorr.append(len(c_ids))
                sp_dep_u.append(run_span_ta[s_i])

            # Compute n_seg_in_span and span_weight = 1.0 / n_seg_in_span
            if n_rsegs > 0:
                span_counts: dict[str, int] = {}
                for sid_key in run_seg_span_id:
                    span_counts[sid_key] = span_counts.get(sid_key, 0) + 1

                seg_trip_run_id.extend([trip_run_id] * n_rsegs)
                seg_vid.extend([vid_str] * n_rsegs)
                seg_rid.extend([rid_str] * n_rsegs)
                seg_sid.extend([sid_str] * n_rsegs)
                seg_stop_u.extend(run_seg_u_ids)
                seg_stop_v.extend(run_seg_v_ids)
                seg_corridor_id.extend(run_seg_cids)
                seg_dep_u.extend(run_seg_dep_u)
                seg_dep_v.extend(run_seg_dep_v)
                seg_tt.extend(run_seg_tt)
                seg_len_col.extend(run_seg_len)
                seg_pace.extend(run_seg_pace)
                seg_span_id.extend(run_seg_span_id)
                for sid_key in run_seg_span_id:
                    cnt = span_counts[sid_key]
                    seg_n_span.append(cnt)
                    seg_weight.append(1.0 / cnt)

    if seg_trip_run_id:
        seg_df = pl.DataFrame(
            {
                "trip_run_id": seg_trip_run_id,
                "vehicle_id": seg_vid,
                "route_id": seg_rid,
                "shape_id": seg_sid,
                "stop_u": seg_stop_u,
                "stop_v": seg_stop_v,
                "corridor_segment_id": seg_corridor_id,
                "dep_time_u": seg_dep_u,
                "dep_time_v": seg_dep_v,
                "travel_time_sec": seg_tt,
                "segment_length_m": seg_len_col,
                "pace_sec_per_m": seg_pace,
                "span_id": seg_span_id,
                "n_seg_in_span": seg_n_span,
                "span_weight": seg_weight,
            },
            schema=seg_schema,
        )
    else:
        seg_df = pl.DataFrame(schema=seg_schema)

    if sp_id_col:
        span_df = pl.DataFrame(
            {
                "span_id": sp_id_col,
                "trip_run_id": sp_trip_run_id,
                "vehicle_id": sp_vid,
                "route_id": sp_rid,
                "shape_id": sp_sid,
                "t_a": sp_ta,
                "t_b": sp_tb,
                "travel_time_sec": sp_tt,
                "d_a": sp_da,
                "d_b": sp_db,
                "span_length_m": sp_len,
                "span_pace_sec_per_m": sp_pace,
                "corridor_ids": sp_cids,
                "overlap_fractions": sp_ofracs,
                "overlap_meters": sp_ometers,
                "n_corridors": sp_ncorr,
                "dep_time_u": sp_dep_u,
            },
            schema=span_schema,
        )
    else:
        span_df = pl.DataFrame(schema=span_schema)

    return seg_df, span_df


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
