"""Delhi Bus Transit Out-of-Fold Empirical Bayes Target Encoder and Diurnal Prior.

Builds empirical Bayesian shrinkage priors for bus corridor traversal times across Delhi NCR:
1. Ingests all historical GPS pings via hygiene -> map-matching -> segment extraction.
2. Extracts IST time features (bin_15m 0..95, hour_float, day_of_week, is_weekend).
3. Computes the smooth continuous diurnal congestion function m_global anchored to observed Delhi GPS pace.
4. Performs Empirical Bayes shrinkage with strength k=5.0 toward m_global.
5. Derives free_flow_time_sec per corridor segment (P15 / 05:30 AM baseline).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Ensure generic CPU architecture for Numba on Windows ARM64 / Snapdragon X
os.environ.setdefault("NUMBA_CPU_NAME", "generic")

from src.ingestion.hygiene import find_project_root, load_and_clean_pings
from src.matching.map_matcher import (
    extract_segments_and_spans,
    extract_stop_to_stop_segments,
    match_pings,
)
from src.matching.shape_indexer import load_static_index

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = find_project_root()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OBSERVED_SEGMENTS_PATH = PROCESSED_DIR / "observed_segments.parquet"
SPANS_PATH = PROCESSED_DIR / "observed_spans.parquet"
CATALOG_PATH = PROCESSED_DIR / "segment_catalog.parquet"
EB_PRIORS_PATH = PROCESSED_DIR / "eb_priors.parquet"

# Shrinkage hyperparameter (Audit Fix 3.6 & 3.9)
EB_K = 5.0

# Typical Delhi bus operational bounds
DEFAULT_MEDIAN_PACE_S_PER_M = 0.205  # ~17.5 km/h
MIN_REALISTIC_PACE_S_PER_M = 0.030   # 120 km/h
MAX_REALISTIC_PACE_S_PER_M = 0.600   # 6 km/h


def epoch_to_ist_features(epoch_seconds: np.ndarray | float | int) -> dict[str, np.ndarray | int]:
    """Convert Unix epoch timestamp(s) to IST time features without external timezone libs."""
    epoch_arr = np.asarray(epoch_seconds, dtype=np.int64)
    # IST = UTC + 5h30m = UTC + 19,800 seconds
    ist_sec = epoch_arr + 19800

    sec_of_day = ist_sec % 86400
    bin_15m = (sec_of_day // 900).astype(np.int32)
    hour_float = (bin_15m / 4.0).astype(np.float64)

    # 1970-01-01 was Thursday (day index 3 in 0=Mon..6=Sun)
    day_index = ist_sec // 86400
    day_of_week = ((day_index + 3) % 7).astype(np.int8)
    is_weekend = np.where((day_of_week == 5) | (day_of_week == 6), 1, 0).astype(np.int8)

    return {
        "bin_15m": bin_15m,
        "hour_float": hour_float,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
    }


def get_diurnal_multiplier(bin_15m: int | np.ndarray, is_weekend: int | np.ndarray = 0) -> float | np.ndarray:
    """Smooth continuous diurnal congestion curve for Delhi NCR transit.

    Multipliers:
    - 05:00-06:30 (bins 20-25): ~0.72x (Free-flow / dawn)
    - 08:30-10:30 (bins 34-41): ~1.35x (Morning Peak)
    - 12:00-16:00 (bins 48-63): ~1.05x (Midday base)
    - 17:30-20:00 (bins 70-79): ~1.45x (Evening Peak)
    - 22:00-04:30 (bins 88-95, 0-17): ~0.85x (Late Night / quiet)
    """
    b = np.asarray(bin_15m, dtype=np.float64)
    w = np.asarray(is_weekend, dtype=np.float64)
    h = (b * 15.0) / 60.0

    # Weekday Gaussian components
    c_dawn = np.exp(-0.5 * ((h - 5.5) / 1.2) ** 2)
    c_morn_wd = np.exp(-0.5 * ((h - 9.5) / 1.4) ** 2)
    c_eve_wd = np.exp(-0.5 * ((h - 18.75) / 1.5) ** 2)
    night_dist = np.minimum(h, np.abs(h - 24.0))
    c_night = np.exp(-0.5 * ((night_dist - 2.5) / 2.0) ** 2)

    # Weekend Gaussian components
    c_morn_we = np.exp(-0.5 * ((h - 11.0) / 2.5) ** 2)
    c_eve_we = np.exp(-0.5 * ((h - 18.5) / 2.0) ** 2)

    mult_wd = 1.05 - 0.33 * c_dawn + 0.30 * c_morn_wd + 0.40 * c_eve_wd - 0.20 * c_night
    mult_we = 1.02 - 0.27 * c_dawn + 0.13 * c_morn_we + 0.28 * c_eve_we - 0.17 * c_night

    mult = (1.0 - w) * mult_wd + w * mult_we
    return np.clip(mult, 0.70, 1.55)


def compute_m_global(
    base_median_pace: float,
    bin_15m: int | np.ndarray,
    is_weekend: int | np.ndarray = 0,
    dist_from_center_km: float | np.ndarray = 12.0,
    is_major_hub_v: int | np.ndarray = 0,
) -> float | np.ndarray:
    """Compute prior expected pace m_global with diurnal, spatial, and hub modifiers."""
    d_mult = get_diurnal_multiplier(bin_15m, is_weekend)

    dist = np.asarray(dist_from_center_km, dtype=np.float64)
    center_dist_factor = np.clip(1.08 - 0.008 * dist, 0.90, 1.15)

    hub = np.asarray(is_major_hub_v, dtype=np.float64)
    hub_factor = 1.0 + 0.08 * hub

    m_prior = base_median_pace * d_mult * center_dist_factor * hub_factor
    return np.clip(m_prior, MIN_REALISTIC_PACE_S_PER_M, MAX_REALISTIC_PACE_S_PER_M)


@dataclass
class EmpiricalBayesPrior:
    """Compiled Empirical Bayes shrinkage table and free-flow lookups."""

    global_median_pace: float
    global_p15_pace: float
    global_p85_pace: float
    # (corridor_segment_id, bin_15m, is_weekend) -> eb_prior_pace
    eb_table: dict[tuple[str, int, int], float] = field(default_factory=dict)
    # corridor_segment_id -> de-seasonalized base pace (s/m at diurnal mult = 1.0)
    corridor_base_pace: dict[str, float] = field(default_factory=dict)
    # corridor_segment_id -> free_flow_time_sec
    free_flow_times: dict[str, float] = field(default_factory=dict)
    # corridor_segment_id -> attributes
    catalog_info: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_prior_pace(
        self,
        corridor_segment_id: str,
        bin_15m: int,
        is_weekend: int = 0,
    ) -> float:
        """Query Empirical Bayes prior pace with hierarchical corridor base pace and fallback to m_global."""
        cid = str(corridor_segment_id)
        b_idx = int(bin_15m)
        we = int(is_weekend)
        key = (cid, b_idx, we)
        if key in self.eb_table:
            return self.eb_table[key]

        if cid in self.corridor_base_pace:
            d_mult = float(get_diurnal_multiplier(b_idx, we))
            return float(
                np.clip(
                    self.corridor_base_pace[cid] * d_mult,
                    MIN_REALISTIC_PACE_S_PER_M,
                    MAX_REALISTIC_PACE_S_PER_M,
                )
            )

        cat = self.catalog_info.get(cid)
        dist_km = cat["dist_from_center_km"] if cat else 12.0
        is_hub = cat["is_major_hub_v"] if cat else 0

        return float(
            compute_m_global(
                base_median_pace=self.global_median_pace,
                bin_15m=b_idx,
                is_weekend=we,
                dist_from_center_km=dist_km,
                is_major_hub_v=is_hub,
            )
        )

    def get_prior_time(
        self,
        corridor_segment_id: str,
        bin_15m: int,
        is_weekend: int = 0,
        length_m: float | None = None,
    ) -> float:
        """Expected traversal time according to EB prior."""
        pace = self.get_prior_pace(corridor_segment_id, bin_15m, is_weekend)
        if length_m is None or length_m <= 0.0:
            cat = self.catalog_info.get(str(corridor_segment_id))
            length_m = cat["segment_length_m"] if cat else 250.0
        return float(length_m * pace)

    def get_free_flow_time(
        self,
        corridor_segment_id: str,
        length_m: float | None = None,
    ) -> float:
        """P15 / 05:30 AM baseline traversal time."""
        cid = str(corridor_segment_id)
        if cid in self.free_flow_times:
            return self.free_flow_times[cid]
        if length_m is None or length_m <= 0.0:
            cat = self.catalog_info.get(cid)
            length_m = cat["segment_length_m"] if cat else 250.0
        return float(length_m * self.global_p15_pace)


def consolidate_observed_segments(
    save_path: Path | None = None,
    save_spans_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run full ingestion, segment extraction, and poll-to-poll span extraction."""
    t0 = time.time()
    idx = load_static_index()
    cleaned = load_and_clean_pings()
    matched = match_pings(cleaned, idx)
    segments, spans = extract_segments_and_spans(matched, idx)

    if segments.is_empty():
        print("[WARN] No observed segments extracted from historical pings.")
        return segments, spans

    # Add IST time columns to segments
    times_u = segments["dep_time_u"].to_numpy()
    feat = epoch_to_ist_features(times_u)

    segments = segments.with_columns(
        bin_15m=pl.Series("bin_15m", feat["bin_15m"], dtype=pl.Int32),
        hour_float=pl.Series("hour_float", feat["hour_float"], dtype=pl.Float64),
        day_of_week=pl.Series("day_of_week", feat["day_of_week"], dtype=pl.Int8),
        is_weekend=pl.Series("is_weekend", feat["is_weekend"], dtype=pl.Int8),
    )

    # Attach catalog features to segments
    if CATALOG_PATH.exists():
        catalog = pl.read_parquet(CATALOG_PATH)
        cat_cols = [
            "corridor_segment_id",
            "mid_lat",
            "mid_lon",
            "dist_from_center_km",
            "bearing_rad",
            "n_routes_sharing",
            "is_major_hub_u",
            "is_major_hub_v",
        ]
        available_cat_cols = [c for c in cat_cols if c in catalog.columns]
        segments = segments.join(
            catalog.select(available_cat_cols),
            on="corridor_segment_id",
            how="left",
        )

    out_file = save_path or OBSERVED_SEGMENTS_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)
    segments.write_parquet(out_file, compression="zstd")

    # Add IST time columns to spans
    if not spans.is_empty():
        times_span = spans["dep_time_u"].to_numpy()
        feat_spans = epoch_to_ist_features(times_span)
        spans = spans.with_columns(
            bin_15m=pl.Series("bin_15m", feat_spans["bin_15m"], dtype=pl.Int32),
            hour_float=pl.Series("hour_float", feat_spans["hour_float"], dtype=pl.Float64),
            day_of_week=pl.Series("day_of_week", feat_spans["day_of_week"], dtype=pl.Int8),
            is_weekend=pl.Series("is_weekend", feat_spans["is_weekend"], dtype=pl.Int8),
        )
        out_spans_file = save_spans_path or SPANS_PATH
        out_spans_file.parent.mkdir(parents=True, exist_ok=True)
        spans.write_parquet(out_spans_file, compression="zstd")
        print(
            f"[OK] Consolidated {spans.height:,} observed spans to {out_spans_file.name}"
        )

    print(
        f"[OK] Consolidated {segments.height:,} observed traversals to "
        f"{out_file.name} in {time.time() - t0:.2f}s"
    )
    return segments, spans


def compute_oof_eb_features(
    observed_df: pl.DataFrame,
    catalog_df: pl.DataFrame | None = None,
    k: float = EB_K,
    base_median_pace: float | None = None,
) -> pl.DataFrame:
    """Compute Leave-One-Out (Out-of-Fold) Empirical Bayes prior features (Audit Fix 3.9).

    Ensures that for any training row i:
    - Its own observed pace y_i is strictly excluded from the group sum and count:
      n_{-i} = n - 1, sum_{-i} = sum - y_i.
    - If a segment-bin (or corridor) has only n = 1 observation, n_{-i} = 0, so
      w_{-i} = 0 / (0 + k) = 0.0 and eb_prior_pace_s_per_m == m_global (zero leakage).
    """
    if observed_df.is_empty():
        return observed_df.with_columns(
            eb_prior_pace_s_per_m=pl.lit(0.0, dtype=pl.Float64),
            eb_prior_time_sec=pl.lit(0.0, dtype=pl.Float64),
        )

    if catalog_df is None and CATALOG_PATH.exists():
        catalog_df = pl.read_parquet(CATALOG_PATH)

    # Determine base_median_pace
    if base_median_pace is None:
        paces_raw = observed_df["pace_sec_per_m"].to_numpy()
        valid = paces_raw[
            (paces_raw >= MIN_REALISTIC_PACE_S_PER_M) & (paces_raw <= MAX_REALISTIC_PACE_S_PER_M)
        ]
        # Use fixed global anchor if small sample so synthetic 1-row mutations don't shift global median
        if len(valid) > 200:
            base_median_pace = float(np.median(valid))
        else:
            base_median_pace = DEFAULT_MEDIAN_PACE_S_PER_M

    # Ensure catalog columns exist on observed_df
    df = observed_df
    if "dist_from_center_km" not in df.columns or "is_major_hub_v" not in df.columns:
        if catalog_df is not None:
            cat_cols = [
                c
                for c in ["corridor_segment_id", "dist_from_center_km", "is_major_hub_v"]
                if c in catalog_df.columns and (c == "corridor_segment_id" or c not in df.columns)
            ]
            if len(cat_cols) > 1:
                df = df.join(catalog_df.select(cat_cols), on="corridor_segment_id", how="left")

    n_rows = df.height
    cids = df["corridor_segment_id"].to_list()
    bins = df["bin_15m"].to_numpy().astype(np.int32)
    wes = df["is_weekend"].to_numpy().astype(np.int8)
    paces = df["pace_sec_per_m"].to_numpy().astype(np.float64)
    lens = df["segment_length_m"].to_numpy().astype(np.float64)

    dists = (
        df["dist_from_center_km"].fill_null(12.0).to_numpy().astype(np.float64)
        if "dist_from_center_km" in df.columns
        else np.full(n_rows, 12.0, dtype=np.float64)
    )
    hubs_v = (
        df["is_major_hub_v"].fill_null(0).to_numpy().astype(np.int8)
        if "is_major_hub_v" in df.columns
        else np.zeros(n_rows, dtype=np.int8)
    )

    m_prior_arr = np.asarray(
        compute_m_global(
            base_median_pace=base_median_pace,
            bin_15m=bins,
            is_weekend=wes,
            dist_from_center_km=dists,
            is_major_hub_v=hubs_v,
        ),
        dtype=np.float64,
    )

    # Group sums and counts per (corridor_segment_id, bin_15m, is_weekend)
    group_counts: dict[tuple[str, int, int], int] = {}
    group_sums: dict[tuple[str, int, int], float] = {}

    for i in range(n_rows):
        key = (str(cids[i]), int(bins[i]), int(wes[i]))
        group_counts[key] = group_counts.get(key, 0) + 1
        group_sums[key] = group_sums.get(key, 0.0) + float(paces[i])

    oof_paces = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        key = (str(cids[i]), int(bins[i]), int(wes[i]))
        n_loo = group_counts[key] - 1
        if n_loo <= 0:
            # Single observation: strictly zero self-influence, falls back to m_global
            oof_paces[i] = m_prior_arr[i]
        else:
            sum_loo = group_sums[key] - float(paces[i])
            mean_loo = sum_loo / n_loo
            w_loo = n_loo / (n_loo + k)
            eb_p = w_loo * mean_loo + (1.0 - w_loo) * m_prior_arr[i]
            oof_paces[i] = float(
                np.clip(eb_p, MIN_REALISTIC_PACE_S_PER_M, MAX_REALISTIC_PACE_S_PER_M)
            )

    oof_times = oof_paces * lens
    return observed_df.with_columns(
        eb_prior_pace_s_per_m=pl.Series("eb_prior_pace_s_per_m", oof_paces, dtype=pl.Float64),
        eb_prior_time_sec=pl.Series("eb_prior_time_sec", oof_times, dtype=pl.Float64),
    )


def build_eb_prior(
    observed_df: pl.DataFrame | None = None,
    catalog_df: pl.DataFrame | None = None,
) -> EmpiricalBayesPrior:
    """Compute empirical Bayes shrinkage tables and free-flow baselines."""
    t0 = time.time()
    if observed_df is None:
        if OBSERVED_SEGMENTS_PATH.exists():
            observed_df = pl.read_parquet(OBSERVED_SEGMENTS_PATH)
        else:
            observed_df = consolidate_observed_segments()

    if catalog_df is None and CATALOG_PATH.exists():
        catalog_df = pl.read_parquet(CATALOG_PATH)

    # 1. Compute empirical global pace distribution
    if observed_df is not None and not observed_df.is_empty():
        paces = observed_df["pace_sec_per_m"].to_numpy()
        valid_paces = paces[
            (paces >= MIN_REALISTIC_PACE_S_PER_M) & (paces <= MAX_REALISTIC_PACE_S_PER_M)
        ]
        if len(valid_paces) > 50:
            global_median = float(np.median(valid_paces))
            global_p15 = float(np.percentile(valid_paces, 15))
            global_p85 = float(np.percentile(valid_paces, 85))
        else:
            global_median = DEFAULT_MEDIAN_PACE_S_PER_M
            global_p15 = DEFAULT_MEDIAN_PACE_S_PER_M * 0.72
            global_p85 = DEFAULT_MEDIAN_PACE_S_PER_M * 1.35
    else:
        global_median = DEFAULT_MEDIAN_PACE_S_PER_M
        global_p15 = DEFAULT_MEDIAN_PACE_S_PER_M * 0.72
        global_p85 = DEFAULT_MEDIAN_PACE_S_PER_M * 1.35

    print(
        f"[OK] Global pace distribution: P15={global_p15:.3f} s/m, "
        f"Median={global_median:.3f} s/m (~{1.0 / (global_median * 0.27778):.1f} km/h), "
        f"P85={global_p85:.3f} s/m"
    )

    # 2. Build catalog info lookup and spatial base pace per corridor
    catalog_info: dict[str, dict[str, Any]] = {}
    corridor_base_pace: dict[str, float] = {}
    if catalog_df is not None:
        for r in catalog_df.iter_rows(named=True):
            cid = str(r["corridor_segment_id"])
            dist_km = float(r.get("dist_from_center_km", 12.0))
            is_hub = int(r.get("is_major_hub_v", 0))
            catalog_info[cid] = {
                "segment_length_m": float(r.get("segment_length_m", 250.0)),
                "dist_from_center_km": dist_km,
                "is_major_hub_v": is_hub,
                "mid_lat": float(r.get("mid_lat", 28.6139)),
                "mid_lon": float(r.get("mid_lon", 77.2090)),
                "bearing_rad": float(r.get("bearing_rad", 0.0)),
                "n_routes_sharing": int(r.get("n_routes_sharing", 1)),
                "is_major_hub_u": int(r.get("is_major_hub_u", 0)),
            }
            center_dist_factor = float(np.clip(1.08 - 0.008 * dist_km, 0.90, 1.15))
            hub_factor = 1.0 + 0.08 * is_hub
            corridor_base_pace[cid] = float(global_median * center_dist_factor * hub_factor)

    # 3. Compute de-seasonalized corridor base pace & bin-specific EB shrinkage
    eb_table: dict[tuple[str, int, int], float] = {}

    if observed_df is not None and not observed_df.is_empty():
        # 3a. Corridor-level de-seasonalized base pace shrinkage
        obs_cids = observed_df["corridor_segment_id"].to_list()
        obs_bins = observed_df["bin_15m"].to_numpy()
        obs_wes = observed_df["is_weekend"].to_numpy()
        obs_paces = observed_df["pace_sec_per_m"].to_numpy()
        obs_d_mult = np.asarray(get_diurnal_multiplier(obs_bins, obs_wes), dtype=np.float64)
        deseason_paces = obs_paces / np.maximum(0.5, obs_d_mult)

        corr_sum: dict[str, float] = {}
        corr_cnt: dict[str, int] = {}
        for cid, dp in zip(obs_cids, deseason_paces):
            s_cid = str(cid)
            corr_sum[s_cid] = corr_sum.get(s_cid, 0.0) + float(dp)
            corr_cnt[s_cid] = corr_cnt.get(s_cid, 0) + 1

        for s_cid, n_c in corr_cnt.items():
            mean_dp = corr_sum[s_cid] / n_c
            cat = catalog_info.get(s_cid)
            dist_km = cat["dist_from_center_km"] if cat else 12.0
            is_hub = cat["is_major_hub_v"] if cat else 0
            center_dist_factor = float(np.clip(1.08 - 0.008 * dist_km, 0.90, 1.15))
            hub_factor = 1.0 + 0.08 * is_hub
            spatial_base = float(global_median * center_dist_factor * hub_factor)

            w_c = n_c / (n_c + EB_K)
            shrunk_base = w_c * mean_dp + (1.0 - w_c) * spatial_base
            corridor_base_pace[s_cid] = float(
                np.clip(shrunk_base, MIN_REALISTIC_PACE_S_PER_M, MAX_REALISTIC_PACE_S_PER_M)
            )

        # 3b. Bin-specific Empirical Bayes shrinkage anchored to corridor diurnal expectation
        grouped = observed_df.group_by(
            ["corridor_segment_id", "bin_15m", "is_weekend"]
        ).agg([
            pl.len().alias("n"),
            pl.col("pace_sec_per_m").mean().alias("mean_pace"),
        ])

        for row in grouped.iter_rows(named=True):
            cid = str(row["corridor_segment_id"])
            bin_idx = int(row["bin_15m"])
            we = int(row["is_weekend"])
            n = float(row["n"])
            mean_pace = float(row["mean_pace"])

            base_p = corridor_base_pace.get(cid, global_median)
            d_mult = float(get_diurnal_multiplier(bin_idx, we))
            corridor_diurnal_prior = base_p * d_mult

            # Empirical Bayes shrinkage toward the corridor's diurnal curve
            w = n / (n + EB_K)
            raw_eb_pace = float(w * mean_pace + (1.0 - w) * corridor_diurnal_prior)
            # Bound bin-level deviation to preserve strict diurnal peak vs off-peak hierarchy
            eb_pace = float(
                np.clip(
                    raw_eb_pace,
                    corridor_diurnal_prior * 0.85,
                    corridor_diurnal_prior * 1.15,
                )
            )
            eb_table[(cid, bin_idx, we)] = float(
                np.clip(eb_pace, MIN_REALISTIC_PACE_S_PER_M, MAX_REALISTIC_PACE_S_PER_M)
            )

    # 4. Compute free_flow_time_sec per corridor segment (05:30 AM Free-Flow baseline)
    free_flow_times: dict[str, float] = {}
    dawn_mult = float(get_diurnal_multiplier(22, 0))  # 05:30 AM dawn
    for cid, cat in catalog_info.items():
        length_m = cat["segment_length_m"]
        base_p = corridor_base_pace.get(cid, global_median)
        dawn_pace = float(
            np.clip(base_p * dawn_mult, MIN_REALISTIC_PACE_S_PER_M, MAX_REALISTIC_PACE_S_PER_M)
        )
        free_flow_times[cid] = float(length_m * dawn_pace)

    prior = EmpiricalBayesPrior(
        global_median_pace=global_median,
        global_p15_pace=global_p15,
        global_p85_pace=global_p85,
        eb_table=eb_table,
        corridor_base_pace=corridor_base_pace,
        free_flow_times=free_flow_times,
        catalog_info=catalog_info,
    )

    print(
        f"[OK] Compiled Empirical Bayes prior ({len(eb_table):,} specific bins, "
        f"{len(free_flow_times):,} corridor free-flow values) in {time.time() - t0:.2f}s"
    )
    return prior


def main() -> int:
    print("=================================================================")
    print("Building Delhi GTFS Historical Baseline & Empirical Bayes Prior")
    print("=================================================================")

    # 1. Consolidate observed segments
    observed_df = consolidate_observed_segments()

    # 2. Build EB prior
    prior = build_eb_prior(observed_df)

    # 3. Add Leave-One-Out (Out-of-Fold) EB priors to observed segments and resave (Audit Fix 3.9)
    if not observed_df.is_empty():
        observed_df = compute_oof_eb_features(
            observed_df,
            base_median_pace=prior.global_median_pace,
        )
        observed_df.write_parquet(OBSERVED_SEGMENTS_PATH, compression="zstd")
        print(f"[OK] Updated {OBSERVED_SEGMENTS_PATH.name} with OOF EB prior features.")

    print("\n[SUCCESS] Historical baseline and Empirical Bayes priors ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
