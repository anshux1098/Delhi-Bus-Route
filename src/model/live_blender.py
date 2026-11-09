"""Live Anomaly Blender and Precomputed Horizon Matrix for Delhi Bus Transit.

Implements Audit Fixes 3.6, 3.7, 3.8, and 4.5:
1. Live Observation Tracker:
   - Aggregates recent matched pings (last 45 mins) per corridor segment.
   - Uses the median of log-pace log(pace_sec_per_m) across distinct vehicles (never arithmetic speed).
   - Computes log-residual anomaly against the historical model:
     r_live = log(t_live_obs) - log(m_hist(t_obs)).
2. Anomaly Decay & Blending:
   - For an arrival at future timestamp t_a:
     delta_t_mins = max(0.0, (t_a - t_obs) / 60.0)
     alpha = (n_veh / (n_veh + k)) * exp(-delta_t_mins / tau) with k = 3.0, tau = 35.0 mins
     t_hat(t_a) = m_hist(t_a) * exp(alpha * r_live)
3. 60-Second Horizon Matrix:
   - Precomputes all 15,130 corridor travel times across 12 15-minute bins (next 3 hours) in < 1 second.
   - Provides an O(1) linear interpolation helper get_segment_time for smooth, FIFO-compliant routing.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Ensure generic CPU architecture for Numba on Windows ARM64 / Snapdragon X
os.environ.setdefault("NUMBA_CPU_NAME", "generic")

from src.ingestion.hygiene import find_project_root

PROJECT_ROOT = find_project_root()

# Auto-reexec under .venv python if running under python without lightgbm
try:
    import lightgbm as lgb
except (ImportError, OSError):
    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists() and sys.executable.lower() != str(venv_py).lower():
        os.execv(str(venv_py), [str(venv_py), "-m", "src.model.live_blender"] + sys.argv[1:])
    raise

import numpy as np
import polars as pl

from src.model.historical_baseline import (
    DEFAULT_MEDIAN_PACE_S_PER_M,
    MAX_REALISTIC_PACE_S_PER_M,
    MIN_REALISTIC_PACE_S_PER_M,
    OBSERVED_SEGMENTS_PATH,
    compute_m_global,
    epoch_to_ist_features,
    get_diurnal_multiplier,
)
from src.model.lgbm_trainer import FEATURE_COLS, MODELS_PKL_PATH, load_travel_time_models

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Hyperparameters for Anomaly Decay (Audit Fix 3.6 & 4.5)
LIVE_K = 3.0           # Vehicle sample weight scale
LIVE_TAU_MINS = 35.0   # Temporal memory half-life decay in minutes
OBS_WINDOW_SEC = 45 * 60  # 45 minutes observation window


@dataclass
class LiveObservation:
    """Aggregated live traffic state for a single corridor segment."""

    corridor_segment_id: str
    t_obs: float              # Latest observation epoch in seconds
    n_veh: int                # Number of distinct vehicles observed in last 45m
    median_log_pace: float    # Median of log(pace_sec_per_m)
    live_pace: float          # exp(median_log_pace)
    live_time_obs: float      # segment_length_m * live_pace
    m_hist_obs: float         # Historical model expectation at t_obs
    r_live: float             # log(t_live_obs) - log(m_hist_obs)


class LiveStateTracker:
    """Tracks rolling 45-minute live vehicle pace observations across Delhi corridors."""

    def __init__(self, models_artifact: dict[str, Any] | None = None) -> None:
        self.artifacts = models_artifact or load_travel_time_models()
        self.model_expected = self.artifacts["model_expected"]
        self.calibration_factor = float(self.artifacts["calibration_factor"])
        self.catalog_df = self.artifacts["catalog_df"]
        self.eb_prior = self.artifacts["eb_prior"]
        self.catalog_lookup = self.eb_prior.catalog_info
        self.mu_tensor = self.artifacts.get("mu_tensor")
        self.sigma_tensor = self.artifacts.get("sigma_tensor")
        cids = self.catalog_df["corridor_segment_id"].to_list()
        self.corridor_to_idx = {cid: i for i, cid in enumerate(cids)}

        self.observations: dict[str, LiveObservation] = {}

    def get_historical_baseline(
        self,
        corridor_segment_id: str,
        epoch: float,
        length_m: float | None = None,
    ) -> float:
        """Evaluate historical baseline travel time m_hist(t) at a given epoch timestamp."""
        cid = str(corridor_segment_id)
        tf = epoch_to_ist_features(epoch)
        b_idx = int(tf["bin_15m"])
        we = int(tf["is_weekend"])
        if self.mu_tensor is not None and cid in self.corridor_to_idx:
            idx = self.corridor_to_idx[cid]
            return float(self.mu_tensor[idx, b_idx, we])
        m_hist = self.eb_prior.get_prior_time(
            cid, b_idx, we, length_m=length_m
        )
        return float(max(1.0, m_hist))

    def record_observation(
        self,
        corridor_segment_id: str,
        t_obs: float,
        t_live_obs: float,
        n_veh: int = 3,
        m_hist_obs: float | None = None,
    ) -> LiveObservation:
        """Record a direct live traversal observation on a corridor segment."""
        cid = str(corridor_segment_id)
        cat = self.catalog_lookup.get(cid)
        length_m = float(cat["segment_length_m"]) if cat else 250.0
        if m_hist_obs is None:
            m_hist_obs = self.get_historical_baseline(cid, t_obs, length_m=length_m)
        m_hist_obs = float(max(1.0, m_hist_obs))

        live_pace = float(t_live_obs / max(1.0, length_m))
        median_log_p = float(np.log(max(1e-6, live_pace)))
        r_live = float(np.clip(np.log(max(1.0, t_live_obs)) - np.log(m_hist_obs), -1.2, 1.5))

        obs = LiveObservation(
            corridor_segment_id=cid,
            t_obs=float(t_obs),
            n_veh=int(n_veh),
            median_log_pace=median_log_p,
            live_pace=live_pace,
            live_time_obs=float(t_live_obs),
            m_hist_obs=m_hist_obs,
            r_live=r_live,
        )
        self.observations[cid] = obs
        return obs

    def update_from_segments(
        self,
        segments_df: pl.DataFrame,
        current_epoch: int | float | None = None,
    ) -> int:
        """Update live observation states from a DataFrame of observed segments."""
        if segments_df.is_empty():
            return 0

        if current_epoch is None:
            current_epoch = float(segments_df["dep_time_u"].max())

        min_cutoff = current_epoch - OBS_WINDOW_SEC

        # Filter to recent on-route observations within last 45 minutes
        recent = segments_df.filter(
            (pl.col("dep_time_u") >= min_cutoff)
            & (pl.col("dep_time_u") <= current_epoch + 120.0)
            & (pl.col("pace_sec_per_m") >= MIN_REALISTIC_PACE_S_PER_M)
            & (pl.col("pace_sec_per_m") <= MAX_REALISTIC_PACE_S_PER_M)
        )

        if recent.is_empty():
            return 0

        # Group by corridor_segment_id
        grouped = recent.group_by("corridor_segment_id").agg([
            pl.col("vehicle_id").n_unique().alias("n_veh"),
            pl.col("dep_time_u").max().alias("t_obs"),
            pl.col("segment_length_m").first().alias("segment_length_m"),
            pl.col("pace_sec_per_m").alias("paces"),
        ])

        updated_count = 0
        for row in grouped.iter_rows(named=True):
            cid = str(row["corridor_segment_id"])
            n_veh = int(row["n_veh"])
            t_obs = float(row["t_obs"])
            length_m = float(row["segment_length_m"])
            paces = np.array(row["paces"], dtype=np.float64)

            # Audit Fix 3.7: Median of log-pace, never arithmetic speed
            log_paces = np.log(np.clip(paces, MIN_REALISTIC_PACE_S_PER_M, MAX_REALISTIC_PACE_S_PER_M))
            median_log_p = float(np.median(log_paces))
            live_pace = float(np.exp(median_log_p))
            t_live_obs = float(length_m * live_pace)

            # Evaluate historical baseline at t_obs
            m_hist_obs = self.get_historical_baseline(cid, t_obs, length_m=length_m)

            # Log-residual anomaly
            r_live = float(np.clip(np.log(t_live_obs) - np.log(m_hist_obs), -1.2, 1.5))

            self.observations[cid] = LiveObservation(
                corridor_segment_id=cid,
                t_obs=t_obs,
                n_veh=n_veh,
                median_log_pace=median_log_p,
                live_pace=live_pace,
                live_time_obs=t_live_obs,
                m_hist_obs=m_hist_obs,
                r_live=r_live,
            )
            updated_count += 1

        return updated_count

    def get_blended_time(
        self,
        corridor_segment_id: str,
        arrival_epoch: float,
        m_hist_ta: float | None = None,
    ) -> float:
        """Blend historical travel time with live anomaly decay (Audit Fixes 3.6 & 4.5)."""
        cid = str(corridor_segment_id)
        if m_hist_ta is None:
            m_hist_ta = self.get_historical_baseline(cid, arrival_epoch)

        obs = self.observations.get(cid)
        if obs is None or obs.n_veh <= 0:
            return float(m_hist_ta)

        delta_t_mins = max(0.0, (arrival_epoch - obs.t_obs) / 60.0)
        alpha = (obs.n_veh / (obs.n_veh + LIVE_K)) * np.exp(-delta_t_mins / LIVE_TAU_MINS)
        t_hat = m_hist_ta * np.exp(alpha * obs.r_live)
        return float(max(1.0, t_hat))


class PrecomputedHorizonTable(dict):
    """Dictionary holding the 60-second precomputed corridor lookup matrix."""

    def __init__(
        self,
        query_epoch: int,
        n_bins: int,
        corridor_to_idx: dict[str, int],
        matrix: np.ndarray,
        free_flow_times: np.ndarray,
        corridors: list[str],
        build_latency_s: float,
        sigma_matrix: np.ndarray | None = None,
    ) -> None:
        self.query_epoch = int(query_epoch)
        self.n_bins = int(n_bins)
        self.corridor_to_idx = corridor_to_idx
        self.matrix = matrix.astype(np.float32)
        self.free_flow_times = free_flow_times.astype(np.float32)
        self.corridors = corridors
        self.build_latency_s = float(build_latency_s)
        self.sigma_matrix = (
            sigma_matrix.astype(np.float32)
            if sigma_matrix is not None
            else np.maximum(1.0, 0.15 * self.matrix)
        )

        # Populate dictionary keys for dict compatibility
        super().__init__({
            "query_epoch": self.query_epoch,
            "n_bins": self.n_bins,
            "corridor_to_idx": self.corridor_to_idx,
            "matrix": self.matrix,
            "free_flow_times": self.free_flow_times,
            "corridors": self.corridors,
            "build_latency_s": self.build_latency_s,
            "sigma_matrix": self.sigma_matrix,
            "get_segment_time": self.get_segment_time,
            "get_segment_sigma": self.get_segment_sigma,
            "get_free_flow_time": self.get_free_flow_time,
        })

    def get_segment_time(
        self,
        corridor_segment_id: str,
        arrival_epoch: float | int,
    ) -> float:
        """O(1) linearly interpolated traversal time across adjacent 15-minute horizon bins."""
        idx = self.corridor_to_idx.get(str(corridor_segment_id))
        if idx is None:
            return 60.0

        # Fractional bin position
        u = (float(arrival_epoch) - self.query_epoch) / 900.0

        if u <= 0.0:
            return float(self.matrix[idx, 0])
        if u >= self.n_bins - 1:
            return float(self.matrix[idx, self.n_bins - 1])

        b0 = int(u)
        b1 = b0 + 1
        frac = u - b0

        # Linear interpolation between adjacent 15-minute bins
        t0 = float(self.matrix[idx, b0])
        t1 = float(self.matrix[idx, b1])
        return (1.0 - frac) * t0 + frac * t1

    def get_segment_sigma(
        self,
        corridor_segment_id: str,
        arrival_epoch: float | int,
    ) -> float:
        """O(1) linearly interpolated traversal time uncertainty sigma_e across adjacent horizon bins."""
        idx = self.corridor_to_idx.get(str(corridor_segment_id))
        if idx is None:
            return 10.0

        u = (float(arrival_epoch) - self.query_epoch) / 900.0

        if u <= 0.0:
            return float(self.sigma_matrix[idx, 0])
        if u >= self.n_bins - 1:
            return float(self.sigma_matrix[idx, self.n_bins - 1])

        b0 = int(u)
        b1 = b0 + 1
        frac = u - b0

        s0 = float(self.sigma_matrix[idx, b0])
        s1 = float(self.sigma_matrix[idx, b1])
        return (1.0 - frac) * s0 + frac * s1

    def get_free_flow_time(self, corridor_segment_id: str) -> float:
        """Return free-flow traversal time (P15 / 05:30 AM baseline)."""
        idx = self.corridor_to_idx.get(str(corridor_segment_id))
        if idx is not None:
            return float(self.free_flow_times[idx])
        return 45.0


def build_precomputed_horizon_table(
    query_epoch: int,
    n_bins: int = 12,
    tracker: LiveStateTracker | None = None,
    models_artifact: dict[str, Any] | None = None,
) -> PrecomputedHorizonTable:
    """Precompute all corridor travel times across n_bins 15-min intervals in < 50ms using dense tensors (Audit Fixes 0b & 0c).

    Directly slices from offline precomputed mu_tensor and sigma_tensor without any model.predict() calls.
    Returns:
        PrecomputedHorizonTable dict providing O(1) get_segment_time and get_segment_sigma interpolation.
    """
    t0 = time.time()
    artifacts = models_artifact or load_travel_time_models()
    catalog_df = artifacts["catalog_df"]
    cids = catalog_df["corridor_segment_id"].to_list()
    n_corridors = len(cids)
    corridor_to_idx = {cid: i for i, cid in enumerate(cids)}

    mu_tensor = artifacts.get("mu_tensor")
    sigma_tensor = artifacts.get("sigma_tensor")
    free_flow_arr = artifacts.get("free_flow_mu")

    if tracker is None:
        tracker = LiveStateTracker(models_artifact=artifacts)

    # 1. Direct slice from offline precomputed 96-bin tensors without calling model.predict
    bin_epochs = query_epoch + np.arange(n_bins, dtype=np.int64) * 900
    matrix = np.empty((n_corridors, n_bins), dtype=np.float32)
    sigma_matrix = np.empty((n_corridors, n_bins), dtype=np.float32)

    if mu_tensor is not None and sigma_tensor is not None:
        for b in range(n_bins):
            t_b = int(bin_epochs[b])
            tf_b = epoch_to_ist_features(t_b)
            b_idx = int(tf_b["bin_15m"])
            we_idx = int(tf_b["is_weekend"])
            matrix[:, b] = mu_tensor[:n_corridors, b_idx, we_idx]
            sigma_matrix[:, b] = sigma_tensor[:n_corridors, b_idx, we_idx]
    else:
        # Fallback if tensors not yet compiled
        prior = artifacts["eb_prior"]
        lens = catalog_df["segment_length_m"].to_numpy().astype(np.float32)
        for b in range(n_bins):
            t_b = int(bin_epochs[b])
            tf_b = epoch_to_ist_features(t_b)
            b_idx = int(tf_b["bin_15m"])
            we_idx = int(tf_b["is_weekend"])
            paces = np.array([prior.get_prior_pace(c, b_idx, we_idx) for c in cids], dtype=np.float32)
            matrix[:, b] = paces * lens
            sigma_matrix[:, b] = np.maximum(1.0, 0.15 * matrix[:, b])

    if free_flow_arr is None:
        free_flow_arr = np.array([
            artifacts["eb_prior"].get_free_flow_time(cid) for cid in cids
        ], dtype=np.float32)
    else:
        free_flow_arr = free_flow_arr[:n_corridors]

    # 2. Vectorized application of live anomalies if any observations are active
    if tracker.observations:
        obs_indices = []
        obs_t = []
        obs_w = []
        obs_r = []
        for cid, obs in tracker.observations.items():
            idx = corridor_to_idx.get(cid)
            if idx is not None and obs.n_veh > 0:
                obs_indices.append(idx)
                obs_t.append(obs.t_obs)
                obs_w.append(obs.n_veh / (obs.n_veh + LIVE_K))
                obs_r.append(obs.r_live)

        if obs_indices:
            idx_arr = np.array(obs_indices, dtype=np.int32)
            t_arr = np.array(obs_t, dtype=np.float64)[:, None]
            w_arr = np.array(obs_w, dtype=np.float64)[:, None]
            r_arr = np.array(obs_r, dtype=np.float64)[:, None]

            delta_t_mins = np.maximum(0.0, (bin_epochs[None, :] - t_arr) / 60.0)
            alpha_mat = w_arr * np.exp(-delta_t_mins / LIVE_TAU_MINS)
            scale_mat = np.exp(alpha_mat * r_arr).astype(np.float32)
            matrix[idx_arr, :] *= scale_mat

    latency = time.time() - t0
    table = PrecomputedHorizonTable(
        query_epoch=query_epoch,
        n_bins=n_bins,
        corridor_to_idx=corridor_to_idx,
        matrix=matrix,
        free_flow_times=free_flow_arr,
        corridors=cids,
        build_latency_s=latency,
        sigma_matrix=sigma_matrix,
    )

    print(
        f"[OK] Precomputed horizon matrix ({n_corridors:,} corridors x {n_bins} bins = "
        f"{n_corridors * n_bins:,} cells) in {latency * 1000.0:.2f}ms (Target: < 50ms)"
    )
    return table


def main() -> int:
    print("=================================================================")
    print("Testing Live Anomaly Blender and Horizon Lookup Matrix")
    print("=================================================================")

    # 1. Load models and initialize LiveStateTracker
    artifacts = load_travel_time_models()
    tracker = LiveStateTracker(models_artifact=artifacts)

    # 2. Ingest recent real observed segments into tracker
    if OBSERVED_SEGMENTS_PATH.exists():
        obs_df = pl.read_parquet(OBSERVED_SEGMENTS_PATH)
        n_obs = tracker.update_from_segments(obs_df)
        print(f"[OK] Ingested {n_obs:,} live corridor anomalies into LiveStateTracker.")

    # 3. Build Precomputed Horizon Matrix
    now_epoch = int(time.time())
    table = build_precomputed_horizon_table(
        query_epoch=now_epoch,
        n_bins=12,
        tracker=tracker,
        models_artifact=artifacts,
    )

    # 4. Test O(1) linear interpolation speed
    sample_cid = table.corridors[0]
    t_interp_start = time.perf_counter()
    n_queries = 100_000
    test_arrivals = now_epoch + np.linspace(0, 10800, n_queries)
    for t_arr in test_arrivals[:10]:
        t_est = table.get_segment_time(sample_cid, t_arr)
    t_interp_elapsed = time.perf_counter() - t_interp_start

    print(f"\nSample Corridor: {sample_cid}")
    print(f"Free-flow time : {table.get_free_flow_time(sample_cid):.1f}s")
    for delta_mins in [0, 15, 30, 60, 120, 180]:
        arr = now_epoch + delta_mins * 60
        t_val = table.get_segment_time(sample_cid, arr)
        s_val = table.get_segment_sigma(sample_cid, arr)
        print(f"Arrival +{delta_mins:>3} mins : {t_val:.1f}s (sigma={s_val:.1f}s)")

    print(f"\n[OK] Linear interpolation test: 10 queries in {t_interp_elapsed * 1e6:.1f} microseconds.")
    assert table.build_latency_s < 0.050, f"Matrix generation took {table.build_latency_s * 1000.0:.2f}ms (> 50ms target)"
    print("\n[SUCCESS] Live anomaly blender and horizon matrix verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
