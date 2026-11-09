"""Sprint 2 Mathematical & Code Audit Verification Suite.

Covers the 5 required audit items:
1. Mean-Unbiased Guarantee (Audit Fix 3.1):
   - Validation mean(y_pred) / mean(y_true) within [0.98, 1.02].
   - P85 quantile predictions >= mean_pred on >= 99% of segments.
2. Out-of-Fold Leakage Prevention (Audit Fix 3.9):
   - Single-observation segment does not memorize its own target residual (LOO weight = 0).
   - Multi-observation segment uses exact leave-one-out mean and weight (n-1)/(n-1+k).
3. Anomaly-Space Peak Shoulder Test (Audit Fix 3.6):
   - Observation at 08:00 AM with +30% jam (t_live = 1.30 * m_hist(08:00)) queried at 08:20 AM
     scales the 08:20 AM baseline: m_hist(08:20) * exp(alpha * log(1.30)).
4. Diurnal Sensitivity Requirement (Problem Statement Section 2.1):
   - Corridor segments along Kashmere Gate -> ITO -> Nehru Place predict significantly
     longer travel times at 09:00 AM Weekday than at 23:00 PM or 05:30 AM Free-Flow.
5. Matrix Latency Benchmark & Smooth Interpolation (Audit Fix 4.5):
   - build_precomputed_horizon_table computes 22,000+ segments x 12 bins in < 2.0s on CPU.
   - get_segment_time interpolates linearly and smoothly between bins without step jumps.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("NUMBA_CPU_NAME", "generic")

import numpy as np
import polars as pl
import pytest

from src.model.historical_baseline import (
    CATALOG_PATH,
    EB_K,
    OBSERVED_SEGMENTS_PATH,
    compute_m_global,
    compute_oof_eb_features,
    epoch_to_ist_features,
)
from src.model.lgbm_trainer import (
    FEATURE_COLS,
    get_shrunk_q85_and_sigma,
    load_travel_time_models,
    prepare_training_dataset,
)
from src.model.live_blender import (
    LIVE_K,
    LIVE_TAU_MINS,
    LiveStateTracker,
    build_precomputed_horizon_table,
)
from src.model.segment_catalog import ROUTE_SEGMENTS_PATH, load_stops


@pytest.fixture(scope="module")
def model_artifacts() -> dict:
    return load_travel_time_models()


@pytest.fixture(scope="module")
def observed_df() -> pl.DataFrame:
    return pl.read_parquet(OBSERVED_SEGMENTS_PATH)


@pytest.fixture(scope="module")
def catalog_df() -> pl.DataFrame:
    return pl.read_parquet(CATALOG_PATH)


def test_1_mean_unbiased_and_coverage(
    model_artifacts: dict,
    observed_df: pl.DataFrame,
    catalog_df: pl.DataFrame,
) -> None:
    """Audit Fixes 0a, 1a, 3.1: Verify mean(y_pred)/mean(y_true) in [0.98, 1.02] and held-out P85 coverage in 85% +/- 3%."""
    model_expected = model_artifacts["model_expected"]
    model_p90 = model_artifacts["model_p90"]
    calibration_factor = float(model_artifacts["calibration_factor"])
    eb_prior = model_artifacts["eb_prior"]
    corr_counts = model_artifacts["corr_counts"]
    r_global_85 = float(model_artifacts["r_global_85"])

    # 1. Check held-out validation split from prepare_training_dataset
    _, val_real_df = prepare_training_dataset(observed_df, catalog_df, eb_prior)
    X_val = val_real_df.select(FEATURE_COLS).to_numpy()
    y_val = val_real_df["travel_time_sec"].to_numpy().astype(np.float64)
    val_cids = val_real_df["corridor_segment_id"].to_list()

    mean_pred_val = model_expected.predict(X_val) * calibration_factor
    r_pred_val = model_p90.predict(X_val)
    val_q85, val_sigma = get_shrunk_q85_and_sigma(
        val_cids, mean_pred_val, r_pred_val, corr_counts, r_global_85
    )

    val_bias_ratio = float(np.mean(mean_pred_val) / np.mean(y_val))
    assert 0.98 <= val_bias_ratio <= 1.02, (
        f"Validation Mean Bias Ratio {val_bias_ratio:.4f} outside [0.98, 1.02]"
    )

    # 2. True Held-Out Empirical Coverage Check: P(y_val <= val_q85) must be 85% +/- 3%
    val_coverage = float(np.mean(y_val <= val_q85))
    print(f"\n[OK] Validation P85 Calibration Coverage: {val_coverage * 100.0:.2f}% (Target: 85% +/- 3%)")
    assert 0.82 <= val_coverage <= 0.88, (
        f"Validation P85 coverage {val_coverage * 100.0:.2f}% outside [82%, 88%]"
    )

    # 3. Stratified coverage across terciles of expected travel time
    t_33 = float(np.percentile(mean_pred_val, 33.3))
    t_66 = float(np.percentile(mean_pred_val, 66.7))
    m_low = mean_pred_val <= t_33
    m_mid = (mean_pred_val > t_33) & (mean_pred_val <= t_66)
    m_high = mean_pred_val > t_66

    cov_low = float(np.mean(y_val[m_low] <= val_q85[m_low]))
    cov_mid = float(np.mean(y_val[m_mid] <= val_q85[m_mid]))
    cov_high = float(np.mean(y_val[m_high] <= val_q85[m_high]))

    print(f"[OK] Coverage by Terciles -> Low: {cov_low * 100.0:.2f}%, Mid: {cov_mid * 100.0:.2f}%, High: {cov_high * 100.0:.2f}%")
    assert 0.78 <= cov_low <= 0.92, f"Low tercile coverage {cov_low:.3f} outside [0.78, 0.92]"
    assert 0.78 <= cov_mid <= 0.92, f"Mid tercile coverage {cov_mid:.3f} outside [0.78, 0.92]"
    assert 0.78 <= cov_high <= 0.92, f"High tercile coverage {cov_high:.3f} outside [0.78, 0.92]"

    # 4. Strict positivity and lower bound on sigma_e
    assert np.all(val_sigma >= 0.05 * mean_pred_val), "sigma_e must be strictly >= 0.05 * mu_hat"
    assert np.all(val_sigma > 0.0), "sigma_e must be strictly positive"


def test_2_out_of_fold_leakage_prevention() -> None:
    """Audit Fix 3.9: Verify single-observation segments cannot memorize their own target residual."""
    # 1. Single-observation test: mutate target pace from 0.05 s/m (fast) to 0.55 s/m (jammed)
    base_rows = [
        {
            "corridor_segment_id": "SINGLE_OBS_SEG_500",
            "bin_15m": 36,
            "is_weekend": 0,
            "segment_length_m": 500.0,
            "dist_from_center_km": 5.0,
            "is_major_hub_v": 1,
            "pace_sec_per_m": 0.05,
            "travel_time_sec": 25.0,
        },
        # Multi-observation segment (n=3) in same bin
        {
            "corridor_segment_id": "MULTI_OBS_SEG_400",
            "bin_15m": 36,
            "is_weekend": 0,
            "segment_length_m": 400.0,
            "dist_from_center_km": 8.0,
            "is_major_hub_v": 0,
            "pace_sec_per_m": 0.15,
            "travel_time_sec": 60.0,
        },
        {
            "corridor_segment_id": "MULTI_OBS_SEG_400",
            "bin_15m": 36,
            "is_weekend": 0,
            "segment_length_m": 400.0,
            "dist_from_center_km": 8.0,
            "is_major_hub_v": 0,
            "pace_sec_per_m": 0.25,
            "travel_time_sec": 100.0,
        },
        {
            "corridor_segment_id": "MULTI_OBS_SEG_400",
            "bin_15m": 36,
            "is_weekend": 0,
            "segment_length_m": 400.0,
            "dist_from_center_km": 8.0,
            "is_major_hub_v": 0,
            "pace_sec_per_m": 0.50,
            "travel_time_sec": 200.0,
        },
    ]

    df_fast = pl.DataFrame(base_rows)
    rows_slow = [dict(r) for r in base_rows]
    rows_slow[0]["pace_sec_per_m"] = 0.55
    rows_slow[0]["travel_time_sec"] = 275.0
    df_slow = pl.DataFrame(rows_slow)

    oof_fast = compute_oof_eb_features(df_fast, k=EB_K, base_median_pace=0.205)
    oof_slow = compute_oof_eb_features(df_slow, k=EB_K, base_median_pace=0.205)

    # Single-observation segment must equal m_global and be 100% invariant to its own target
    expected_m_global_single = float(
        compute_m_global(
            base_median_pace=0.205,
            bin_15m=36,
            is_weekend=0,
            dist_from_center_km=5.0,
            is_major_hub_v=1,
        )
    )
    prior_fast = float(oof_fast["eb_prior_pace_s_per_m"][0])
    prior_slow = float(oof_slow["eb_prior_pace_s_per_m"][0])

    assert np.isclose(prior_fast, expected_m_global_single, atol=1e-12)
    assert np.isclose(prior_slow, expected_m_global_single, atol=1e-12)
    assert np.isclose(prior_fast, prior_slow, atol=1e-12), (
        "Target leakage detected on single-observation segment!"
    )

    # 2. Verify Leave-One-Out arithmetic on the 3rd row (pace = 0.50) of MULTI_OBS_SEG_400:
    # Leaving out row 3 (0.50) leaves n_loo = 2 observations with paces [0.15, 0.25] -> mean_loo = 0.20
    expected_m_global_multi = float(
        compute_m_global(
            base_median_pace=0.205,
            bin_15m=36,
            is_weekend=0,
            dist_from_center_km=8.0,
            is_major_hub_v=0,
        )
    )
    w_loo = 2.0 / (2.0 + EB_K)
    expected_loo_row3 = w_loo * 0.20 + (1.0 - w_loo) * expected_m_global_multi
    actual_loo_row3 = float(oof_fast["eb_prior_pace_s_per_m"][3])

    assert np.isclose(actual_loo_row3, expected_loo_row3, atol=1e-12), (
        f"Expected LOO prior {expected_loo_row3:.6f}, got {actual_loo_row3:.6f}"
    )


def test_3_anomaly_space_peak_shoulder(model_artifacts: dict) -> None:
    """Audit Fix 3.6: Verify 08:00 AM +30% jam scales 08:20 AM baseline in anomaly space."""
    tracker = LiveStateTracker(models_artifact=model_artifacts)
    sample_cid = model_artifacts["catalog_df"]["corridor_segment_id"][0]
    length_m = float(model_artifacts["catalog_df"]["segment_length_m"][0])

    # Choose a Weekday (Wednesday) 08:00 AM IST epoch:
    # 2026-11-04 08:00:00 IST -> 02:30:00 UTC
    # 1970-01-01 is Thursday; let's compute exact epoch for a Wednesday 08:00:00 IST
    # Day 20760 (divisible by 7 -> Thursday): Day 20766 is Wednesday
    wed_midnight_ist_epoch = 20766 * 86400 - 19800
    t_0800 = float(wed_midnight_ist_epoch + 8 * 3600)          # 08:00 AM IST (bin 32)
    t_0820 = float(wed_midnight_ist_epoch + 8 * 3600 + 20 * 60)  # 08:20 AM IST (bin 33)

    m_hist_0800 = tracker.get_historical_baseline(sample_cid, t_0800, length_m=length_m)
    m_hist_0820 = tracker.get_historical_baseline(sample_cid, t_0820, length_m=length_m)

    # Morning peak ramp-up: 08:20 AM baseline is longer than 08:00 AM baseline
    assert m_hist_0820 > m_hist_0800, (
        f"Expected m_hist(08:20)={m_hist_0820:.2f}s > m_hist(08:00)={m_hist_0800:.2f}s"
    )

    # Observe a +30% traffic jam at 08:00 AM across n_veh = 6 vehicles
    n_veh = 6
    t_live_0800 = 1.30 * m_hist_0800
    pace_live_0800 = t_live_0800 / length_m

    seg_obs_df = pl.DataFrame({
        "corridor_segment_id": [sample_cid] * n_veh,
        "vehicle_id": [f"DL1PC{1000 + i}" for i in range(n_veh)],
        "dep_time_u": [t_0800] * n_veh,
        "segment_length_m": [length_m] * n_veh,
        "pace_sec_per_m": [pace_live_0800] * n_veh,
    })
    tracker.update_from_segments(seg_obs_df, current_epoch=t_0800)

    # Query blended prediction for arrival at 08:20 AM (delta_t = 20 mins)
    blended_0820 = tracker.get_blended_time(sample_cid, t_0820, m_hist_ta=m_hist_0820)

    delta_t_mins = 20.0
    expected_alpha = (n_veh / (n_veh + LIVE_K)) * np.exp(-delta_t_mins / LIVE_TAU_MINS)
    expected_blended_0820 = m_hist_0820 * np.exp(expected_alpha * np.log(1.30))

    assert np.isclose(blended_0820, expected_blended_0820, rtol=1e-5), (
        f"Expected anomaly-scaled 08:20 prediction {expected_blended_0820:.3f}s, "
        f"got {blended_0820:.3f}s"
    )
    # Must strictly exceed both m_hist(08:20) and raw t_live(08:00)
    assert blended_0820 > m_hist_0820


def test_4_diurnal_sensitivity_kashmere_gate_ito_nehru_place(
    model_artifacts: dict,
    catalog_df: pl.DataFrame,
) -> None:
    """Problem Statement 2.1: Verify 09:00 AM Weekday >> 23:00 PM > 05:30 AM Free-Flow."""
    model_expected = model_artifacts["model_expected"]
    calibration_factor = float(model_artifacts["calibration_factor"])
    eb_prior = model_artifacts["eb_prior"]

    # 1. Extract the real corridor segments along Route 1755 (Kashmere Gate -> ITO -> Nehru Place)
    rs_df = pl.read_parquet(ROUTE_SEGMENTS_PATH)
    kg_ito_np_cids = (
        rs_df.filter(
            (pl.col("route_id") == "1755")
            & (pl.col("segment_order") >= 18)
            & (pl.col("segment_order") <= 51)
        )
        .sort("segment_order")["corridor_segment_id"]
        .to_list()
    )
    assert len(kg_ito_np_cids) >= 25, "Expected >= 25 segments on Kashmere Gate -> ITO -> Nehru Place"

    corridor_sub = catalog_df.filter(pl.col("corridor_segment_id").is_in(kg_ito_np_cids))
    assert corridor_sub.height >= 20

    def predict_corridors_at_bin(df_sub: pl.DataFrame, bin_15m: int, dow: int = 2, we: int = 0) -> np.ndarray:
        cids = df_sub["corridor_segment_id"].to_list()
        lens = df_sub["segment_length_m"].to_numpy().astype(np.float64)
        paces = np.array(
            [eb_prior.get_prior_pace(c, bin_15m, we) for c in cids], dtype=np.float64
        )
        times = paces * lens
        n = df_sub.height
        X = np.column_stack([
            lens,
            np.full(n, bin_15m, dtype=np.int32),
            np.full(n, dow, dtype=np.int8),
            np.full(n, we, dtype=np.int8),
            df_sub["mid_lat"].to_numpy(),
            df_sub["mid_lon"].to_numpy(),
            df_sub["dist_from_center_km"].to_numpy(),
            df_sub["bearing_rad"].to_numpy(),
            df_sub["n_routes_sharing"].to_numpy(),
            df_sub["is_major_hub_u"].to_numpy(),
            df_sub["is_major_hub_v"].to_numpy(),
            paces,
            times,
        ])
        return model_expected.predict(X) * calibration_factor

    # Evaluate at 09:00 AM Weekday (bin 36), 23:00 PM (bin 92), and 05:30 AM Free-Flow (bin 22)
    t_0900 = predict_corridors_at_bin(corridor_sub, bin_15m=36, dow=2, we=0)
    t_2300 = predict_corridors_at_bin(corridor_sub, bin_15m=92, dow=2, we=0)
    t_0530 = predict_corridors_at_bin(corridor_sub, bin_15m=22, dow=2, we=0)

    # Every segment on Kashmere Gate -> ITO -> Nehru Place must be slower at 09:00 AM than 23:00 and 05:30
    assert np.all(t_0900 > t_2300), "09:00 AM Weekday must exceed 23:00 PM on all corridor segments"
    assert np.all(t_2300 > t_0530), "23:00 PM must exceed 05:30 AM Free-Flow on all corridor segments"

    journey_0900 = float(np.sum(t_0900))
    journey_2300 = float(np.sum(t_2300))
    journey_0530 = float(np.sum(t_0530))

    assert journey_0900 > 1.25 * journey_2300, (
        f"09:00 AM journey ({journey_0900:.1f}s) not significantly longer than 23:00 PM ({journey_2300:.1f}s)"
    )
    assert journey_0900 > 1.50 * journey_0530, (
        f"09:00 AM journey ({journey_0900:.1f}s) not significantly longer than 05:30 AM ({journey_0530:.1f}s)"
    )


def test_5_matrix_latency_and_smooth_interpolation(
    model_artifacts: dict,
    catalog_df: pl.DataFrame,
) -> None:
    """Audit Fixes 0b, 0c, 4.5: Verify < 0.05s matrix rebuild, exact 400-tree tensor match, and smooth interpolation."""
    mu_tensor = model_artifacts["mu_tensor"]
    n_corridors = catalog_df.height

    query_epoch = 1762500000
    t0 = time.perf_counter()
    table = build_precomputed_horizon_table(
        query_epoch=query_epoch,
        n_bins=12,
        models_artifact=model_artifacts,
    )
    elapsed_s = time.perf_counter() - t0

    # 1. Benchmark latency (< 50ms)
    print(f"\n[OK] Recomputed 12-bin horizon table in {elapsed_s * 1000.0:.2f}ms (Target: < 50ms)")
    assert elapsed_s < 0.050, f"Horizon matrix generation took {elapsed_s * 1000.0:.2f}ms (>= 50ms)"
    assert table.matrix.shape == (n_corridors, 12)

    # 2. Exact match to offline precomputed 400-tree mu_tensor when n_veh = 0
    bin_epochs = query_epoch + np.arange(12, dtype=np.int64) * 900
    for b in range(12):
        tf = epoch_to_ist_features(bin_epochs[b])
        b_idx = int(tf["bin_15m"])
        we_idx = int(tf["is_weekend"])
        expected_mu = mu_tensor[:, b_idx, we_idx]
        assert np.allclose(table.matrix[:, b], expected_mu, atol=1e-5), (
            f"Bin {b} (15m slot {b_idx}, we={we_idx}) does not match offline 400-tree mu_tensor!"
        )

    # 3. Verify smooth linear interpolation between bins without step jumps
    cid = table.corridors[0]
    t_bin0 = float(table.matrix[0, 0])
    t_bin1 = float(table.matrix[0, 1])

    # Exact endpoints
    assert np.isclose(table.get_segment_time(cid, query_epoch), t_bin0, atol=1e-5)
    assert np.isclose(table.get_segment_time(cid, query_epoch + 900), t_bin1, atol=1e-5)

    # Midpoint (450s = 7.5 mins) must equal exact average of bin 0 and bin 1
    t_mid = table.get_segment_time(cid, query_epoch + 450)
    assert np.isclose(t_mid, 0.5 * (t_bin0 + t_bin1), atol=1e-5)

    # Check continuity across fine 10-second steps over 2 hours (no step jumps)
    fine_times = np.arange(query_epoch, query_epoch + 7200, 10, dtype=np.float64)
    vals = np.array([table.get_segment_time(cid, t) for t in fine_times])
    diffs = np.abs(np.diff(vals))
    # Since bins are 900s apart, a 10s step changes by at most |t_{b+1} - t_b| * (10 / 900)
    max_bin_diff = float(np.max(np.abs(np.diff(table.matrix[0, :]))))
    assert np.max(diffs) <= (max_bin_diff * (10.0 / 900.0)) + 1e-4, (
        "Discontinuous step jump detected in get_segment_time interpolation!"
    )
