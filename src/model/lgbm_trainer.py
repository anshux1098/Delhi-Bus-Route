"""LightGBM Travel-Time Model Training and Journey-Level Calibration.

Trains two LightGBM models for bus corridor travel time:
1. Model 1 (Expected Travel Time): Objective 'gamma', 400 trees, learning_rate 0.05,
   with post-hoc scalar calibration ensuring zero systematic journey-level bias (Audit Fix 3.1).
2. Model 2 (P90 Commuter Buffer Time): Objective 'quantile' (alpha=0.85), 300 trees,
   predicting the reliable upper travel time bound (Audit Fix 3.1).

Training set:
- Observed segments from observed_segments.parquet (sample_weight = 5.0)
- Synthetic diurnal anchor grid across segment_catalog.parquet over 96 bin_15m slots (sample_weight = 0.5)

Saved to data/processed/travel_time_models.pkl.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

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
        os.execv(str(venv_py), [str(venv_py), "-m", "src.model.lgbm_trainer"] + sys.argv[1:])
    raise

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

from src.model.historical_baseline import (
    CATALOG_PATH,
    OBSERVED_SEGMENTS_PATH,
    EmpiricalBayesPrior,
    build_eb_prior,
    compute_m_global,
    compute_oof_eb_features,
    consolidate_observed_segments,
)

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_PKL_PATH = PROCESSED_DIR / "travel_time_models.pkl"
TENSORS_NPZ_PATH = PROCESSED_DIR / "baseline_tensors.npz"

FEATURE_COLS = [
    "segment_length_m",
    "bin_15m",
    "day_of_week",
    "is_weekend",
    "mid_lat",
    "mid_lon",
    "dist_from_center_km",
    "bearing_rad",
    "n_routes_sharing",
    "is_major_hub_u",
    "is_major_hub_v",
    "eb_prior_pace_s_per_m",
    "eb_prior_time_sec",
]

# Monotone constraints corresponding to FEATURE_COLS:
# +1 on segment_length_m, is_major_hub_v, eb_prior_pace_s_per_m, eb_prior_time_sec
# -1 on is_weekend
MONOTONE_CONSTRAINTS = [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1]


def get_shrunk_q85_and_sigma(
    cids: list[str] | np.ndarray,
    mu_hat: np.ndarray,
    r_pred_85: np.ndarray,
    corr_counts: dict[str, int],
    r_global_85: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Empirical Bayes shrinkage to predicted ratio and derive P85 and edge sigma.

    Formulas:
    w = n_obs / (n_obs + 5.0)
    r_shrunk = w * r_pred_85 + (1.0 - w) * r_global_85
    q85_hat = mu_hat * np.maximum(1.03, r_shrunk)
    sigma_e = np.maximum(0.05 * mu_hat, (q85_hat - mu_hat) / 1.036)
    """
    n_obs = np.array([corr_counts.get(str(c), 0) for c in cids], dtype=np.float64)
    w = n_obs / (n_obs + 5.0)
    r_shrunk = w * r_pred_85 + (1.0 - w) * r_global_85
    q85_hat = mu_hat * np.maximum(1.03, r_shrunk)
    sigma_e = np.maximum(0.05 * mu_hat, (q85_hat - mu_hat) / 1.036)
    return q85_hat, sigma_e


def build_synthetic_anchor_grid(
    catalog_df: pl.DataFrame,
    eb_prior: EmpiricalBayesPrior,
) -> pl.DataFrame:
    """Generate diurnal anchor grid across catalog corridors covering peak/off-peak slots."""
    t0 = time.time()
    # Representative 15-minute bins covering all diurnal traffic regimes:
    # 02:30 (Late night), 05:30 (Free flow dawn), 07:30 (Early morn), 09:15 (AM peak),
    # 11:30 (Midday morn), 14:00 (Midday base), 16:30 (Afternoon build),
    # 18:30 (PM peak), 20:30 (Late eve), 22:30 (Night wind down)
    anchor_bins = [10, 22, 30, 37, 46, 56, 66, 74, 82, 90]
    anchor_regimes = [
        # (bin_15m, day_of_week, is_weekend)
        (b, 2, 0) for b in anchor_bins  # Weekday (Wednesday)
    ] + [
        (b, 6, 1) for b in [10, 22, 44, 56, 74, 88]  # Weekend (Sunday)
    ]

    n_corridors = catalog_df.height
    n_regimes = len(anchor_regimes)
    total_anchors = n_corridors * n_regimes

    cids_rep = []
    lens_rep = []
    lats_rep = []
    lons_rep = []
    dists_rep = []
    bears_rep = []
    routes_rep = []
    hub_u_rep = []
    hub_v_rep = []

    cids = catalog_df["corridor_segment_id"].to_list()
    lens = catalog_df["segment_length_m"].to_numpy().astype(np.float64)
    lats = catalog_df["mid_lat"].to_numpy().astype(np.float64)
    lons = catalog_df["mid_lon"].to_numpy().astype(np.float64)
    dists = catalog_df["dist_from_center_km"].to_numpy().astype(np.float64)
    bears = catalog_df["bearing_rad"].to_numpy().astype(np.float64)
    routes = catalog_df["n_routes_sharing"].to_numpy().astype(np.int32)
    hub_u = catalog_df["is_major_hub_u"].to_numpy().astype(np.int8)
    hub_v = catalog_df["is_major_hub_v"].to_numpy().astype(np.int8)

    bins_arr = np.empty(total_anchors, dtype=np.int32)
    dow_arr = np.empty(total_anchors, dtype=np.int8)
    we_arr = np.empty(total_anchors, dtype=np.int8)
    prior_pace_arr = np.empty(total_anchors, dtype=np.float64)

    ptr = 0
    for b_idx, dow, we in anchor_regimes:
        sz = n_corridors
        bins_arr[ptr : ptr + sz] = b_idx
        dow_arr[ptr : ptr + sz] = dow
        we_arr[ptr : ptr + sz] = we

        for i, cid in enumerate(cids):
            prior_pace_arr[ptr + i] = eb_prior.get_prior_pace(cid, b_idx, we)

        cids_rep.extend(cids)
        lens_rep.append(lens)
        lats_rep.append(lats)
        lons_rep.append(lons)
        dists_rep.append(dists)
        bears_rep.append(bears)
        routes_rep.append(routes)
        hub_u_rep.append(hub_u)
        hub_v_rep.append(hub_v)

        ptr += sz

    lens_full = np.concatenate(lens_rep)
    prior_time_arr = prior_pace_arr * lens_full

    anchor_df = pl.DataFrame({
        "corridor_segment_id": cids_rep,
        "segment_length_m": lens_full,
        "bin_15m": bins_arr,
        "day_of_week": dow_arr,
        "is_weekend": we_arr,
        "mid_lat": np.concatenate(lats_rep),
        "mid_lon": np.concatenate(lons_rep),
        "dist_from_center_km": np.concatenate(dists_rep),
        "bearing_rad": np.concatenate(bears_rep),
        "n_routes_sharing": np.concatenate(routes_rep),
        "is_major_hub_u": np.concatenate(hub_u_rep),
        "is_major_hub_v": np.concatenate(hub_v_rep),
        "eb_prior_pace_s_per_m": prior_pace_arr,
        "eb_prior_time_sec": prior_time_arr,
        "travel_time_sec": prior_time_arr,
        "sample_weight": np.full(total_anchors, 0.5, dtype=np.float32),
        "is_synthetic": np.ones(total_anchors, dtype=np.int8),
    })

    print(f"[OK] Generated {total_anchors:,} synthetic anchor rows in {time.time() - t0:.2f}s")
    return anchor_df


def compute_oof_gamma_predictions(
    train_real: pl.DataFrame,
    n_splits: int = 5,
) -> np.ndarray:
    """Compute K-fold out-of-fold calibrated mean predictions on train_real grouped by vehicle_id (Audit Fix 1a)."""
    t0 = time.time()
    unique_vehs = sorted(train_real["vehicle_id"].unique().to_list())
    veh_to_fold = {v: (idx % n_splits) for idx, v in enumerate(unique_vehs)}
    fold_arr = np.array([veh_to_fold[v] for v in train_real["vehicle_id"].to_list()], dtype=np.int32)

    X_train_real = train_real.select(FEATURE_COLS).to_numpy()
    y_train_real = train_real["travel_time_sec"].to_numpy().astype(np.float64)
    w_train_real = train_real["sample_weight"].to_numpy().astype(np.float64)

    mu_oof = np.empty(len(y_train_real), dtype=np.float64)

    for f in range(n_splits):
        trn_mask = fold_arr != f
        val_mask = fold_arr == f

        X_f_trn, y_f_trn, w_f_trn = X_train_real[trn_mask], y_train_real[trn_mask], w_train_real[trn_mask]
        X_f_val, y_f_val = X_train_real[val_mask], y_train_real[val_mask]

        fold_model = lgb.LGBMRegressor(
            objective="gamma",
            n_estimators=300,
            learning_rate=0.06,
            num_leaves=63,
            min_child_samples=30,
            monotone_constraints=MONOTONE_CONSTRAINTS,
            random_state=42 + f,
            n_jobs=-1,
            verbose=-1,
        )
        fold_model.fit(X_f_trn, y_f_trn, sample_weight=w_f_trn)
        raw_val = fold_model.predict(X_f_val)
        cal_factor = np.sum(y_f_val) / max(1.0, np.sum(raw_val))
        mu_oof[val_mask] = raw_val * cal_factor

    print(f"[OK] Computed 5-fold OOF Gamma mean predictions on {len(mu_oof):,} rows in {time.time() - t0:.2f}s")
    return mu_oof


def prepare_training_dataset(
    observed_df: pl.DataFrame,
    catalog_df: pl.DataFrame,
    eb_prior: EmpiricalBayesPrior,
    return_three_way: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame] | tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Combine real observed data with synthetic anchor grid, split into train, calib, and test.

    Audit Fixes:
    - 60/20/20 deterministic vehicle split (train 60%, calib 20%, untouched test 20%).
    - Span-level weighting: sample_weight = span_weight * 5.0 so each matched ping-pair has weight 5.0.
    """
    t0 = time.time()

    # Prepare real observed data with sample_weight = span_weight * 5.0
    real_df = observed_df.filter(pl.col("travel_time_sec") > 0.0)
    if "span_weight" in real_df.columns:
        real_df = real_df.with_columns(
            sample_weight=(pl.col("span_weight") * 5.0).cast(pl.Float32),
            is_synthetic=pl.lit(0, dtype=pl.Int8),
        )
    else:
        real_df = real_df.with_columns(
            sample_weight=pl.lit(5.0, dtype=pl.Float32),
            is_synthetic=pl.lit(0, dtype=pl.Int8),
        )

    # Deterministic 60/20/20 vehicle split:
    unique_vehs = sorted(real_df["vehicle_id"].unique().to_list())
    test_veh_set = {v for idx, v in enumerate(unique_vehs) if idx % 5 == 0}   # 20% untouched test
    calib_veh_set = {v for idx, v in enumerate(unique_vehs) if idx % 5 == 1}  # 20% post-hoc calib
    train_veh_set = {v for idx, v in enumerate(unique_vehs) if idx % 5 in (2, 3, 4)}  # 60% training

    train_real = real_df.filter(pl.col("vehicle_id").is_in(train_veh_set))
    calib_real = real_df.filter(pl.col("vehicle_id").is_in(calib_veh_set))
    test_real = real_df.filter(pl.col("vehicle_id").is_in(test_veh_set))

    # Recompute Leave-One-Out (OOF) EB features strictly inside train_real
    train_eb_prior = build_eb_prior(train_real, catalog_df)
    train_real = compute_oof_eb_features(
        train_real,
        catalog_df=catalog_df,
        base_median_pace=train_eb_prior.global_median_pace,
    )

    # Evaluate validation & test features strictly out-of-fold from train_eb_prior
    def _apply_eb_prior(df: pl.DataFrame) -> pl.DataFrame:
        cids = df["corridor_segment_id"].to_list()
        bins = df["bin_15m"].to_list()
        wes = df["is_weekend"].to_list()
        lens = df["segment_length_m"].to_numpy().astype(np.float64)
        prior_paces = np.array(
            [train_eb_prior.get_prior_pace(c, b, w) for c, b, w in zip(cids, bins, wes)],
            dtype=np.float64,
        )
        return df.with_columns(
            eb_prior_pace_s_per_m=pl.Series("eb_prior_pace_s_per_m", prior_paces, dtype=pl.Float64),
            eb_prior_time_sec=pl.Series("eb_prior_time_sec", prior_paces * lens, dtype=pl.Float64),
        )

    calib_real = _apply_eb_prior(calib_real)
    test_real = _apply_eb_prior(test_real)

    # Make sure all FEATURE_COLS are present
    for col in FEATURE_COLS:
        if col not in train_real.columns:
            raise ValueError(f"Missing required feature column in observed data: {col}")

    # Compute true 5-fold OOF calibrated mean predictions on train_real
    mu_oof = compute_oof_gamma_predictions(train_real, n_splits=5)
    y_train_real = train_real["travel_time_sec"].to_numpy().astype(np.float64)
    # Dimensionless residual ratio r_i = y_i / mu_oof_i
    r_train_real = y_train_real / np.maximum(1.0, mu_oof)
    r_global_85 = float(np.percentile(r_train_real, 85))

    train_real = train_real.with_columns(
        ratio_target=pl.Series("ratio_target", r_train_real, dtype=pl.Float64)
    )

    # Synthetic anchor grid (sample_weight = 0.5) built from train_eb_prior
    anchor_df = build_synthetic_anchor_grid(catalog_df, train_eb_prior)
    anchor_df = anchor_df.with_columns(
        ratio_target=pl.lit(r_global_85, dtype=pl.Float64),
        vehicle_id=pl.lit("", dtype=pl.Utf8),
        trip_run_id=pl.lit("", dtype=pl.Utf8),
        dep_time_u=pl.lit(0.0, dtype=pl.Float64),
    )

    calib_real = calib_real.with_columns(ratio_target=pl.lit(r_global_85, dtype=pl.Float64))
    test_real = test_real.with_columns(ratio_target=pl.lit(r_global_85, dtype=pl.Float64))

    # Combine train real + synthetic anchors, preserving metadata columns
    keep_cols = (
        FEATURE_COLS
        + ["corridor_segment_id", "vehicle_id", "trip_run_id", "dep_time_u"]
        + ["travel_time_sec", "ratio_target", "sample_weight", "is_synthetic"]
    )
    train_combined = pl.concat([
        train_real.select(keep_cols),
        anchor_df.select(keep_cols),
    ], how="vertical")

    print(
        f"[OK] Training dataset: {train_combined.height:,} rows "
        f"({train_real.height:,} real [span_weighted, r_global_85={r_global_85:.3f}] + "
        f"{anchor_df.height:,} synthetic [weight=0.5]). "
        f"Calib: {calib_real.height:,} rows (20%). Untouched Test: {test_real.height:,} rows (20%). "
        f"Prepared in {time.time() - t0:.2f}s"
    )

    if return_three_way:
        return train_combined, calib_real, test_real
    return train_combined, test_real


def fit_ratio_space_kernels(
    train_and_calib_df: pl.DataFrame,
    model_expected: lgb.LGBMRegressor,
    calibration_factor: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fit ratio-space live kernels gamma_self and gamma_cross via least-squares grid search.

    Upgrades:
    - gamma_self (gamma_0_self, tau_self): fitted on same-vehicle trajectories.
    - gamma_cross (gamma_0_cross, tau_cross): fitted on cross-vehicle corridor observations.
    """
    t0 = time.time()
    trips = train_and_calib_df.partition_by("trip_run_id", as_dict=True)
    sample_trips = list(trips.values())[:3000]

    self_pairs = []
    for t_df in sample_trips:
        if t_df.height >= 2:
            t_sorted = t_df.sort("dep_time_u")
            y = t_sorted["travel_time_sec"].to_numpy().astype(np.float64)
            t_dep = t_sorted["dep_time_u"].to_numpy().astype(np.float64)
            X = t_sorted.select(FEATURE_COLS).to_numpy()
            mu = model_expected.predict(X) * calibration_factor
            q = y / np.maximum(1.0, mu)
            for i in range(len(y) - 1):
                dt_min = (t_dep[i + 1] - t_dep[i]) / 60.0
                if 0.5 <= dt_min <= 60.0:
                    self_pairs.append((dt_min, q[i], q[i + 1]))

    gamma_self = (0.75, 10.0)
    if self_pairs:
        dt_arr = np.array([p[0] for p in self_pairs])
        q0_arr = np.array([p[1] for p in self_pairs])
        q1_arr = np.array([p[2] for p in self_pairs])
        y_target = q1_arr - 1.0
        q0_dev = q0_arr - 1.0

        best_mse = float("inf")
        best_tau = 10.0
        best_gamma0 = 0.75
        for tau in [5.0, 10.0, 20.0, 35.0, 60.0]:
            x = q0_dev * np.exp(-dt_arr / tau)
            denom = float(np.sum(x**2))
            if denom > 0:
                g0 = float(np.clip(np.sum(x * y_target) / denom, 0.10, 0.95))
                mse = float(np.mean((y_target - g0 * x)**2))
                if mse < best_mse:
                    best_mse = mse
                    best_tau = tau
                    best_gamma0 = g0
        gamma_self = (round(best_gamma0, 4), round(best_tau, 1))

    # Cross-vehicle corridor observations across distinct vehicles
    corr_groups = (
        train_and_calib_df.group_by("corridor_segment_id")
        .len()
        .filter(pl.col("len") >= 3)
    )
    sample_cids = corr_groups["corridor_segment_id"][:1500].to_list()
    sub_df = train_and_calib_df.filter(pl.col("corridor_segment_id").is_in(sample_cids))

    cross_pairs = []
    sub_partition = sub_df.partition_by("corridor_segment_id", as_dict=True)
    for cid, c_df in sub_partition.items():
        if c_df.height >= 2:
            c_sorted = c_df.sort("dep_time_u")
            y = c_sorted["travel_time_sec"].to_numpy().astype(np.float64)
            t_dep = c_sorted["dep_time_u"].to_numpy().astype(np.float64)
            vehs = c_sorted["vehicle_id"].to_list()
            X = c_sorted.select(FEATURE_COLS).to_numpy()
            mu = model_expected.predict(X) * calibration_factor
            q = y / np.maximum(1.0, mu)
            for i in range(min(10, len(y) - 1)):
                for j in range(i + 1, min(i + 4, len(y))):
                    if vehs[i] != vehs[j]:
                        dt_min = (t_dep[j] - t_dep[i]) / 60.0
                        if 1.0 <= dt_min <= 60.0:
                            cross_pairs.append((dt_min, q[i], q[j]))

    gamma_cross = (0.55, 20.0)
    if cross_pairs:
        dt_arr = np.array([p[0] for p in cross_pairs])
        q0_arr = np.array([p[1] for p in cross_pairs])
        q1_arr = np.array([p[2] for p in cross_pairs])
        y_target = q1_arr - 1.0
        q0_dev = q0_arr - 1.0

        best_mse = float("inf")
        best_tau = 20.0
        best_gamma0 = 0.55
        for tau in [5.0, 10.0, 20.0, 35.0, 60.0]:
            x = q0_dev * np.exp(-dt_arr / tau)
            denom = float(np.sum(x**2))
            if denom > 0:
                g0 = float(np.clip(np.sum(x * y_target) / denom, 0.10, 0.90))
                mse = float(np.mean((y_target - g0 * x)**2))
                if mse < best_mse:
                    best_mse = mse
                    best_tau = tau
                    best_gamma0 = g0
        gamma_cross = (round(best_gamma0, 4), round(best_tau, 1))

    print(
        f"[OK] Fitted ratio-space kernels in {time.time() - t0:.2f}s: "
        f"gamma_self=(gamma_0={gamma_self[0]}, tau={gamma_self[1]}m), "
        f"gamma_cross=(gamma_0={gamma_cross[0]}, tau={gamma_cross[1]}m)"
    )
    return gamma_self, gamma_cross


def calibrate_journey_p85_kappa(
    calib_real_df: pl.DataFrame,
    model_expected: lgb.LGBMRegressor,
    model_p90: lgb.LGBMRegressor,
    calibration_factor: float,
    corr_counts: dict[str, int],
    r_global_85: float,
    rho: float,
    target_coverage: float = 0.850,
) -> float:
    """Calibrate journey-level P85 factor kappa on calib_vehicles to hit target_coverage."""
    t0 = time.time()
    X = calib_real_df.select(FEATURE_COLS).to_numpy()
    y = calib_real_df["travel_time_sec"].to_numpy().astype(np.float64)
    cids = calib_real_df["corridor_segment_id"].to_list()

    mu = model_expected.predict(X) * calibration_factor
    r_pred = model_p90.predict(X)
    _, sigma = get_shrunk_q85_and_sigma(cids, mu, r_pred, corr_counts, r_global_85)

    calib_with_preds = calib_real_df.with_columns(
        mu_hat=pl.Series("mu_hat", mu),
        sigma_hat=pl.Series("sigma_hat", sigma),
    )

    trips = calib_with_preds.partition_by("trip_run_id", as_dict=True)
    journeys = []
    for _, t_df in trips.items():
        if t_df.height >= 4:
            s_trip = t_df.sort("dep_time_u")
            down = s_trip.slice(1)  # Causal downstream journey
            y_act = float(down["travel_time_sec"].sum())
            mus = down["mu_hat"].to_numpy()
            sigs = down["sigma_hat"].to_numpy()
            mu_J = float(np.sum(mus))
            s1 = float(np.sum(sigs))
            s2 = float(np.sum(sigs**2))
            V_J = (1.0 - rho) * s2 + rho * (s1**2)
            s_J = math.sqrt(math.log(1.0 + V_J / (mu_J**2)))
            journeys.append((y_act, mu_J, s_J))

    if len(journeys) < 30:
        return 1.036

    best_kappa = 1.036
    min_diff = float("inf")
    for k in np.linspace(0.05, 1.50, 146):
        cov = float(
            np.mean([
                y_act <= mu_J * math.exp(-0.5 * s_J**2 + k * s_J)
                for y_act, mu_J, s_J in journeys
            ])
        )
        diff = abs(cov - target_coverage)
        if diff < min_diff:
            min_diff = diff
            best_kappa = float(k)

    print(
        f"[OK] Calibrated journey P85 kappa = {best_kappa:.3f} on {len(journeys):,} multi-stop journeys "
        f"in {time.time() - t0:.2f}s (coverage: {target_coverage*100:.1f}%)"
    )
    return round(best_kappa, 3)


def train_travel_time_models(
    train_df: pl.DataFrame,
    calib_real_df: pl.DataFrame,
    test_real_df_or_prior: pl.DataFrame | EmpiricalBayesPrior | None = None,
    eb_prior: EmpiricalBayesPrior | None = None,
    catalog_df: pl.DataFrame | None = None,
) -> tuple[
    lgb.LGBMRegressor,
    lgb.LGBMRegressor,
    float,
    dict[str, Any],
    dict[str, int],
    float,
    float,
    tuple[float, float],
    tuple[float, float],
    float,
]:
    """Train Expected Travel Time (Gamma) and Ratio Quantile models with 60/20/20 split and ratio kernels."""
    # Handle backwards-compatible 4-argument call (train_df, val_real_df, eb_prior, catalog_df)
    if isinstance(test_real_df_or_prior, EmpiricalBayesPrior):
        catalog_df = eb_prior
        eb_prior = test_real_df_or_prior
        test_real_df = calib_real_df
    else:
        test_real_df = test_real_df_or_prior if test_real_df_or_prior is not None else calib_real_df

    X_train = train_df.select(FEATURE_COLS).to_numpy()
    y_train = train_df["travel_time_sec"].to_numpy().astype(np.float64)
    r_train = train_df["ratio_target"].to_numpy().astype(np.float64)
    w_train = train_df["sample_weight"].to_numpy().astype(np.float64)

    X_calib = calib_real_df.select(FEATURE_COLS).to_numpy()
    y_calib = calib_real_df["travel_time_sec"].to_numpy().astype(np.float64)

    X_test = test_real_df.select(FEATURE_COLS).to_numpy()
    y_test = test_real_df["travel_time_sec"].to_numpy().astype(np.float64)
    test_cids = test_real_df["corridor_segment_id"].to_list()

    # Precompute real observation count per corridor for EB shrinkage
    real_train_df = train_df.filter(pl.col("is_synthetic") == 0)
    corr_counts = dict(real_train_df.group_by("corridor_segment_id").len().iter_rows())
    r_real_train = real_train_df["ratio_target"].to_numpy()
    r_global_85 = float(np.percentile(r_real_train, 85))

    # 1. Model 1: Expected Travel Time (Gamma objective with monotone constraints)
    print("\nTraining Model 1 (Expected Travel Time — Gamma Objective, 400 trees)...")
    t1 = time.time()
    model_expected = lgb.LGBMRegressor(
        objective="gamma",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model_expected.fit(X_train, y_train, sample_weight=w_train)
    print(f"[OK] Model 1 trained in {time.time() - t1:.2f}s")

    # Post-hoc scalar calibration strictly on calib_real (Audit Fix 0.4 & Fix 3.1)
    raw_calib_pred = model_expected.predict(X_calib)
    sum_actual_calib = float(np.sum(y_calib))
    sum_pred_raw_calib = float(np.sum(raw_calib_pred))
    calibration_factor = sum_actual_calib / max(1.0, sum_pred_raw_calib)

    print(
        f"[OK] Post-hoc calibration factor: {calibration_factor:.4f} "
        f"(Calib Raw Bias: {sum_pred_raw_calib / sum_actual_calib:.4f} -> Calibrated Bias: "
        f"{np.sum(raw_calib_pred * calibration_factor) / sum_actual_calib:.4f})"
    )

    # 2. Model 2: Ratio Quantile Model (alpha=0.85, 300 trees, Audit Fix 1a)
    print("\nTraining Model 2 (P85 Ratio Quantile Model — alpha=0.85, 300 trees on r_i = y_i / mu_oof)...")
    t2 = time.time()
    model_p90 = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.85,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model_p90.fit(X_train, r_train, sample_weight=w_train)
    print(f"[OK] Model 2 trained in {time.time() - t2:.2f}s")

    # 3. Validation / Test Metrics on Untouched 20% Test Split
    test_pred_cal = model_expected.predict(X_test) * calibration_factor
    r_test_pred_85 = model_p90.predict(X_test)
    test_q85, test_sigma = get_shrunk_q85_and_sigma(
        test_cids, test_pred_cal, r_test_pred_85, corr_counts, r_global_85
    )

    sum_actual_test = float(np.sum(y_test))
    mae = float(mean_absolute_error(y_test, test_pred_cal))
    mape = float(mean_absolute_percentage_error(y_test, test_pred_cal)) * 100.0
    rmse = float(root_mean_squared_error(y_test, test_pred_cal))
    bias_ratio = float(np.sum(test_pred_cal) / sum_actual_test)
    coverage_overall = float(np.mean(y_test <= test_q85))

    t_33 = float(np.percentile(test_pred_cal, 33.3))
    t_66 = float(np.percentile(test_pred_cal, 66.7))
    m_low = test_pred_cal <= t_33
    m_mid = (test_pred_cal > t_33) & (test_pred_cal <= t_66)
    m_high = test_pred_cal > t_66

    cov_low = float(np.mean(y_test[m_low] <= test_q85[m_low]))
    cov_mid = float(np.mean(y_test[m_mid] <= test_q85[m_mid]))
    cov_high = float(np.mean(y_test[m_high] <= test_q85[m_high]))

    # 4. Estimate unclipped empirical intra-trip residual correlation rho from standardized residuals
    common_cols = [c for c in real_train_df.columns if c in calib_real_df.columns]
    tc_real = pl.concat(
        [real_train_df.select(common_cols), calib_real_df.select(common_cols)],
        how="vertical",
    )
    X_tc = tc_real.select(FEATURE_COLS).to_numpy()
    y_tc = tc_real["travel_time_sec"].to_numpy().astype(np.float64)
    tc_cids = tc_real["corridor_segment_id"].to_list()

    mu_tc = model_expected.predict(X_tc) * calibration_factor
    r_tc_pred = model_p90.predict(X_tc)
    _, sigma_tc = get_shrunk_q85_and_sigma(tc_cids, mu_tc, r_tc_pred, corr_counts, r_global_85)
    z_tc = (y_tc - mu_tc) / np.maximum(0.5, sigma_tc)

    rho = 0.417
    if "trip_run_id" in tc_real.columns and "dep_time_u" in tc_real.columns:
        tc_with_z = tc_real.with_columns(z_e=pl.Series("z_e", z_tc))
        grouped_trips = tc_with_z.sort(["trip_run_id", "dep_time_u"]).group_by(
            "trip_run_id", maintain_order=True
        )
        z_curr_list = []
        z_next_list = []
        for _, trip_df in grouped_trips:
            if trip_df.height >= 2:
                zs = trip_df["z_e"].to_numpy()
                z_curr_list.extend(zs[:-1])
                z_next_list.extend(zs[1:])

        if len(z_curr_list) >= 50:
            r_mat = np.corrcoef(z_curr_list, z_next_list)
            rho_emp = float(r_mat[0, 1])
            rho = float(np.clip(rho_emp, 0.05, 0.95)) if not np.isnan(rho_emp) else 0.417

    # 5. Fit ratio-space live kernels gamma_self and gamma_cross
    gamma_self, gamma_cross = fit_ratio_space_kernels(tc_real, model_expected, calibration_factor)

    # 6. Fit journey P85 calibration factor kappa on calib_real_df multi-stop trips
    kappa = calibrate_journey_p85_kappa(
        calib_real_df, model_expected, model_p90, calibration_factor, corr_counts, r_global_85, rho
    )

    metrics = {
        "mae_sec": mae,
        "mape_pct": mape,
        "rmse_sec": rmse,
        "bias_ratio": bias_ratio,
        "coverage_overall": coverage_overall,
        "coverage_low_tercile": cov_low,
        "coverage_mid_tercile": cov_mid,
        "coverage_high_tercile": cov_high,
        "rho": 0.600,
        "rho_unclipped": rho,
        "calibration_factor": calibration_factor,
        "r_global_85": r_global_85,
        "val_samples": len(y_test),
        "gamma_self": gamma_self,
        "gamma_cross": gamma_cross,
        "kappa": kappa,
    }

    print("\n=================================================================")
    print("Metrics on Untouched 20% Held-Out Test Vehicle Segments:")
    print("=================================================================")
    print(f"MAE (seconds)          : {mae:.2f}s")
    print(f"MAPE                   : {mape:.2f}%")
    print(f"RMSE (seconds)         : {rmse:.2f}s")
    print(f"Mean Bias Ratio        : {bias_ratio:.4f} (Target: [0.98, 1.02])")
    print(f"Overall Coverage P85   : {coverage_overall * 100.0:.2f}% (Target: 85% +/- 3%)")
    print(f"  - Low Tercile Cov    : {cov_low * 100.0:.2f}%")
    print(f"  - Mid Tercile Cov    : {cov_mid * 100.0:.2f}%")
    print(f"  - High Tercile Cov   : {cov_high * 100.0:.2f}%")
    print(f"Unclipped Intra-Trip rho: {rho:.3f}")
    print(f"Journey P85 kappa      : {kappa:.3f}")
    print(f"Live gamma_self        : gamma_0={gamma_self[0]}, tau={gamma_self[1]}m")
    print(f"Live gamma_cross       : gamma_0={gamma_cross[0]}, tau={gamma_cross[1]}m")
    print("=================================================================")

    return (
        model_expected,
        model_p90,
        calibration_factor,
        metrics,
        corr_counts,
        r_global_85,
        rho,
        gamma_self,
        gamma_cross,
        kappa,
    )


def precompute_baseline_tensors(
    model_expected: lgb.LGBMRegressor,
    model_p90: lgb.LGBMRegressor,
    calibration_factor: float,
    eb_prior: EmpiricalBayesPrior,
    catalog_df: pl.DataFrame,
    corr_counts: dict[str, int],
    r_global_85: float,
    out_npz_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute full dense 96-bin tensors for expected travel time and edge sigma (Audit Fixes 0b & 0c).

    Output shapes:
    mu_tensor: (n_corridors, 96, 2)
    sigma_tensor: (n_corridors, 96, 2)
    free_flow_mu: (n_corridors,)
    """
    t0 = time.time()
    n_corridors = catalog_df.height
    cids = catalog_df["corridor_segment_id"].to_list()
    lens = catalog_df["segment_length_m"].to_numpy().astype(np.float32)
    lats = catalog_df["mid_lat"].to_numpy().astype(np.float32)
    lons = catalog_df["mid_lon"].to_numpy().astype(np.float32)
    dists = catalog_df["dist_from_center_km"].to_numpy().astype(np.float32)
    bears = catalog_df["bearing_rad"].to_numpy().astype(np.float32)
    routes = catalog_df["n_routes_sharing"].to_numpy().astype(np.int32)
    hub_u = catalog_df["is_major_hub_u"].to_numpy().astype(np.int8)
    hub_v = catalog_df["is_major_hub_v"].to_numpy().astype(np.int8)

    # 05:30 AM dawn bin is bin 22, weekday we=0
    dawn_pace = compute_m_global(
        base_median_pace=eb_prior.global_median_pace,
        bin_15m=22,
        is_weekend=0,
        dist_from_center_km=dists,
        is_major_hub_v=hub_v,
    )
    free_flow_mu = (lens * dawn_pace).astype(np.float32)

    mu_tensor = np.empty((n_corridors, 96, 2), dtype=np.float32)
    sigma_tensor = np.empty((n_corridors, 96, 2), dtype=np.float32)

    # Pre-extract corridor shrinkage factors
    n_obs_arr = np.array([corr_counts.get(str(c), 0) for c in cids], dtype=np.float32)
    w_arr = (n_obs_arr / (n_obs_arr + 5.0)).astype(np.float32)
    inv_w_r_global = ((1.0 - w_arr) * r_global_85).astype(np.float32)

    # Process by weekend flag: we=0 (Wednesday, dow=2), we=1 (Sunday, dow=6)
    for we in [0, 1]:
        dow = 2 if we == 0 else 6
        # Batch in chunks of 8 bins (8 * 15,130 = 121,040 rows) for optimal L2/L3 CPU cache efficiency
        chunk_size_bins = 8
        for b_start in range(0, 96, chunk_size_bins):
            b_end = min(96, b_start + chunk_size_bins)
            n_chunk_bins = b_end - b_start
            n_rows_chunk = n_chunk_bins * n_corridors

            # Replicate static features
            rep_lens = np.tile(lens, n_chunk_bins)
            rep_lats = np.tile(lats, n_chunk_bins)
            rep_lons = np.tile(lons, n_chunk_bins)
            rep_dists = np.tile(dists, n_chunk_bins)
            rep_bears = np.tile(bears, n_chunk_bins)
            rep_routes = np.tile(routes, n_chunk_bins)
            rep_hub_u = np.tile(hub_u, n_chunk_bins)
            rep_hub_v = np.tile(hub_v, n_chunk_bins)

            bins_list = []
            paces_list = []
            for b_idx in range(b_start, b_end):
                bins_list.append(np.full(n_corridors, b_idx, dtype=np.int32))
                b_paces = np.array(
                    [eb_prior.get_prior_pace(c, b_idx, we) for c in cids], dtype=np.float32
                )
                paces_list.append(b_paces)

            rep_bins = np.concatenate(bins_list)
            rep_paces = np.concatenate(paces_list)
            rep_times = rep_paces * rep_lens

            X_chunk = np.column_stack([
                rep_lens,
                rep_bins,
                np.full(n_rows_chunk, dow, dtype=np.int8),
                np.full(n_rows_chunk, we, dtype=np.int8),
                rep_lats,
                rep_lons,
                rep_dists,
                rep_bears,
                rep_routes,
                rep_hub_u,
                rep_hub_v,
                rep_paces,
                rep_times,
            ]).astype(np.float32)

            raw_mu_chunk = model_expected.booster_.predict(X_chunk, num_threads=8)
            cal_mu_chunk = (raw_mu_chunk * calibration_factor).astype(np.float32)
            r_pred_chunk = model_p90.booster_.predict(X_chunk, num_threads=8).astype(np.float32)

            # Assign back to tensors
            for i, b_idx in enumerate(range(b_start, b_end)):
                sub_slice = slice(i * n_corridors, (i + 1) * n_corridors)
                mu_b = cal_mu_chunk[sub_slice]
                r_b = r_pred_chunk[sub_slice]

                # EB shrinkage and sigma derivation
                r_shrunk_b = w_arr * r_b + inv_w_r_global
                q85_b = mu_b * np.maximum(1.03, r_shrunk_b)
                sigma_b = np.maximum(0.05 * mu_b, (q85_b - mu_b) / 1.036)

                mu_tensor[:, b_idx, we] = mu_b
                sigma_tensor[:, b_idx, we] = sigma_b

    target_npz = out_npz_path or TENSORS_NPZ_PATH
    target_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_npz,
        mu_tensor=mu_tensor,
        sigma_tensor=sigma_tensor,
        free_flow_mu=free_flow_mu,
        corridors=np.array(cids),
    )
    print(
        f"[OK] Precomputed full dense baseline tensors ({n_corridors:,} corridors x 96 bins x 2 weekend flags = "
        f"{mu_tensor.size:,} cells) saved to {target_npz.name} in {time.time() - t0:.2f}s"
    )
    return mu_tensor, sigma_tensor, free_flow_mu


def save_models_artifact(
    model_expected: lgb.LGBMRegressor,
    model_p90: lgb.LGBMRegressor,
    calibration_factor: float,
    metrics: dict[str, Any],
    eb_prior: EmpiricalBayesPrior,
    catalog_df: pl.DataFrame,
    mu_tensor: np.ndarray,
    sigma_tensor: np.ndarray,
    free_flow_mu: np.ndarray,
    corr_counts: dict[str, int],
    r_global_85: float,
    rho: float,
    gamma_self: tuple[float, float] = (0.75, 10.0),
    gamma_cross: tuple[float, float] = (0.55, 20.0),
    kappa: float = 1.036,
    out_path: Path | None = None,
) -> Path:
    """Serialize trained models and dense tensors to pickle."""
    target_path = out_path or MODELS_PKL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    artifact = {
        "model_expected": model_expected,
        "model_p90": model_p90,
        "calibration_factor": calibration_factor,
        "metrics": metrics,
        "feature_cols": FEATURE_COLS,
        "eb_prior": eb_prior,
        "free_flow_times": eb_prior.free_flow_times,
        "free_flow_mu": free_flow_mu,
        "catalog_df": catalog_df,
        "mu_tensor": mu_tensor,
        "sigma_tensor": sigma_tensor,
        "corr_counts": corr_counts,
        "r_global_85": r_global_85,
        "rho": 0.600,
        "rho_unclipped": rho,
        "gamma_self": gamma_self,
        "gamma_cross": gamma_cross,
        "kappa": kappa,
    }

    with open(target_path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = target_path.stat().st_size / (1024.0 * 1024.0)
    print(f"[OK] Saved travel_time_models.pkl ({file_size_mb:.2f} MB in {time.time() - t0:.2f}s)")
    return target_path


def load_travel_time_models(path: Path | None = None) -> dict[str, Any]:
    """Load cached travel time models and dense tensors."""
    target_path = path or MODELS_PKL_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {target_path}. Run training first.")
    with open(target_path, "rb") as f:
        art = pickle.load(f)

    # If tensors not in pkl, load from npz
    if "mu_tensor" not in art and TENSORS_NPZ_PATH.exists():
        data = np.load(TENSORS_NPZ_PATH)
        art["mu_tensor"] = data["mu_tensor"]
        art["sigma_tensor"] = data["sigma_tensor"]
        art["free_flow_mu"] = data["free_flow_mu"]
    return art


def main() -> int:
    print("=================================================================")
    print("Training LightGBM Travel-Time Models for Delhi Bus Transit")
    print("=================================================================")

    if not CATALOG_PATH.exists():
        from src.model.segment_catalog import build_catalog, load_stops
        from src.matching.shape_indexer import load_static_index
        idx = load_static_index()
        stops = load_stops()
        cat_df, _ = build_catalog(idx, stops)
        cat_df.write_parquet(CATALOG_PATH)
    else:
        cat_df = pl.read_parquet(CATALOG_PATH)

    # 1. Ingest observed segments and build EB prior
    if not OBSERVED_SEGMENTS_PATH.exists():
        obs_df = consolidate_observed_segments()
    else:
        obs_df = pl.read_parquet(OBSERVED_SEGMENTS_PATH)

    eb_prior = build_eb_prior(obs_df, cat_df)

    # 2. Build training dataset with 60/20/20 vehicle split
    train_df, calib_df, test_df = prepare_training_dataset(
        obs_df, cat_df, eb_prior, return_three_way=True
    )

    # 3. Train models
    (
        model_exp,
        model_p90,
        calib_factor,
        metrics,
        corr_counts,
        r_global_85,
        rho,
        gamma_self,
        gamma_cross,
        kappa,
    ) = train_travel_time_models(train_df, calib_df, test_df, eb_prior, cat_df)

    # 4. Precompute full 96-bin tensors offline
    mu_tensor, sigma_tensor, free_flow_mu = precompute_baseline_tensors(
        model_exp,
        model_p90,
        calib_factor,
        eb_prior,
        cat_df,
        corr_counts,
        r_global_85,
    )

    # 5. Save models artifact
    save_models_artifact(
        model_exp,
        model_p90,
        calib_factor,
        metrics,
        eb_prior,
        cat_df,
        mu_tensor,
        sigma_tensor,
        free_flow_mu,
        corr_counts,
        r_global_85,
        rho,
        gamma_self=gamma_self,
        gamma_cross=gamma_cross,
        kappa=kappa,
    )

    print("\n[SUCCESS] Model training and dense tensor precomputation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
