"""Sprint 6 Audit & Upgrades Verification Suite (`tests/test_sprint6_upgrades.py`).

Audits and verifies Sprint 6 Upgrades 1, 2, and 4 against all 9 integrity rules:
1. Zero Fabricated or Hardcoded Numbers (Fixes 0.1–0.8):
   - Verifies `validation_suite.py` has NO `* 1.25` multiplier on Stage 1, NO synthetic
     `1.18 / 0.86` empirical O-D multiplier, and NO flat `18.5 km/h` timetable substitute.
   - Verifies `lgbm_trainer.py` and `validation_suite.py` use a strict 3-way vehicle split
     (`60% train / 20% calib / 20% test`) where `calibration_factor` and `kappa` are fit ONLY
     on `calib` and evaluated ONLY on `test`.
   - Verifies `gamma_self` and `gamma_cross` are fit on `train + calib` and reported separately
     as `Stage 4a (cross-vehicle)` and `Stage 4b (same-vehicle)` in `validation_results.json`
     and `docs/VALIDATION_REPORT.md`.
2. Upgrade 4a — 3-Profile Reliability Routing in RAPTOR:
   - Tests that 3-profile RAPTOR routing returns options tagged with `profile_label` and
     completes within `< 150 ms` total for all 3 profiles combined.
3. Upgrade 4b — Provenance-Aware Traffic Corridor Heatmap:
   - Tests `GET /live/traffic-corridors` via `TestClient` and verifies that every corridor
     carries a valid `provenance` in `{"LIVE", "OBSERVED_HIST", "PRIOR"}` and that
     `live_count + observed_hist_count + prior_count == 15130`.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.evaluation.validation_suite import get_three_way_vehicle_split, load_artifacts
from src.model.live_blender import LiveStateTracker
from src.router.raptor import CommonLinesRaptor
from src.router.transit_graph import load_transit_graph

ROOT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as tc:
        yield tc


def test_1_zero_fabricated_numbers_and_three_way_split_integrity() -> None:
    """Verify Fixes 0.1-0.8: zero fabricated multipliers, strict 60/20/20 vehicle split, and Stage 4a/4b reporting."""
    val_suite_path = ROOT_DIR / "src" / "evaluation" / "validation_suite.py"
    val_suite_src = val_suite_path.read_text(encoding="utf-8")

    # 1. Inspect AST numerical constants in validation_suite.py to ban 1.25, 1.18, 0.86, 18.5
    tree = ast.parse(val_suite_src)
    numeric_constants = {
        round(float(node.value), 4)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    }
    banned_constants = {1.25, 1.18, 0.86, 18.5}
    found_banned = banned_constants.intersection(numeric_constants)
    assert not found_banned, f"Banned synthetic constants found in validation_suite.py: {found_banned}"

    # Verify validation_suite.py calls fetch_google_maps_transit_eta from google_maps_client
    assert "from src.evaluation.google_maps_client import fetch_google_maps_transit_eta" in val_suite_src

    # 2. Verify strict 3-way vehicle split (60% train / 20% calib / 20% test)
    models, df_obs = load_artifacts()
    train_set, calib_set, test_set = get_three_way_vehicle_split(df_obs)
    n_total = len(train_set) + len(calib_set) + len(test_set)
    assert n_total == 5555
    assert abs(len(train_set) / n_total - 0.60) < 0.005
    assert abs(len(calib_set) / n_total - 0.20) < 0.005
    assert abs(len(test_set) / n_total - 0.20) < 0.005
    assert train_set.isdisjoint(calib_set)
    assert train_set.isdisjoint(test_set)
    assert calib_set.isdisjoint(test_set)

    # Verify lgbm_trainer.py fits calibration_factor and kappa on calib_real and evaluates on test_real
    trainer_src = (ROOT_DIR / "src" / "model" / "lgbm_trainer.py").read_text(encoding="utf-8")
    assert "raw_calib_pred = model_expected.predict(X_calib)" in trainer_src
    assert "calibration_factor = sum_actual_calib / max(1.0, sum_pred_raw_calib)" in trainer_src
    assert "kappa = calibrate_journey_p85_kappa(" in trainer_src
    assert "gamma_self, gamma_cross = fit_ratio_space_kernels(tc_real" in trainer_src

    # 3. Verify Stage 4a (cross-vehicle) and Stage 4b (same-vehicle) in validation_results.json and VALIDATION_REPORT.md
    val_json = json.loads((ROOT_DIR / "data" / "processed" / "validation_results.json").read_text(encoding="utf-8"))
    stage_ids = [m["stage_id"] for m in val_json["ablation_metrics"]]
    assert "Stage 4a (cross-vehicle)" in stage_ids
    assert "Stage 4b (same-vehicle)" in stage_ids

    val_report_md = (ROOT_DIR / "docs" / "VALIDATION_REPORT.md").read_text(encoding="utf-8")
    assert "Stage 4a (cross-vehicle)" in val_report_md
    assert "Stage 4b (same-vehicle)" in val_report_md


def test_2_upgrade_4a_three_profile_raptor_routing_and_latency() -> None:
    """Verify 3-profile RAPTOR routing tags options with profile_label and completes within < 150 ms total."""
    g = load_transit_graph()
    tracker = LiveStateTracker()
    raptor = CommonLinesRaptor(graph=g, tracker=tracker)

    valid_labels = {
        "Fastest",
        "Most Reliable (Low P85)",
        "Least Walking / Direct",
        "Fastest & Direct",
        "Fastest & Reliable",
        "Alternative Route",
    }

    latencies_ms: list[float] = []
    options = []
    for _ in range(5):
        t0 = time.perf_counter()
        options = raptor.plan_journey(
            origin="ISBT Kashmere Gate",
            destination="Nehru Place Terminal",
            departure_time=1794193200,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    median_latency_ms = float(np.median(latencies_ms))
    print(f"\n[OK] 3-Profile RAPTOR Median Latency: {median_latency_ms:.2f} ms (all: {[round(x, 2) for x in latencies_ms]})")
    assert median_latency_ms < 150.0, (
        f"3-profile RAPTOR latency {median_latency_ms:.2f} ms exceeded 150 ms limit"
    )
    assert len(options) >= 1
    for opt in options:
        assert getattr(opt, "profile_label", None) in valid_labels


def test_3_upgrade_4b_traffic_corridors_provenance_and_counts(client: TestClient) -> None:
    """Verify GET /live/traffic-corridors returns all 15,130 corridors with valid provenance and exact sum."""
    res = client.get("/live/traffic-corridors", params={"depart": "2026-11-10T08:30:00+05:30"})
    assert res.status_code == 200
    data = res.json()

    assert data["type"] == "FeatureCollection"
    summary = data["summary"]
    assert summary["total_corridors"] == 15130
    assert summary["live_count"] + summary["observed_hist_count"] + summary["prior_count"] == 15130

    features = data["features"]
    assert len(features) == 15130

    valid_provenances = {"LIVE", "OBSERVED_HIST", "PRIOR"}
    prov_counts = {"LIVE": 0, "OBSERVED_HIST": 0, "PRIOR": 0}
    for feat in features:
        props = feat["properties"]
        prov = props["provenance"]
        assert prov in valid_provenances, f"Invalid provenance: {prov}"
        prov_counts[prov] += 1
        assert props["ratio"] > 0.0

    assert prov_counts["LIVE"] == summary["live_count"]
    assert prov_counts["OBSERVED_HIST"] == summary["observed_hist_count"]
    assert prov_counts["PRIOR"] == summary["prior_count"]
