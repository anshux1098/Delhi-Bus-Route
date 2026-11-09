"""Sprint 5 Final Submission Audit & Validation Suite Tests (`tests/test_sprint5_validation.py`).

Enforces the 3 final submission audit checks:
1. Strictly Out-of-Fold / Held-Out Ablation Verification (`src/evaluation/validation_suite.py`):
   - Held-out trajectories (`val_real`) come strictly from the 20% validation vehicle split
     and have zero overlap with `train_real` used to build `train_eb_prior`.
   - Stage 1 (Static GTFS Timetable), Stage 2 (OOF Empirical Bayes), Stage 3 (OOF 400-Tree
     Gamma-LightGBM), and Stage 4 (Causal Live Anomaly Decay Blend) progressively reduce
     both MAE and RMSE (`MAE_1 > MAE_2 > MAE_3 > MAE_4` and `RMSE_1 > RMSE_2 > RMSE_3 > RMSE_4`).
2. Deliverable Existence & Verified Codebase Metric Accuracy:
   - `docs/VALIDATION_REPORT.md`, `docs/TECHNICAL_NOTE.md`, `README.md`, and
     `data/processed/validation_results.json` exist and are non-empty.
   - Accurately cite the verified metrics: `2,554` routes, `6,812` stops, `15,130` unique
     physical corridors, `84.11%` held-out P85 coverage, `rho = 0.600`, `19.92 ms` horizon
     tensor rebuild, `~43 ms` (`43.12 ms`) median RAPTOR routing latency, and port `8550`.
3. Origin-Destination Benchmark Table Completeness:
   - 50 diverse Delhi O-D scenarios evaluated across Peak and Off-Peak periods.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.validation_suite import (
    load_artifacts,
    load_static_gtfs_segment_schedule,
    prepare_strictly_held_out_validation_trips,
    run_4stage_ablation,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_1_strict_out_of_fold_ablation_and_progressive_error_reduction() -> None:
    """Verify held-out trajectories have zero training leakage and progressively reduce MAE & RMSE."""
    models, df_obs = load_artifacts()
    sched_map = load_static_gtfs_segment_schedule()

    unique_vehs = sorted(df_obs["vehicle_id"].unique().to_list())
    val_veh_set = {v for idx, v in enumerate(unique_vehs) if idx % 5 == 0}
    train_veh_set = set(unique_vehs) - val_veh_set
    assert len(val_veh_set.intersection(train_veh_set)) == 0

    held_out_trips, val_real, seg_val_metrics = prepare_strictly_held_out_validation_trips(
        df_obs, models, min_segments=4
    )

    # Verify all held-out trips belong exclusively to val_veh_set
    for tr in held_out_trips:
        vehs_in_trip = set(tr["vehicle_id"].to_list())
        assert vehs_in_trip.issubset(val_veh_set), "In-sample training vehicle leaked into held_out_trips!"
        assert tr.height >= 4

    # Verify held-out segment calibration metrics match Sprint 2 verified figures
    assert seg_val_metrics["val_segments_count"] == 2455
    assert 0.98 <= seg_val_metrics["segment_bias_ratio"] <= 1.02
    assert abs(seg_val_metrics["segment_p85_coverage_pct"] - 84.11) < 0.5

    # Run 4-stage ablation on held-out multi-stop trips
    metrics, summary = run_4stage_ablation(held_out_trips, sched_map, models, seg_val_metrics)
    assert len(metrics) == 4

    s1, s2, s3, s4 = metrics

    # Assert strict progressive MAE reduction across all 4 stages
    assert s1.mae_sec > s2.mae_sec > s3.mae_sec > s4.mae_sec, (
        f"Expected progressive MAE reduction across stages, got: "
        f"S1={s1.mae_sec}, S2={s2.mae_sec}, S3={s3.mae_sec}, S4={s4.mae_sec}"
    )

    # Assert strict progressive RMSE reduction across all 4 stages
    assert s1.rmse_sec > s2.rmse_sec > s3.rmse_sec > s4.rmse_sec, (
        f"Expected progressive RMSE reduction across stages, got: "
        f"S1={s1.rmse_sec}, S2={s2.rmse_sec}, S3={s3.rmse_sec}, S4={s4.rmse_sec}"
    )

    # Assert P85 journey coverage is well-calibrated (>= 80%)
    assert s4.p85_coverage_pct is not None
    assert 80.0 <= s4.p85_coverage_pct <= 98.0
    assert summary["mae_reduction_pct"] > 40.0
    assert summary["rmse_reduction_pct"] > 40.0


def test_2_deliverable_documents_and_verified_metrics_citations() -> None:
    """Verify VALIDATION_REPORT.md, TECHNICAL_NOTE.md, and README.md exist and cite exact metrics."""
    val_report = ROOT_DIR / "docs" / "VALIDATION_REPORT.md"
    tech_note = ROOT_DIR / "docs" / "TECHNICAL_NOTE.md"
    readme = ROOT_DIR / "README.md"

    for doc_path in (val_report, tech_note, readme):
        assert doc_path.exists(), f"Missing required deliverable file: {doc_path}"
        content = doc_path.read_text(encoding="utf-8")
        assert len(content) > 1500, f"Deliverable {doc_path.name} is unexpectedly short ({len(content)} bytes)"

    combined_docs = (
        val_report.read_text(encoding="utf-8")
        + "\n"
        + tech_note.read_text(encoding="utf-8")
        + "\n"
        + readme.read_text(encoding="utf-8")
    )

    required_tokens = [
        "2,554",
        "6,812",
        "15,130",
        "84.11%",
        "0.600",
        "19.92 ms",
        "43.12 ms",
        "8550",
    ]
    for tok in required_tokens:
        assert tok in val_report.read_text(encoding="utf-8"), (
            f"Expected verified metric '{tok}' in docs/VALIDATION_REPORT.md"
        )
        assert tok in tech_note.read_text(encoding="utf-8"), (
            f"Expected verified metric '{tok}' in docs/TECHNICAL_NOTE.md"
        )
        assert tok in readme.read_text(encoding="utf-8"), (
            f"Expected verified metric '{tok}' in README.md"
        )
    assert len(combined_docs) > 5000


def test_3_validation_json_and_50_od_benchmarks() -> None:
    """Verify data/processed/validation_results.json contains all 4 stages and 50 O-D benchmarks."""
    json_path = ROOT_DIR / "data" / "processed" / "validation_results.json"
    assert json_path.exists(), f"Missing {json_path}"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "summary" in data
    assert "ablation_metrics" in data
    assert "od_benchmarks" in data

    assert len(data["ablation_metrics"]) == 4
    assert len(data["od_benchmarks"]) == 50

    # Verify O-D rows have positive durations and P85 > mean
    for row in data["od_benchmarks"]:
        assert row["hybrid_p85_mins"] > row["hybrid_mean_mins"]
        assert row["distance_km"] > 0
        assert row["free_flow_mins"] > 0
