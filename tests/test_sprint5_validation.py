"""Sprint 5 & 6 Final Submission Audit & Self-Verifying Documentation Suite (`tests/test_sprint5_validation.py`).

Enforces:
1. Strictly Out-of-Fold Three-Way Vehicle Split (`60% train / 20% calib / 20% test`) & 5-Stage Ablation:
   - Untouched test trajectories (`test_vehicles`, 20%) have zero overlap with `train_vehicles` (60%)
     and `calib_vehicles` (20%).
   - Stage 1 (Raw Static GTFS Timetable), Stage 2 (OOF Empirical Bayes), Stage 3 (OOF 400-Tree
     Gamma-LightGBM), Stage 4a (cross-vehicle ratio kernel `gamma_cross`), and Stage 4b (same-vehicle
     causal ratio kernel `gamma_self`) progressively reduce both MAE and RMSE:
     `MAE_1 > MAE_2 > MAE_3 > MAE_4a > MAE_4b` and `RMSE_1 > RMSE_2 > RMSE_3 > RMSE_4a > RMSE_4b`.
2. Self-Verifying Documentation Audit (Audit Fix 0.9):
   - Parses the ablation tables, summary metrics, span-level metrics, and O-D benchmark rows printed
     inside `docs/VALIDATION_REPORT.md`, `docs/TECHNICAL_NOTE.md`, and `README.md` and asserts that
     every parsed number matches `data/processed/validation_results.json` within rounding tolerance (`1e-2`).
3. Origin-Destination Benchmark Table Completeness:
   - 50 diverse Delhi O-D scenarios evaluated across Peak and Off-Peak periods using
     `fetch_google_maps_transit_eta` (`Static_GTFS_Schedule_Baseline` CSA or `Google_Maps_Live_API`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.evaluation.validation_suite import (
    get_three_way_vehicle_split,
    load_artifacts,
    load_static_gtfs_segment_schedule,
    prepare_strictly_held_out_validation_trips,
    run_4stage_ablation,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def _parse_ablation_table_from_markdown(md_text: str) -> dict[str, dict[str, float | None]]:
    """Parse markdown 5-stage ablation table rows into numeric dictionaries keyed by stage_id."""
    parsed: dict[str, dict[str, float | None]] = {}
    for line in md_text.splitlines():
        line_s = line.strip()
        if not line_s.startswith("| **Stage"):
            continue
        cells = [c.strip() for c in line_s.strip("|").split("|")]
        if len(cells) < 8:
            continue
        stage_id = cells[0].replace("*", "").strip()
        mae_sec = float(cells[2].rstrip("s"))
        mae_mins = float(cells[3].rstrip("m"))
        mape_pct = float(cells[4].rstrip("%"))
        rmse_sec = float(cells[5].rstrip("s"))
        bias_ratio = float(cells[6])
        cov_raw = cells[7].replace("*", "").strip()
        p85_cov = None if cov_raw in ("—", "-", "") else float(cov_raw.rstrip("%"))
        parsed[stage_id] = {
            "mae_sec": mae_sec,
            "mae_mins": mae_mins,
            "mape_pct": mape_pct,
            "rmse_sec": rmse_sec,
            "bias_ratio": bias_ratio,
            "p85_coverage_pct": p85_cov,
        }
    return parsed


def _parse_od_table_from_markdown(md_text: str) -> list[dict[str, Any]]:
    """Parse O-D benchmark markdown table rows into numeric dicts."""
    rows: list[dict[str, Any]] = []
    in_od_table = False
    for line in md_text.splitlines():
        line_s = line.strip()
        if line_s.startswith("| Origin Hub | Destination Hub |"):
            in_od_table = True
            continue
        if in_od_table:
            if not line_s.startswith("|"):
                break
            if ":---" in line_s:
                continue
            cells = [c.strip() for c in line_s.strip("|").split("|")]
            if len(cells) < 11:
                continue
            p85_cell = cells[8].replace("*", "").strip()
            p85_main = float(p85_cell.split()[0].rstrip("m"))
            rows.append(
                {
                    "origin": cells[0],
                    "destination": cells[1],
                    "period": cells[2],
                    "distance_km": float(cells[3]),
                    "routes": cells[4].strip("`"),
                    "static_gtfs_mins": float(cells[5].rstrip("m")),
                    "free_flow_mins": float(cells[6].rstrip("m")),
                    "hybrid_mean_mins": float(cells[7].replace("*", "").rstrip("m")),
                    "hybrid_p85_mins": p85_main,
                    "traffic_delay_mins": float(cells[9].lstrip("+").rstrip("m")),
                    "baseline_source": cells[10].strip("`"),
                }
            )
    return rows


def test_1_strict_out_of_fold_ablation_and_progressive_error_reduction() -> None:
    """Verify 60/20/20 vehicle split has zero leakage and all 5 stages progressively reduce MAE & RMSE."""
    models, df_obs = load_artifacts()
    sched_map = load_static_gtfs_segment_schedule()

    train_veh_set, calib_veh_set, test_veh_set = get_three_way_vehicle_split(df_obs)
    assert len(train_veh_set.intersection(calib_veh_set)) == 0
    assert len(train_veh_set.intersection(test_veh_set)) == 0
    assert len(calib_veh_set.intersection(test_veh_set)) == 0

    held_out_trips, val_real, seg_val_metrics = prepare_strictly_held_out_validation_trips(
        df_obs, models, min_segments=4
    )

    # Verify all held-out trips belong exclusively to untouched 20% test_veh_set
    for tr in held_out_trips:
        vehs_in_trip = set(tr["vehicle_id"].to_list())
        assert vehs_in_trip.issubset(test_veh_set), "Non-test vehicle leaked into held_out_trips!"
        assert tr.height >= 4

    # Verify untouched test segment calibration metrics
    assert seg_val_metrics["val_segments_count"] >= 24000
    assert 0.98 <= seg_val_metrics["segment_bias_ratio"] <= 1.02
    assert 80.0 <= seg_val_metrics["segment_p85_coverage_pct"] <= 90.0

    # Run 5-stage ablation on untouched test multi-stop trips
    metrics, summary = run_4stage_ablation(held_out_trips, sched_map, models, seg_val_metrics)
    assert len(metrics) == 5

    s1, s2, s3, s4a, s4b = metrics
    assert s4a.stage_id == "Stage 4a (cross-vehicle)"
    assert s4b.stage_id == "Stage 4b (same-vehicle)"

    # Assert strict progressive MAE reduction across all 5 stages
    assert s1.mae_sec > s2.mae_sec > s3.mae_sec > s4a.mae_sec > s4b.mae_sec, (
        f"Expected progressive MAE reduction across all 5 stages, got: "
        f"S1={s1.mae_sec}, S2={s2.mae_sec}, S3={s3.mae_sec}, S4a={s4a.mae_sec}, S4b={s4b.mae_sec}"
    )

    # Assert strict progressive RMSE reduction across all 5 stages
    assert s1.rmse_sec > s2.rmse_sec > s3.rmse_sec > s4a.rmse_sec > s4b.rmse_sec, (
        f"Expected progressive RMSE reduction across all 5 stages, got: "
        f"S1={s1.rmse_sec}, S2={s2.rmse_sec}, S3={s3.rmse_sec}, S4a={s4a.rmse_sec}, S4b={s4b.rmse_sec}"
    )

    # Assert P85 journey coverage is well-calibrated around 85%
    assert s4a.p85_coverage_pct is not None and 82.0 <= s4a.p85_coverage_pct <= 90.0
    assert s4b.p85_coverage_pct is not None and 82.0 <= s4b.p85_coverage_pct <= 90.0
    assert summary["mae_reduction_pct"] > 40.0
    assert summary["rmse_reduction_pct"] > 40.0


def test_2_deliverable_documents_and_verified_metrics_citations() -> None:
    """Self-verifying documentation test (Fix 0.9): parse numbers from docs and compare to validation_results.json."""
    val_report_path = ROOT_DIR / "docs" / "VALIDATION_REPORT.md"
    tech_note_path = ROOT_DIR / "docs" / "TECHNICAL_NOTE.md"
    readme_path = ROOT_DIR / "README.md"
    json_path = ROOT_DIR / "data" / "processed" / "validation_results.json"

    for doc_path in (val_report_path, tech_note_path, readme_path, json_path):
        assert doc_path.exists(), f"Missing required deliverable file: {doc_path}"

    val_data = json.loads(json_path.read_text(encoding="utf-8"))
    expected_stages = {m["stage_id"]: m for m in val_data["ablation_metrics"]}
    assert len(expected_stages) == 5

    # 1. Parse and verify 5-Stage Ablation tables across all 3 markdown documents
    for doc_path in (val_report_path, tech_note_path, readme_path):
        text = doc_path.read_text(encoding="utf-8")
        parsed_table = _parse_ablation_table_from_markdown(text)
        assert len(parsed_table) == 5, (
            f"Expected 5 ablation stages in {doc_path.name}, found {list(parsed_table.keys())}"
        )
        for stage_id, exp_m in expected_stages.items():
            assert stage_id in parsed_table, f"Missing {stage_id} in {doc_path.name}"
            got_m = parsed_table[stage_id]
            for key in ("mae_sec", "mae_mins", "mape_pct", "rmse_sec", "bias_ratio"):
                assert abs(float(got_m[key]) - float(exp_m[key])) <= 1e-2, (
                    f"Mismatch in {doc_path.name} [{stage_id}][{key}]: "
                    f"doc={got_m[key]} vs json={exp_m[key]}"
                )
            if exp_m["p85_coverage_pct"] is None:
                assert got_m["p85_coverage_pct"] is None
            else:
                assert got_m["p85_coverage_pct"] is not None
                assert abs(float(got_m["p85_coverage_pct"]) - float(exp_m["p85_coverage_pct"])) <= 1e-2

    # 2. Parse and verify summary & span metrics in VALIDATION_REPORT.md against JSON
    val_report_text = val_report_path.read_text(encoding="utf-8")
    summary = val_data["summary"]
    span = val_data["span_metrics"]

    m_mae_red = re.search(r"reduces multi-stop journey MAE by \*\*([\d.]+)%\*\*", val_report_text)
    m_rmse_red = re.search(r"and RMSE by \*\*([\d.]+)%\*\*", val_report_text)
    assert m_mae_red and abs(float(m_mae_red.group(1)) - float(summary["mae_reduction_pct"])) <= 1e-2
    assert m_rmse_red and abs(float(m_rmse_red.group(1)) - float(summary["rmse_reduction_pct"])) <= 1e-2

    m_seg_bias = re.search(r"Mean Bias Ratio of `([\d.]+)`", val_report_text)
    m_seg_cov = re.search(r"and \*\*`([\d.]+)%` held-out P85 empirical coverage\*\*", val_report_text)
    assert m_seg_bias and abs(float(m_seg_bias.group(1)) - float(summary["held_out_segment_bias_ratio"])) <= 1e-2
    assert m_seg_cov and abs(float(m_seg_cov.group(1)) - float(summary["held_out_segment_p85_coverage_pct"])) <= 1e-2

    m_span_mae = re.search(r"Span-Level MAE\*\*: `([\d.]+)s`", val_report_text)
    m_span_rmse = re.search(r"Span-Level RMSE\*\*: `([\d.]+)s`", val_report_text)
    m_span_bias = re.search(r"Span-Level Mean Bias Ratio\*\*: `([\d.]+)`", val_report_text)
    assert m_span_mae and abs(float(m_span_mae.group(1)) - float(span["span_mae_sec"])) <= 1e-2
    assert m_span_rmse and abs(float(m_span_rmse.group(1)) - float(span["span_rmse_sec"])) <= 1e-2
    assert m_span_bias and abs(float(m_span_bias.group(1)) - float(span["span_bias_ratio"])) <= 1e-2

    # 3. Parse O-D tables from VALIDATION_REPORT.md and README.md and verify every row matches JSON
    json_od = val_data["od_benchmarks"]
    rep_od = _parse_od_table_from_markdown(val_report_text)
    readme_od = _parse_od_table_from_markdown(readme_path.read_text(encoding="utf-8"))
    assert len(rep_od) == 15
    assert len(readme_od) == 5

    for idx, doc_row in enumerate(rep_od):
        exp_row = json_od[idx]
        assert doc_row["origin"] == exp_row["origin"]
        assert doc_row["destination"] == exp_row["destination"]
        assert doc_row["routes"] == exp_row["routes"]
        for num_k in ("distance_km", "static_gtfs_mins", "free_flow_mins", "hybrid_mean_mins", "hybrid_p85_mins", "traffic_delay_mins"):
            assert abs(float(doc_row[num_k]) - float(exp_row[num_k])) <= 1e-2, (
                f"O-D row {idx} mismatch on {num_k}: doc={doc_row[num_k]} vs json={exp_row[num_k]}"
            )


def test_3_validation_json_and_50_od_benchmarks() -> None:
    """Verify data/processed/validation_results.json contains all 5 stages and 50 O-D benchmarks."""
    json_path = ROOT_DIR / "data" / "processed" / "validation_results.json"
    assert json_path.exists(), f"Missing {json_path}"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "summary" in data
    assert "ablation_metrics" in data
    assert "span_metrics" in data
    assert "od_benchmarks" in data

    assert len(data["ablation_metrics"]) == 5
    stage_ids = [m["stage_id"] for m in data["ablation_metrics"]]
    assert stage_ids == [
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Stage 4a (cross-vehicle)",
        "Stage 4b (same-vehicle)",
    ]
    assert len(data["od_benchmarks"]) == 50

    for row in data["od_benchmarks"]:
        assert row["hybrid_p85_mins"] > row["hybrid_mean_mins"]
        assert row["distance_km"] > 0
        assert row["free_flow_mins"] > 0
        assert row["static_gtfs_mins"] > 0
        assert row["baseline_source"] in {"Static_GTFS_Schedule_Baseline", "Google_Maps_Live_API"}
        assert "observed_benchmark_mins" not in row
