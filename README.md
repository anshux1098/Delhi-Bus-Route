<div align="center">

# 🚍 Delhi Transit Engine

### **Time-Dependent Common-Lines RAPTOR & Hybrid Travel-Time Prediction System**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gamma%20%2B%20Quantile-9C27B0?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![Numba JIT](https://img.shields.io/badge/Numba-JIT%20Accelerated-00A3E0?style=for-the-badge&logo=numba&logoColor=white)](https://numba.pydata.org/)
[![Polars](https://img.shields.io/badge/Polars-Dataframes-CD792C?style=for-the-badge&logo=polars&logoColor=white)](https://pola.rs/)

<p align="center">
  <b>Production-grade real-time transit routing and arrival-time prediction across the entire Delhi National Capital Territory (`2,554` routes, `6,812` stops, `15,130` physical corridors).</b>
</p>

<p align="center">
  <a href="#-system-benchmarks-at-a-glance">Benchmarks</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-algorithmic--mathematical-pillars">Mathematical Engine</a> •
  <a href="#-ablation--diurnal-validation">Validation</a> •
  <a href="#-quickstart--installation">Quickstart</a> •
  <a href="#-rest-api-reference">API Reference</a>
</p>

</div>

---

## ⚡ System Benchmarks at a Glance

| Metric Category | Verified Benchmark | Engineering Mechanism |
| :--- | :---: | :--- |
| **Network Coverage** | **`2,554` Routes / `6,812` Stops** | 100% shape projection coverage (`86,326` stop projections) with automatic stop-to-stop polyline fallback |
| **Physical Corridor Pooling** | **`15,130` Unique Corridors** | Route-agnostic `corridor_segment_id` pooling (`5.7x` denser telemetry per road link) |
| **Horizon Tensor Rebuild** | **`2.25 ms` median** (`19.92 ms` baseline) | Vectorized 3D tensor slice (`15,130 × 12` 15-min horizon bins) with live anomaly decay (`< 50 ms` SLA) |
| **Routing Latency** | **`43.12 ms` (`~43 ms`) / `< 100 ms` 3-Profile** | Numba `@njit(fastmath=True)` 3-Profile Common-Lines RAPTOR scan + $O(1)$ route prefix arrays |
| **Untouched Test Bias Ratio** | **`1.0046`** *(calib `1.0000`, target `[0.98, 1.02]`)* | Strict `60% train / 20% calib / 20% test` vehicle split + 400-tree Gamma objective (`calibration_factor = 0.9977`) |
| **Untouched Test P85 Coverage** | **`84.50%`** *(segment, `84.11%` baseline)* / **`88.4%–89.1%`** *(journey)* | Out-of-fold Ratio Quantile regressor + lognormal correlated subpath aggregation (`rho_unclipped = 0.364`, `rho = 0.600`, `kappa = 0.950`) |
| **End-to-End Error Reduction** | **`-52.9%` MAE / `-52.1%` RMSE** | Strictly out-of-fold 5-stage ablation (`349.1s -> 164.5s` MAE, `529.6s -> 253.5s` RMSE) vs. raw `stop_times.txt` |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph S1["Sprint 1: Telemetry Hygiene & Planar Map-Matching"]
        A["GTFS Static Feeds\n(2,554 Routes, 6,812 Stops)"] --> B["Planar Shape Indexer\n(Equirectangular x_m, y_m)"]
        C["GTFS-RT Vehicle Pings\n(Protobuf Stream / Historical)"] --> D["Telemetry Hygiene Filter\n(Bounds, Dedup, Teleport <= 28.0 m/s)"]
        B & D --> E["Windowed Isotonic Map-Matcher\n(75m 3-Strike Gate, 100m Layover Strip, PAVA)"]
        E --> F["Observed Segments & Poll-to-Poll Spans\n(span_weight = 1 / n_seg_in_span)"]
    end

    subgraph S2["Sprint 2 & 6: 60/20/20 Hybrid Travel-Time & Ratio Kernels"]
        F --> G["Cross-Route Corridor Pooling\n(86,326 Pairs -> 15,130 Corridors)"]
        G --> H["60% Train LOO Empirical Bayes Prior\n+ 400-Tree Gamma & 300-Tree Ratio Quantile"]
        H --> I["20% Calib Post-Hoc Scalar (c = 0.9977)\n+ Journey P85 kappa = 0.950"]
        I --> J["Fitted Ratio Kernels (gamma_cross, gamma_self)\n+ 2.25 ms Horizon Tensor Builder"]
    end

    subgraph S3["Sprint 3, 4 & 6: 3-Profile RAPTOR & Provenance Heatmap UI"]
        J --> K["O(1) Route Prefix Arrays\n(Tcum, S1, S2, FFcum)"]
        K --> L["3-Profile Numba Common-Lines RAPTOR\n(Fastest / Most Reliable P85 / Least Walking)"]
        L --> M["FastAPI Server (Port 8550)\n+ Provenance-Aware Traffic Sensor Layer"]
    end
```

---

## 🔬 Algorithmic & Mathematical Pillars

1. **Planar Map-Matching, Layover Truncation & Span Extraction (`src/matching/`)**:
   - Projects all shapes and stops relative to Connaught Place (`lat0 = 28.6139`, `lon0 = 77.2090`) into planar meters (`x_m`, `y_m`).
   - Searches within a causal forward window $[d_{\text{prev}} - 50.0\text{m}, d_{\text{prev}} + 22.0\Delta t + 200.0\text{m}]$, penalizing opposing-direction bearings ($\vec{v}\cdot\hat{t} < 0$) and rejecting off-route pings ($d_\perp > 75.0\text{m}$ for `3` consecutive pings).
   - Strips terminal layover jitter within `100.0m` of first/last stops, applies **Isotonic Regression (PAVA)** on $d(t)$, and records `span_weight = 1.0 / n_seg_in_span` alongside raw poll-to-poll spans (`observed_spans.parquet`).
2. **Three-Way `60/20/20` Vehicle Split & Ratio-Space Live Kernels (`src/model/`)**:
   - Partitions `6,166` vehicles deterministically into `60% train` (`3,699` vehicles), `20% calib` (`1,233` vehicles), and `20% untouched test` (`1,234` vehicles).
   - Fits `gamma_cross = (0.1, 5.0m)` (cross-vehicle corridor signal) and `gamma_self = (0.3993, 20.0m)` (same-vehicle upstream signal) on `train + calib`.
3. **3-Profile Reliability Routing & Provenance Heatmap (`src/router/` & `src/api/`)**:
   - Scans 3 additive cost surrogates (`Fastest`, `Most Reliable (Low P85)`, `Least Walking / Direct`), rescores with correlated subpath variance, and deduplicates itineraries.
   - `GET /live/traffic-corridors` colors all `15,130` corridors by congestion ratio with strict provenance (`LIVE`, `OBSERVED_HIST`, `PRIOR`).

---

## 📊 Ablation & Diurnal Validation

### 1. Strictly Out-of-Fold 5-Stage Model Ablation (`3,423` Untouched Test Trips)

Evaluated strictly on the **20% untouched `test_vehicles` split** (`60,365` segments, `3,423` multi-stop trajectories with $K \ge 4$ stops). Segment `0` serves exclusively as the causal upstream ping ($t_{\text{obs}} \le t_{\text{dep}, 1}$), and all 5 stages are evaluated on the downstream trajectory `tr[1:]`:

| Stage | Architecture Description | MAE (sec) | MAE (mins) | MAPE (%) | RMSE (sec) | Mean Bias Ratio | P85 Empirical Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | Baseline 1: Raw Static GTFS Timetable (stop_times.txt Schedule) | 349.1s | 5.82m | 28.68% | 529.6s | 1.1605 | — |
| **Stage 2** | Stage 2: Out-of-Fold Empirical Bayes Corridor Profile Only | 180.9s | 3.02m | 16.69% | 270.6s | 1.0169 | — |
| **Stage 3** | Stage 3: + OOF Calibrated 400-Tree Gamma-LightGBM | 168.2s | 2.80m | 15.36% | 258.8s | 1.0119 | — |
| **Stage 4a (cross-vehicle)** | Stage 4a (cross-vehicle): + Cross-Vehicle Ratio Live Kernel (gamma_cross) | 167.2s | 2.79m | 15.27% | 257.4s | 1.0101 | **88.4%** |
| **Stage 4b (same-vehicle)** | Stage 4b (same-vehicle): + Same-Vehicle Causal Ratio Live Kernel (gamma_self) | 164.5s | 2.74m | 14.89% | 253.5s | 1.0109 | **89.1%** |

- **Progressive Error Reduction (`Stage 1 -> Stage 4b`)**: Reduces multi-stop journey MAE by **`52.9%`** (`349.1s` $\to$ `164.5s`) and RMSE by **`52.1%`** (`529.6s` $\to$ `253.5s`).
- **Span-Level Ground Truth Verification**: Across **`5,000` raw consecutive matched GPS ping-pairs** `(t_a, d_a) -> (t_b, d_b)` on untouched test vehicles, the model achieves **`23.7s` MAE**, **`65.9s` RMSE**, **`34.9%` MAPE**, and **`0.9858` Mean Bias Ratio**.

### 2. Sample Delhi Arterial Corridor Benchmarks (Computed via `CommonLinesRaptor` & `fetch_google_maps_transit_eta`)

| Origin Hub | Destination Hub | Period | Dist (km) | Routes | Static GTFS / Google | 05:30 Free-Flow | Hybrid Model Mean | Hybrid P85 Safe | Traffic Delay | Baseline Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ISBT Kashmere Gate | ITO BSZ Marg | Peak Morning (08:30) | 5.0 | `405EUP` | 17.9m | 17.4m | **30.7m** | **37.6m** (+6.9m) | +13.3m | `Static_GTFS_Schedule_Baseline` |
| ISBT Kashmere Gate | Nehru Place | Midday Off-Peak (14:00) | 16.1 | `425CLUP` | 55.9m | 65.7m | **79.2m** | **100.8m** (+21.5m) | +13.5m | `Static_GTFS_Schedule_Baseline` |
| ISBT Kashmere Gate | Shivaji Stadium | Peak Evening (18:30) | 7.6 | `246DOWN` | 33.0m | 32.3m | **56.4m** | **69.8m** (+13.4m) | +24.1m | `Static_GTFS_Schedule_Baseline` |
| ISBT Kashmere Gate | AIIMS | Night (23:00) | 15.7 | `502UP` | 75.2m | 59.5m | **66.0m** | **78.8m** (+12.8m) | +6.5m | `Static_GTFS_Schedule_Baseline` |
| ISBT Kashmere Gate | Rohini Sec 22 | Peak Morning (08:30) | 21.3 | `247UP` | 78.8m | 63.1m | **100.6m** | **122.8m** (+22.2m) | +37.5m | `Static_GTFS_Schedule_Baseline` |

> [!WARNING]
> **Telemetry Provenance & Diurnal Prior Disclosure**: Historical GPS observations in `data/historical/` were captured on Saturday (`day_of_week = 5`, hours `12:00–12:59` and `20:00–20:59` IST). Weekday peak (`08:30`) vs. night (`23:00`) contrasts reflect the explicit diurnal prior (`get_diurnal_multiplier` in `historical_baseline.py`) combined with learned spatial corridor baselines, pending weekday peak telemetry capture.

---

## 🚀 Quickstart & API Reference (`Port 8550`)

```powershell
.\.venv\Scripts\python.exe -m src.api.app
```

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Interactive MapLibre GL JS dark-mode transit dashboard with Traffic Sensor Layer |
| `/journeys` | `GET` | 3-profile reliability routing (`Fastest`, `Most Reliable (Low P85)`, `Least Walking / Direct`) |
| `/live/traffic-corridors` | `GET` | GeoJSON heatmap across all `15,130` corridors with strict `LIVE`, `OBSERVED_HIST`, `PRIOR` provenance |
| `/stops/search` | `GET` | Prefix + substring autocomplete across all `6,812` Delhi bus stops |
| `/live/buses` | `GET` | Active live GPS vehicles with route short names, bearings, speeds, and ping age |
| `/demo/presets` | `GET` | Curated arterial benchmark scenarios across Delhi |
| `/health` | `GET` | Operational health check (`6,812` stops, `2,554` routes, `15,130` corridors) |

Run the full verification suite:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
