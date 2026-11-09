# Delhi Transit Common-Lines RAPTOR & Hybrid Travel-Time Engine: Technical Note

> **Author**: Anshuman Singh  
> **Audience**: Technical Evaluation Committee & System Auditors  
> **Version**: 1.1.0 (Sprint 6 Production Release)  
> **Target Service Area**: Delhi National Capital Territory (`2,554` routes, `6,812` stops, `15,130` physical corridor segments)

---

## 1. Map-Matching & Trajectory Cleaning Pipeline (`src/ingestion/` & `src/matching/`)

Delhi's real-time transit telemetry is characterized by dense GPS noise, urban canyon reflections around high-density junctions (ITO, Kashmere Gate, AIIMS), multi-minute layovers at terminal depots, and irregular polling intervals (`15s` to `120s`). To extract ground-truth travel times from noisy pings, we implemented a deterministic hygiene and matching pipeline:

### 1.1 Telemetry Hygiene (`src/ingestion/hygiene.py`)
- Raw GTFS-Realtime pings are filtered to the Delhi NCR bounding box (`lat in [28.35, 28.90]`, `lon in [76.80, 77.45]`), deduplicated on `(vehicle_id, timestamp)`, and sorted chronologically.
- **Frozen Ping Collapse**: Stationary runs with step distance `< 1.0 m` (`FROZEN_THRESH_M = 1.0`) collapse intermediate pings, retaining only the initial arrival and final departure timestamps.
- **Teleport Filter**: Steps with implied speed `> 28.0 m/s` (`~100.8 km/h`, `MAX_SPEED_MPS = 28.0`) are rejected as GPS teleports.
- **Movement Bearing Vector**: Unit movement vectors $(\hat{v}_x, \hat{v}_y)$ are computed for consecutive steps with displacement `>= 20.0 m` (`BEARING_DIST_THRESH_M = 20.0`).

### 1.2 Planar Metric Projection & Shape Indexing (`src/matching/shape_indexer.py`)
- All GTFS geometries are projected onto an Equirectangular metric plane centered at Connaught Place (`lat0 = 28.6139`, `lon0 = 77.2090`):
  $$x = (\text{lon} - \text{lon}_0) \times 97,799.66 \text{ m/deg}, \quad y = (\text{lat} - \text{lat}_0) \times 111,132.92 \text{ m/deg}$$
- Shape polylines are parameterized by monotonic arc-length $s_k$ with precomputed unit tangent vectors $(\hat{t}_{x,k}, \hat{t}_{y,k})$.
- Routes missing a static `shape_id` in `trips.txt` automatically fall back to monotonic stop-to-stop polylines, ensuring **100% of the `2,554` Delhi GTFS routes** (`6,812` stops, `86,326` canonical stop projections) possess valid geometry.

### 1.3 Windowed Local Projection & 3-Strike Off-Route Gate (`src/matching/map_matcher.py`)
- Given a cleaned GPS ping $p = (x_p, y_p, \hat{v}_x, \hat{v}_y)$ along a route:
  - After initial global acquisition on $[0, L]$, subsequent pings search within a gap-scaled forward arc-length window:
    $$[d_{\text{prev}} - 50.0\text{ m}, \quad d_{\text{prev}} + 22.0 \cdot \Delta t + 200.0\text{ m}]$$
  - The ping is orthogonal-projected onto each candidate polyline segment:
    $$d_{\text{proj}} = s_k + \mathrm{clip}\left((p - v_k) \cdot \hat{t}_k, 0, \|\Delta v_k\|\right)$$
  - **Opposing-Bearing Penalty**: When a valid movement vector is present (`has_bearing = True`) and the dot product with the segment tangent is negative ($\vec{v} \cdot \hat{t}_k < 0$, i.e. heading $> 90^\circ$ against the route direction), an opposing-direction penalty of $100.0 \cdot (1 - \vec{v} \cdot \hat{t}_k)\text{ m}$ is added to the candidate score.
  - **3-Strike Off-Route Gate**: If perpendicular distance $d_{\perp} > 75.0\text{ m}$ for `3` consecutive pings, the ping is flagged `is_off_route = True` and global re-acquisition on $[0, L]$ is triggered on the next ping.

### 1.4 Terminal Layover Truncation, Isotonic Regression & Poll-to-Poll Spans
- **Depot Layover Stripping**: Pings within `100.0 m` of the origin terminal stop (`r_d <= d_first + 100.0`) prior to final departure, and pings after first entry within `100.0 m` of the destination terminal stop (`r_d >= d_last - 100.0`), are stripped.
- **Monotonicity Enforcement**: Runs covering `> 50.0 m` (`MIN_RUN_DISPLACEMENT_M = 50.0`) are smoothed via **Isotonic Regression** (Pool Adjacent Violators Algorithm) on $(t_i, d_i)$, guaranteeing $d(t_i) \le d(t_{i+1})$.
- **Span-Level Ground Truth & Overlap Weighting**: Because a single consecutive matched ping-pair $(t_a, d_a) \to (t_b, d_b)$ can span multiple stops, `extract_segments_and_spans` records `n_seg_in_span` and assigns each interpolated segment `span_weight = 1.0 / n_seg_in_span` so one real ping-pair counts once during training. Real poll-to-poll spans (`dt >= 15s`, `dd >= 80m`) and their exact corridor overlap fractions $|[d_a, d_b] \cap [d_u^e, d_v^e]| / L_e$ are saved to `data/processed/observed_spans.parquet`.

---

## 2. Sparse Corridors, 60/20/20 Vehicle Split & Hybrid Travel-Time Engine (`src/model/`)

### 2.1 Physical Corridor Segment Pooling (`segment_catalog.py`)
- Every directed stop-to-stop link is mapped to a route-agnostic physical token:
  $$\text{corridor\_segment\_id} = \text{stop\_u} \mathbin{\_} \text{stop\_v} \mathbin{\_} \mathrm{round}(L, 100\text{m})$$
- This pools `86,326` route-specific stop pairs onto **`15,130` unique physical corridor segments**, increasing observation density by **5.7x**.

### 2.2 Three-Way Deterministic Vehicle Split (`60% Train / 20% Calib / 20% Test`)
- Unique `vehicle_id`s (`5,555` total vehicles) are partitioned deterministically into three disjoint sets:
  - **`train_vehicles` (`60%`, `3,333` vehicles)**: Builds the Leave-One-Out (LOO) Empirical Bayes prior (`train_eb_prior`, shrinkage $k = 5.0$) and trains the 400-tree Gamma-LightGBM expected time regressor and 300-tree Ratio Quantile ($\alpha = 0.85$) model using `sample_weight = span_weight * 5.0`.
  - **`calib_vehicles` (`20%`, `1,111` vehicles)**: Fits the post-hoc mean calibration scalar (`calibration_factor = 1.0120`) so `calib` bias is `1.0000`, and calibrates the journey-level P85 scalar (`kappa = 0.930`) to hit `85.0%` multi-stop coverage.
  - **`test_vehicles` (`20%`, `1,111` vehicles, `24,053` segments, `1,488` multi-stop trips with $K \ge 4$)**: Strictly untouched held-out test split used exclusively for final evaluation (`held_out_segment_bias_ratio = 0.9979`, `held_out_segment_p85_coverage_pct = 83.44%`, baseline verification `84.11%`).

### 2.3 Fitted Ratio-Space Live Kernels (`gamma_self` & `gamma_cross`)
- On `train + calib`, we fit two mean-unbiased ratio-space exponential decay kernels $\gamma(\Delta t) = \gamma_0 \exp(-\Delta t / \tau)$:
  $$\hat{y}_m = \hat{\mu}_m \left[1 + w \cdot \gamma_0 \exp(-\Delta t / \tau) (q_0 - 1)\right], \qquad q_0 = \frac{y_0}{\hat{\mu}_0}$$
  - **Cross-Vehicle Corridor Kernel (`gamma_cross = (0.10, 5.0m)`)**: Captures corridor congestion reported by *other* vehicles in the last 45 minutes (`Stage 4a (cross-vehicle)`).
  - **Same-Vehicle Upstream Kernel (`gamma_self = (0.3892, 60.0m)`)**: Captures causal upstream pace anomalies of the *same* tracked bus (`Stage 4b (same-vehicle)`).
- **Precomputed Horizon Tensor (`live_blender.py`)**: The 96-bin 3D tensor (`15,130` corridors $\times$ `96` bins $\times$ `2` day types) is sliced across 12 horizon bins in **`2.82 ms` median** (`19.92 ms` baseline verification, target `< 50 ms`).

> [!WARNING]
> **Telemetry Provenance & Diurnal Prior Disclosure (Audit Fix 0.8)**: Historical GPS observations in `data/historical/` were captured on Saturday (`day_of_week = 5`, hours `12:00–12:59` and `20:00–20:59` IST). Spatial corridor base speeds are learned from observed telemetry, while weekday peak (`08:30`) vs. night (`23:00`) contrasts reflect the explicit diurnal prior (`get_diurnal_multiplier`), pending weekday peak telemetry capture.

---

## 3. 3-Profile Reliability Routing & Provenance Heatmap (`src/router/` & `src/api/`)

### 3.1 Chriqui-Robillard Common-Lines & Correlated Variance
- **Attractive Route Sets $A(u, v)$**: Pools overlapping routes sharing a corridor using Welding/Osuna headway regularity $E[W_r] = \frac{1}{2}E[H_r](1 + CV_r^2)$ and live upstream bus ETAs.
- **$O(1)$ Correlated Subpath Variance**: Precomputed route prefix arrays ($T_{\text{cum}}, S_1, S_2$) evaluate subpath variance in $O(1)$ time (`rho_unclipped = 0.376` on test trips, router parameter `rho = 0.600`):
  $$V(i \to j) = (1 - \rho)(S_2[j] - S_2[i]) + \rho(S_1[j] - S_1[i])^2, \qquad q_{85, J} = \mu_J \exp\left(-0.5 s_J^2 + \kappa s_J\right)$$

### 3.2 3-Profile Numba RAPTOR Scan (`raptor.py`)
- `CommonLinesRaptor.plan_journey` executes three additive cost surrogate scans sharing Round-0 initialization (`~43 ms` / `43.12 ms` single-profile median, `< 100 ms` 3-profile median on port **`8550`**):
  1. **`Fastest`**: Minimizes expected arrival time ($T_{\text{cum}} + E[W]$).
  2. **`Most Reliable (Low P85)`**: Penalizes high-variance corridors ($T_{\text{cum}} + \kappa S_1$ and $E[W] + \kappa\sqrt{\text{Var}(W)}$).
  3. **`Least Walking / Direct`**: Scales footpath walking by `1.5x` and adds `+600.0s` per transfer.
- Candidate journeys are rescored with exact correlated variance, deduplicated by `(route_sequence, board_stops, alight_stops)`, and tagged with `profile_label`.

### 3.3 Provenance-Aware Corridor Traffic Heatmap (`GET /live/traffic-corridors`)
- Serves all `15,130` simplified road-snapped corridor LineStrings (`data/processed/corridors_simplified.json`) with congestion ratio $\hat{t}(t)/\mu_{\text{free\_flow}}$ and strict data provenance (`LIVE`, `OBSERVED_HIST`, `PRIOR`) plus live summary counts.

---

## 4. Verified 5-Stage Out-of-Fold Ablation Summary (`1,488` Untouched Test Trips)

| Stage | Architecture Description | MAE (sec) | MAE (mins) | MAPE (%) | RMSE (sec) | Mean Bias Ratio | P85 Empirical Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | Baseline 1: Raw Static GTFS Timetable (stop_times.txt Schedule) | 292.4s | 4.87m | 25.38% | 427.0s | 1.1162 | — |
| **Stage 2** | Stage 2: Out-of-Fold Empirical Bayes Corridor Profile Only | 180.4s | 3.01m | 16.53% | 257.7s | 1.0262 | — |
| **Stage 3** | Stage 3: + OOF Calibrated 400-Tree Gamma-LightGBM | 159.7s | 2.66m | 14.57% | 240.1s | 1.0116 | — |
| **Stage 4a (cross-vehicle)** | Stage 4a (cross-vehicle): + Cross-Vehicle Ratio Live Kernel (gamma_cross) | 158.6s | 2.64m | 14.46% | 238.8s | 1.0092 | **85.8%** |
| **Stage 4b (same-vehicle)** | Stage 4b (same-vehicle): + Same-Vehicle Causal Ratio Live Kernel (gamma_self) | 156.9s | 2.61m | 14.15% | 236.3s | 1.0097 | **86.1%** |

- **Total Error Reduction (`Stage 1 -> Stage 4b`)**: **`46.3%` MAE reduction** (`292.4s` $\to$ `156.9s`) and **`44.7%` RMSE reduction** (`427.0s` $\to$ `236.3s`).
- **Span-Level Ground Truth (`5,000` raw test ping-pairs)**: **`24.3s` MAE**, **`76.0s` RMSE**, **`34.4%` MAPE**, and **`0.9798` Mean Bias Ratio**.
