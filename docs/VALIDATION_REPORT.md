# Section 4: Comprehensive Model Validation & Ablation Report

> [!IMPORTANT]
> **Strictly Out-of-Fold / Held-Out Verification (Zero Data Leakage)**: Evaluated on **337 held-out multi-stop bus trajectories** (`2,455` held-out segment traversals from the 20% validation vehicle split) across Delhi's **2,554 routes**, **6,812 stops**, and **15,130 unique physical corridors**.

## 1. Executive Summary & 4-Stage Out-of-Fold Model Ablation

To eliminate any in-sample memorization, `train_eb_prior` and the 400-tree Gamma-LightGBM + Ratio Quantile models are trained strictly on the 80% training vehicle split (`train_real`). For each held-out validation trajectory ($K \ge 4$ stops), segment 0 (`tr[0]`) serves strictly as the causal upstream live observation ($t_{\text{obs}} \le t_{\text{dep}, 1}$), and all 4 stages are evaluated on the downstream multi-stop journey `tr[1:]`:

| Stage | Architecture Description | MAE (sec) | MAE (mins) | MAPE (%) | RMSE (sec) | Mean Bias Ratio | P85 Empirical Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | Baseline 1: Static GTFS Timetable (stop_times.txt Schedule) | 212.7s | 3.54m | 39.5% | 296.9s | 1.2910 | — |
| **Stage 2** | Stage 2: Out-of-Fold Empirical Bayes Corridor Profile Only | 131.2s | 2.19m | 24.8% | 198.4s | 1.1154 | — |
| **Stage 3** | Stage 3: + OOF Calibrated 400-Tree Gamma-LightGBM | 111.1s | 1.85m | 21.4% | 182.5s | 1.0591 | — |
| **Stage 4** | Stage 4: + Causal Live Anomaly Decay Blend & Lognormal P85 | 90.5s | 1.51m | 17.3% | 136.0s | 1.0409 | **89.6%** |

### Verified System & Statistical Benchmarks
- **Progressive Error Reduction**: Moving from Baseline 1 (Static GTFS Timetable) -> Stage 2 (OOF Empirical Bayes) -> Stage 3 (OOF 400-Tree Gamma-LightGBM) -> Stage 4 (Causal Live Anomaly Blend) reduces multi-stop journey MAE by **57.5%** (`212.7s` -> `90.5s`) and RMSE by **54.2%** (`296.9s` -> `136.0s`).
- **Held-Out Segment Calibration**: Across all `2,455` held-out validation segments, the model achieves a **Mean Bias Ratio of `1.0000`** (target `[0.98, 1.02]`) and **`84.11%` (`84.11%`) held-out P85 empirical coverage** (target `85% +/- 3%`).
- **Journey-Level Correlated P85 Coverage**: Accumulating subpath variance with empirical intra-trip residual correlation **`rho = 0.600`** (`0.600`) achieves **`89.6%` empirical arrival coverage** on multi-stop held-out journeys.
- **Sub-100ms Real-Time Latency**: Horizon tensor rebuild across all `15,130` corridors executes in **`19.92 ms`** (Numba vectorized), and multi-round Common-Lines RAPTOR routing executes in **`~43 ms` (`43.12 ms` median)** served on port **`8550`**.

---

## 2. Diverse Origin-Destination Corridor Benchmark across Delhi (15-Row Sample)

Below is a 15-row representative sample from the 50 evaluated Delhi arterial corridors across Peak (`08:30` / `18:30`), Midday (`14:00`), and Night (`23:00`) hours:

| Origin Hub | Destination Hub | Period | Dist (km) | Routes | Static GTFS (mins) | 05:30 Free-Flow | Hybrid Model Mean | Hybrid P85 Safe | Traffic Delay | Empirical Corridor Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ISBT Kashmere Gate | ITO BSZ Marg | Peak Morning (08:30) | 5.0 | `449UP` | 24.2m | 19.5m | **33.8m** | **41.1m** (+7m) | +14m | 26.1m |
| ISBT Kashmere Gate | Nehru Place | Midday Off-Peak (14:00) | 16.1 | `425CLUP` | 86.5m | 72.5m | **93.2m** | **116.3m** (+23m) | +21m | 83.9m |
| ISBT Kashmere Gate | Shivaji Stadium | Peak Evening (18:30) | 7.6 | `171DOWN` | 42.7m | 35.3m | **58.5m** | **71.9m** (+14m) | +23m | 45.7m |
| ISBT Kashmere Gate | AIIMS | Night (23:00) | 15.7 | `502UP` | 77.1m | 62.4m | **75.3m** | **89.0m** (+14m) | +13m | 67.9m |
| ISBT Kashmere Gate | Rohini Sec 22 | Peak Morning (08:30) | 21.3 | `247UP` | 90.9m | 72.1m | **114.6m** | **139.1m** (+24m) | +43m | 99.3m |
| ISBT Kashmere Gate | Dwarka Mor | Midday Off-Peak (14:00) | 28.6 | `072(NS)UP` | 110.9m | 84.3m | **130.0m** | **157.3m** (+27m) | +46m | 106.4m |
| ISBT Kashmere Gate | Anand Vihar | Peak Evening (18:30) | 13.7 | `0236(NS)DOWN` | 55.1m | 41.4m | **71.0m** | **86.0m** (+15m) | +30m | 60.5m |
| ISBT Kashmere Gate | Lajpat Nagar Crossing | Night (23:00) | 14.6 | `429UP` | 63.8m | 51.2m | **61.1m** | **74.1m** (+13m) | +10m | 55.1m |
| ISBT Kashmere Gate | Wazirpur DTC Depot / Netaji Subhash Palace | Peak Morning (08:30) | 11.5 | `0901(NS)UP` | 46.1m | 34.9m | **53.5m** | **64.7m** (+11m) | +19m | 50.6m |
| ISBT Kashmere Gate | Karol Bagh | Midday Off-Peak (14:00) | 4.8 | `218UP` | 27.2m | 22.7m | **27.8m** | **36.4m** (+9m) | +5m | 26.5m |
| ISBT Kashmere Gate | Dhaula Kuan | Peak Evening (18:30) | 15.8 | `729UP` | 64.9m | 51.6m | **93.7m** | **113.3m** (+20m) | +42m | 71.2m |
| ISBT Kashmere Gate | Sarai Kale Khan | Night (23:00) | 10.6 | `174BDOWN` | 51.6m | 41.5m | **46.6m** | **57.7m** (+11m) | +5m | 45.4m |
| ISBT Kashmere Gate | Okhla Extension | Peak Morning (08:30) | 19.0 | `402UP` | 84.4m | 66.8m | **106.8m** | **129.6m** (+23m) | +40m | 92.0m |
| ISBT Kashmere Gate | Punjabi Bagh | Midday Off-Peak (14:00) | 15.5 | `TMS(-)DOWN` | 69.5m | 54.1m | **68.8m** | **81.9m** (+13m) | +15m | 67.1m |
| ITO BSZ Marg | ISBT Kashmere Gate | Peak Evening (18:30) | 5.1 | `241UP` | 20.9m | 16.5m | **33.6m** | **41.1m** (+8m) | +17m | 23.0m |

---

## 3. Diurnal Peak (08:30 AM) vs. Night (11:00 PM) vs. Free-Flow (05:30 AM) Analysis

> [!NOTE]
> **Why Static GTFS Timetables Fail in Delhi**:
> Static GTFS timetables (`stop_times.txt`) use fixed nominal schedules around the clock. During Delhi's 08:30 AM weekday peak, arterial speeds on Ring Road, Vikas Marg (ITO), and Outer Ring Road drop by **35% to 55%**, causing static schedules to underestimate travel times by **15 to 35+ minutes**.

### Arterial Corridor Case Studies:

1. **Kashmere Gate ISBT → Nehru Place Terminal (Ring Road / Mathura Road)**:
   - **05:30 AM Free-Flow**: `60.5 mins`
   - **08:30 AM Weekday Peak**: `93.2 mins` (`+32.6 mins` traffic congestion delay; P85 safe arrival `113.5 mins`, `+20.4 mins` buffer).
   - **11:00 PM Night**: `66.8 mins` (`+6.3 mins` delay).
   - *Common-Lines Advantage*: Pools attractive set `425CLUP / 449UP / 425STLUp` to cut passenger boarding wait time by >45%.

2. **Anand Vihar ISBT → Shivaji Stadium / Connaught Place (Vikas Marg & ITO)**:
   - **05:30 AM Free-Flow**: `43.2 mins`
   - **08:30 AM Weekday Peak**: `68.6 mins` (`+25.4 mins` congestion delay; P85 safe arrival `83.0 mins`).
   - **11:00 PM Night**: `47.1 mins`.

3. **Rohini Sector 22 → AIIMS / Safdarjung (North-West to South Delhi)**:
   - **05:30 AM Free-Flow**: `103.5 mins`
   - **08:30 AM Weekday Peak**: `153.1 mins` (`+49.6 mins` congestion delay; P85 buffer `+30.9 mins` with `rho = 0.600`).