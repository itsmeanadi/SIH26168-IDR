# Phase 26 — Authentic GNSS Blackout Benchmark Report

**Project:** SIH 2026 Problem Statement 26168 — AI-ML Intelligent Dead Reckoning (IDR)  
**Evaluation Target:** 15-State 3D Error-State Kalman Filter (`ErrorStateKalmanFilter`) vs. Baselines  
**Dataset:** Authentic IO-VNBD Driving Dataset (SHA-256 Provenance Verified)  
**Execution Date:** 2026-09-08  
**Evaluation Status:** COMPLETE & REPRODUCIBLE (100% Deterministic)  
**Official Scientific Verdict:** **`SIH TARGET NOT ACHIEVED`** (0% Pass Rate across 180 Outage Scenarios)

---

## 1. Executive Summary & Final Scientific Verdict

In accordance with strict scientific integrity protocols, Phase 26 conducted an autonomous, exhaustive, and leak-free empirical evaluation of the newly integrated **15-State 3D Error-State Kalman Filter (ES-EKF)** during simulated GNSS outages on authentic smartphone and vehicle CAN-bus recordings from the IO-VNBD dataset.

### Core Verdict
> [!WARNING]
> **FINAL VERDICT: SIH TARGET NOT ACHIEVED**
>
> Across **180 evaluation runs** spanning 3 outage durations (10s, 30s, 60s) and 5 vehicle motion regimes (Straight Cruising, Acceleration/Deceleration, Sustained Turning, Low-Speed/Stop-and-Go, High-Speed Motorway) on both Validation (Driver B / M) and Held-Out Test (Driver D / Y1) drives:
> - **Pass Rate for SIH <10% Drift:** **0.0% (0 / 180 runs passed)**
> - **15-State Full ES-EKF Median Drift Percentage:** **1897.65%**
> - **15-State Full ES-EKF 90th Percentile Drift:** **4902.81%**
> - **Legacy 9-State Planar EKF:** **Exploded numerically** ($>10^{13}\text{ m}$ in severe turns due to 2D planar singularity under 3D smartphone tilt).
> - **Constant-Velocity Extrapolation Baseline:** **Median Drift: 971.68%** (outperformed inertial integration on straight road by avoiding MEMS accelerometer noise, but failed completely in turns).

This is an honest, scientifically valid baseline result. It establishes the true empirical dead reckoning capability of uncalibrated consumer smartphone MEMS sensors before machine-learned orientation leveling and adaptive bias compensation are engaged.

---

## 2. Scientific Leakage Controls & Pre-Benchmark Audit

Prior to benchmark execution, all evaluation scripts and data loaders were audited against strict leakage rules:

| Leakage Category | Audit Status | Verification Mechanism |
| :--- | :--- | :--- |
| **GT $\to$ Estimator Leakage** | **STRICTLY ZERO** | GNSS measurements are strictly blocked (`nav_gnss is None`) during the blackout window. Reference coordinates are only used offline for metric computation. |
| **GT $\to$ Map Leakage** | **STRICTLY ZERO** | No ground-truth waypoints were used to generate map geometries or snap trajectories. |
| **EKF $\to$ AI Feedback** | **STRICTLY ZERO** | `VelocityEstimatorNet` receives exclusively raw causal IMU window arrays $(1, 6, 50)$; zero filter states, covariances, or positions enter the neural network. |
| **Future Data Access** | **STRICTLY ZERO** | The AI inference window indexes strictly in $[t - 49 : t]$ (causal sliding window). |
| **Privileged GT Velocity Baseline** | **DISALLOWED / AUDITED** | The oracle baseline (holding previous GT velocity constant) was removed and replaced with a legitimate filter-velocity kinematic baseline (`const_vel_baseline`). |
| **Held-Out Test Tuning** | **STRICTLY ZERO** | Driver D (Drive Y1) was strictly held out; zero filter parameters, thresholds, or scalers were tuned on Drive Y1. |
| **Repeatability / Determinism** | **VERIFIED ($\Delta = 0.000\text{ m}$)** | An identical 10s turning case was run twice; the difference across all states was exactly $0.000000000000\text{ m}$ (bitwise identical). |

---

## 3. Benchmark Design & Data Splits

### 3.1 Data Splits
- **Training Set (Frozen):** Drivers E + A (`Vf`, `Vta`, `Vtb`, `Vw`, `S`) — Used for training `VelocityEstimatorNet`.
- **Validation Set:** Driver B / Drive M (`data/raw/categorised_authentic/M (Driver B)`) — 105,942 synchronized samples (~2.94 hours).
- **Held-Out Test Set:** Driver D / Drive Y1 (`data/raw/categorised_authentic/Y (Driver D)/Y1`) — 70,285 synchronized samples (~1.95 hours).

### 3.2 Evaluated Baselines
1. **Baseline A (`15state_imu_only`):** 15-state ES-EKF pure inertial propagation with IMU only during outage (no AI speed updates, no NHC, no ZUPT).
2. **Baseline B (`15state_ai`):** 15-state ES-EKF + causal neural network forward speed updates (`update_ai_velocity`).
3. **Baseline C (`15state_ai_nhc`):** 15-state ES-EKF + AI speed updates + Non-Holonomic Constraints (`update_nhc`, $\sigma_{lat}=0.05, \sigma_{vert}=0.05$).
4. **Baseline D (`15state_full`):** 15-state ES-EKF + AI speed + NHC + StationaryDetector ZUPT/ZARU (`update_zupt`, `update_zaru`).
5. **Baseline E (`legacy_ekf_full`):** Legacy 9-state planar EKF with AI + NHC + ZUPT.
6. **Baseline F (`const_vel_baseline`):** Pure kinematic constant-velocity extrapolation $p(t) = p(t_0) + v_{est}(t_0) \cdot (t - t_0)$ from pre-blackout estimated state.

### 3.3 Motion Regimes Tested
For both Validation (Drive M) and Test (Drive Y1) across 10s, 30s, and 60s outages:
- **Regime 1: Straight Cruising** (Low yaw rate $<2.0^\circ/\text{s}$, steady moderate/high speed).
- **Regime 2: Acceleration / Deceleration** (Speed variation $>6.0\text{ m/s}$ during outage).
- **Regime 3: Turning** (Sustained cornering with mean yaw rate $>5.0^\circ/\text{s}$).
- **Regime 4: Low-Speed / Stop-and-Go** (Speeds $<4.0\text{ m/s}$, urban crawl).
- **Regime 5: High-Speed Cruising** (Motorway speeds $>15.0\text{ m/s}$ / $>54\text{ km/h}$).

---

## 4. Comprehensive Benchmark Results

### 4.1 Global Performance by Baseline (All 180 Runs)

| Baseline Name | Total Runs | SIH Pass Count (<10%) | Pass Rate (%) | Mean Drift (%) | Median Drift (%) | P90 Drift (%) | P95 Drift (%) | Worst Drift (%) | Mean Pos RMSE (m) | Mean Speed RMSE (m/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15-State Full ES-EKF** | 30 | 0 | **0.0%** | 2577.53% | **1897.65%** | 4902.81% | 5807.47% | 13133.77% | 4159.12 m | 160.54 m/s |
| **15-State ES-EKF + AI + NHC** | 30 | 0 | **0.0%** | 2594.70% | **1986.82%** | 4902.81% | 5789.86% | 13133.77% | 4163.43 m | 160.57 m/s |
| **15-State ES-EKF + AI** | 30 | 0 | **0.0%** | 3280.12% | **2273.03%** | 5783.48% | 9834.61% | 19266.21% | 4332.06 m | 163.38 m/s |
| **15-State Pure IMU DR** | 30 | 0 | **0.0%** | 3003.62% | **2293.03%** | 6655.00% | 8296.69% | 11596.32% | 4251.69 m | 169.04 m/s |
| **Kinematic Const-Vel Extrap.** | 30 | 0 | **0.0%** | 1726.05% | **971.68%** | 3496.22% | 5460.13% | 9089.91% | 2487.55 m | 75.28 m/s |
| **Legacy 9-State EKF** | 30 | 0 | **0.0%** | $>10^{15}\%$ | **13285.69%** | $>10^{17}\%$ | $>10^{25}\%$ | $>10^{57}\%$ | $>10^{55}\text{ m}$ | $>10^{56}\text{ m/s}$ |

---

### 4.2 Held-Out Test Drive (Driver D / Y1) Results by Outage Duration

| Duration | Baseline | Test Cases | Pass (<10%) | Pass Rate | Median Drift (%) | P90 Drift (%) | Median Final Error (m) | Mean Pos RMSE (m) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10s Outage** | **15-State Full ES-EKF** | 5 | 0 | **0.0%** | **1450.34%** | 3551.26% | 1734.78 m | 1548.95 m |
| | 15-State ES-EKF + AI + NHC | 5 | 0 | **0.0%** | **1446.01%** | 3555.70% | 1737.95 m | 1542.76 m |
| | 15-State ES-EKF + AI | 5 | 0 | **0.0%** | **1462.39%** | 3555.70% | 1737.95 m | 1536.70 m |
| | 15-State Pure IMU DR | 5 | 0 | **0.0%** | **2292.65%** | 3555.70% | 1737.95 m | 1450.17 m |
| | Const-Vel Extrapolation | 5 | 0 | **0.0%** | **1657.04%** | 3654.86% | 1883.92 m | 1410.55 m |
| | Legacy 9-State EKF | 5 | 0 | **0.0%** | **2406.17%** | $3.1 \times 10^6\%$ | 3281.72 m | $3.9 \times 10^5\text{ m}$ |
| **30s Outage** | **15-State Full ES-EKF** | 5 | 0 | **0.0%** | **1742.05%** | 2295.10% | 4876.11 m | 3159.14 m |
| | 15-State ES-EKF + AI + NHC | 5 | 0 | **0.0%** | **1933.57%** | 2295.10% | 4876.11 m | 3186.43 m |
| | 15-State ES-EKF + AI | 5 | 0 | **0.0%** | **2465.12%** | 3516.58% | 6251.94 m | 3448.74 m |
| | 15-State Pure IMU DR | 5 | 0 | **0.0%** | **2512.31%** | 3866.16% | 2320.09 m | 2927.25 m |
| | Const-Vel Extrapolation | 5 | 0 | **0.0%** | **925.20%** | 6064.39% | 3087.58 m | 2416.63 m |
| | Legacy 9-State EKF | 5 | 0 | **0.0%** | **3349.68%** | $2.1 \times 10^{18}\%$ | 7552.22 m | $4.8 \times 10^{17}\text{ m}$ |
| **60s Outage** | **15-State Full ES-EKF** | 5 | 0 | **0.0%** | **2559.57%** | 4727.20% | 15628.79 m | 7190.35 m |
| | 15-State ES-EKF + AI + NHC | 5 | 0 | **0.0%** | **2559.57%** | 4697.22% | 15628.79 m | 7193.34 m |
| | 15-State ES-EKF + AI | 5 | 0 | **0.0%** | **2294.14%** | 4532.58% | 14008.11 m | 7311.48 m |
| | 15-State Pure IMU DR | 5 | 0 | **0.0%** | **2627.63%** | 5259.30% | 16044.37 m | 7418.23 m |
| | Const-Vel Extrapolation | 5 | 0 | **0.0%** | **530.80%** | 1052.76% | 4193.75 m | 2997.17 m |
| | Legacy 9-State EKF | 5 | 0 | **0.0%** | **21002.58%** | $2.5 \times 10^{57}\%$ | 91909.92 m | $6.0 \times 10^{56}\text{ m}$ |

---

## 5. Deep Scientific Root-Cause Analysis

Detailed analysis of the exported 10 Hz CSV traces reveals 4 fundamental failure modes dominating the drift:

### Root Cause 1: Uncalibrated Smartphone Tilt & Gravity Vector Leakage
- In authentic smartphone recordings (IO-VNBD), smartphones are mounted on dashboard cradles with static pitch ($\sim 5.6^\circ$ to $8.5^\circ$) and roll.
- A static pitch tilt of $\theta = 6.8^\circ$ leaks Earth's gravity into the forward longitudinal acceleration:
  $$a_{\text{leak}} = g \cdot \sin(6.8^\circ) \approx 1.16\text{ m/s}^2$$
- In pure inertial double-integration over $T$ seconds, this constant unmodeled specific force produces quadratic position error:
  $$\Delta p(T) = \frac{1}{2} a_{\text{leak}} T^2$$
  - At $T = 10\text{s}$: $\Delta p = \frac{1}{2} (1.16) (100) = 58\text{ meters}$
  - At $T = 30\text{s}$: $\Delta p = \frac{1}{2} (1.16) (900) = 522\text{ meters}$
  - At $T = 60\text{s}$: $\Delta p = \frac{1}{2} (1.16) (3600) = 2088\text{ meters}$
- Because the vehicle was in continuous motion throughout the drives, the online stationary gravity aligner never completed a static leveling lock, causing the nominal gravity vector to misalign with the sensor body frame.

### Root Cause 2: Innovation Gate Lockout (False Rejection of AI Observations)
- Across the 30 evaluated runs, **798 out of 1000 AI forward speed updates (79.8%) were rejected** by the chi-square innovation gating in `15state_full`.
- **Mechanism:** As specific force integrates during the 1.0s interval between neural inferences, filter forward speed climbs by $\sim 1.2\text{ m/s}$. The innovation $\nu = v_{ai} - v_{body\_x}$ exceeds the innovation gate ($NIS > \chi^2_1 = 9.0$).
- **Consequence:** The filter rejects the AI velocity update, abandoning speed feedback. The velocity continues to climb unbounded ($>50\text{ m/s}$ in 10s), locking out all subsequent AI updates.

### Root Cause 3: Legacy Planar EKF Numerical Singularity
- The legacy 9-state EKF assumes a 2D planar vehicle model ($v_z = 0, \text{pitch} = 0, \text{roll} = 0$).
- When presented with 3D smartphone accelerometer vectors with large out-of-plane components, its $2\times 2$ covariance update becomes ill-conditioned, resulting in catastrophic numerical overflow ($>10^{15}\text{ m}$).
- The 15-state 3D ES-EKF remained 100% numerically stable and finite on all runs, demonstrating software robustness despite the physical sensor drift.

### Root Cause 4: Stationary Detection & ZUPT Behavior
- In `15state_full`, total ZUPT activations across 30 runs were **1409**.
- **357 activations (25.3%) occurred during true vehicle motion** (GT speed $>0.1\text{ m/s}$ during slow traffic/rolling stops).
- While the velocity-aware gating corrected in Phase 22 prevented catastrophic resets during high-speed cruising, low-speed traffic vibrations still occasionally trigger false zero-velocity updates.

---

## 6. Full Artifact & Trace Inventory

All raw benchmark data, machine-readable metrics, and high-frequency time-series traces have been preserved:

1. **Master Benchmark JSON:** [`reports/PHASE26_AUTHENTIC_BLACKOUT_BENCHMARK.json`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/PHASE26_AUTHENTIC_BLACKOUT_BENCHMARK.json)
2. **Representative CSV Time-Series Traces (10 Hz):**
   - 10s Turning: [`reports/traces/trace_Y1_10s_turning_15state_full.csv`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/traces/trace_Y1_10s_turning_15state_full.csv)
   - 10s Accel/Decel: [`reports/traces/trace_Y1_10s_accel_decel_15state_full.csv`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/traces/trace_Y1_10s_accel_decel_15state_full.csv)
   - 10s Straight: [`reports/traces/trace_Y1_10s_straight_15state_full.csv`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/traces/trace_Y1_10s_straight_15state_full.csv)
   - 30s Turning: [`reports/traces/trace_Y1_30s_turning_15state_full.csv`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/traces/trace_Y1_30s_turning_15state_full.csv)
   - 30s Accel/Decel: [`reports/traces/trace_Y1_30s_accel_decel_15state_full.csv`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/traces/trace_Y1_30s_accel_decel_15state_full.csv)
   - 30s Straight: [`reports/traces/trace_Y1_30s_straight_15state_full.csv`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/traces/trace_Y1_30s_straight_15state_full.csv)
   - 60s Accel/Decel: [`reports/traces/trace_Y1_60s_accel_decel_15state_full.csv`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/traces/trace_Y1_60s_accel_decel_15state_full.csv)
   - 60s Turning: [`reports/traces/trace_Y1_60s_turning_15state_full.csv`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/traces/trace_Y1_60s_turning_15state_full.csv)
   - 60s Straight: [`reports/traces/trace_Y1_60s_straight_15state_full.csv`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/traces/trace_Y1_60s_straight_15state_full.csv)

---

## 7. Two-Wheeler / Motorcycle Disjointness Statement

> [!IMPORTANT]
> **NO MOTORCYCLE PERFORMANCE CLAIMED:**
> The IO-VNBD dataset was collected inside a passenger car (four-wheeled automobile). Even if the 15-state ES-EKF achieves stability on automotive data, this experiment does **NOT** prove:
> 1. Motorcycle roll/lean dynamics during cornering.
> 2. Scooter/two-wheeler single-track lateral dynamics.
> 3. Engine vibration harmonics unique to two-wheelers.
> 4. Handlebar mount vs pocket orientation transitions.
>
> Authentic two-wheeler physical validation must be conducted separately on dedicated two-wheeler data.

---

## 8. Summary & Next Engineering Actions

### Separation of Results:
1. **Software Correctness:** **100% VERIFIED** (157/157 tests passing, bitwise deterministic execution).
2. **Scientific Validity:** **100% AUDITED** (Zero GT leakage, zero future data access, zero privileged baselines).
3. **Navigation Accuracy:** **POOR (Expected for uncalibrated raw IMU double integration)**.
4. **Generalization Evidence:** **EVALUATED on Held-Out Driver D / Y1**.
5. **SIH Compliance:** **NOT ACHIEVED (0% pass rate at <10% drift)**.
6. **Motorcycle Readiness:** **NOT DEMONSTRATED (Automotive data only)**.

### Priority Actions for Phase 27:
1. **Dynamic In-Motion Attitude Leveling:** Implement continuous gravity tracking during pre-blackout driving to estimate static pitch/roll mounting offsets.
2. **Adaptive AI Innovation Gating:** Implement covariance inflation and progressive gating for AI speed pseudo-measurements to eliminate the "false rejection filter lockout" failure mode.
3. **Continuous Accelerometer Bias Observability:** Couple AI forward velocity with Kalman accelerometer bias states ($b_a$) to estimate gravity leakage during outages.
