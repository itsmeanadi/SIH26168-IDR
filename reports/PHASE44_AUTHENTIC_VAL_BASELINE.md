# Experiment Report: authentic_baseline_val_drive_m

## 1. Provenance & Metadata
- **Provenance Type:** `AUTHENTIC_DATASET`
- **Dataset Role:** `VALIDATION`
- **Dataset / Session:** `M (Driver B)`
- **Source Path:** `data\raw\categorised_authentic\M (Driver B)`
- **Timestamp (UTC):** `2026-09-08T19:28:33.283817+00:00`
- **Model Checkpoint:** `models\velocity_net.pt`
- **Model SHA256:** `eb6c50d9b964df205ae85916550b26af47284a5434d05d2774aabe1a39bd49b6`
- **Model Authenticity:** `historical_synthetic`
- **Scientific Research Valid:** `NO`
- **Leakage Audit:** `PASSED`
- **Notes:** Evaluated on authentic IO-VNBD dataset M (Driver B) against CAN ground truth.

## 2. Sampling & Temporal Breakdown
- **Total Duration:** `10597.40 s` (10593 samples)
- **Effective Sample Rate:** `10.00 Hz`
- **$\Delta t$ Statistics:** Mean: `100.00 ms`, Median: `100.00 ms`, Std: `0.50 ms` (Min: `99.0 ms`, Max: `101.0 ms`)
- **Stationary Duration:** `1233.00 s`
- **Movement Duration:** `9360.00 s`

## 3. Velocity & Motion Estimation Metrics
| Subsystem | Peak Speed | Mean Speed | Reference Peak | Reference Mean | MAE | RMSE | Mean Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AI VelocityNet** | 17.27 m/s | 17.27 m/s | 27.97 m/s | 9.92 m/s | 7.906 m/s (28.46 km/h) | 9.286 m/s (33.43 km/h) | +7.356 m/s |
| **ES-EKF Velocity** | 17.27 m/s | 17.27 m/s | 27.97 m/s | 9.92 m/s | 7.906 m/s (28.46 km/h) | 9.286 m/s (33.43 km/h) | +7.356 m/s |

## 4. Dead Reckoning & Navigation Filter State
- **Total DR Distance:** `182991.14 m`
- **Reference Ground-Truth Distance:** `105066.50 m`
- **Endpoint Position Error:** `N/A`
- **Final Heading:** `0.0°`
- **ZUPT Intervals Triggered:** `1233`
- **AI Updates Accepted / Rejected:** `10593` accepted, `0` rejected (100.0% acceptance)
- **AI Mean NIS:** `0.00`
- **GNSS Availability State:** `CAN_REFERENCE_VALID`
- **Blackout Duration:** `0.00 s`
