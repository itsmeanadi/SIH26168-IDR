# Experiment Report: phase45_authentic_val_drive_m

## 1. Provenance & Metadata
- **Provenance Type:** `AUTHENTIC_DATASET`
- **Dataset Role:** `VALIDATION`
- **Dataset / Session:** `M (Driver B)`
- **Source Path:** `data\raw\categorised_authentic\M (Driver B)`
- **Timestamp (UTC):** `2026-09-08T21:37:47.114289+00:00`
- **Model Checkpoint:** `models\velocity_net.pt`
- **Model SHA256:** `ef1f004f77e18223e5524932435083ec70fa1be27db981b3ba170e8245110b87`
- **Model Authenticity:** `authentic_iovnbd`
- **Scientific Research Valid:** `YES`
- **Leakage Audit:** `PASSED`
- **Notes:** Evaluated on authentic IO-VNBD dataset M (Driver B) against CAN ground truth.

## 2. Sampling & Temporal Breakdown
- **Total Duration:** `10594.20 s` (10590 samples)
- **Effective Sample Rate:** `10.00 Hz`
- **$\Delta t$ Statistics:** Mean: `100.00 ms`, Median: `100.00 ms`, Std: `0.50 ms` (Min: `99.0 ms`, Max: `101.0 ms`)
- **Stationary Duration:** `1226.00 s`
- **Movement Duration:** `9364.00 s`

## 3. Velocity & Motion Estimation Metrics
| Subsystem | Peak Speed | Mean Speed | Reference Peak | Reference Mean | MAE | RMSE | Mean Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AI VelocityNet** | 24.03 m/s | 9.44 m/s | 27.79 m/s | 9.92 m/s | 2.139 m/s (7.70 km/h) | 3.078 m/s (11.08 km/h) | -0.481 m/s |
| **ES-EKF Velocity** | 24.03 m/s | 9.44 m/s | 27.79 m/s | 9.92 m/s | 2.139 m/s (7.70 km/h) | 3.078 m/s (11.08 km/h) | -0.481 m/s |

## 4. Dead Reckoning & Navigation Filter State
- **Total DR Distance:** `99968.38 m`
- **Reference Ground-Truth Distance:** `105058.84 m`
- **Endpoint Position Error:** `N/A`
- **Final Heading:** `0.0°`
- **ZUPT Intervals Triggered:** `1226`
- **AI Updates Accepted / Rejected:** `10590` accepted, `0` rejected (100.0% acceptance)
- **AI Mean NIS:** `0.00`
- **GNSS Availability State:** `CAN_REFERENCE_VALID`
- **Blackout Duration:** `0.00 s`
