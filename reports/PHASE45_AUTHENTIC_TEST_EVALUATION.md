# Experiment Report: phase45_authentic_test_drive_y1

## 1. Provenance & Metadata
- **Provenance Type:** `AUTHENTIC_DATASET`
- **Dataset Role:** `TEST`
- **Dataset / Session:** `Y1`
- **Source Path:** `data\raw\categorised_authentic\Y (Driver D)\Y1`
- **Timestamp (UTC):** `2026-09-08T21:37:35.004905+00:00`
- **Model Checkpoint:** `models\velocity_net.pt`
- **Model SHA256:** `ef1f004f77e18223e5524932435083ec70fa1be27db981b3ba170e8245110b87`
- **Model Authenticity:** `authentic_iovnbd`
- **Scientific Research Valid:** `YES`
- **Leakage Audit:** `PASSED`
- **Notes:** Evaluated on authentic IO-VNBD dataset Y1 against CAN ground truth.

## 2. Sampling & Temporal Breakdown
- **Total Duration:** `7028.50 s` (7024 samples)
- **Effective Sample Rate:** `10.00 Hz`
- **$\Delta t$ Statistics:** Mean: `100.00 ms`, Median: `100.00 ms`, Std: `0.50 ms` (Min: `99.0 ms`, Max: `101.0 ms`)
- **Stationary Duration:** `1023.00 s`
- **Movement Duration:** `6001.00 s`

## 3. Velocity & Motion Estimation Metrics
| Subsystem | Peak Speed | Mean Speed | Reference Peak | Reference Mean | MAE | RMSE | Mean Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AI VelocityNet** | 23.07 m/s | 7.83 m/s | 23.56 m/s | 8.33 m/s | 2.680 m/s (9.65 km/h) | 3.641 m/s (13.11 km/h) | -0.499 m/s |
| **ES-EKF Velocity** | 23.07 m/s | 7.83 m/s | 23.56 m/s | 8.33 m/s | 2.680 m/s (9.65 km/h) | 3.641 m/s (13.11 km/h) | -0.499 m/s |

## 4. Dead Reckoning & Navigation Filter State
- **Total DR Distance:** `55028.42 m`
- **Reference Ground-Truth Distance:** `58534.31 m`
- **Endpoint Position Error:** `N/A`
- **Final Heading:** `0.0°`
- **ZUPT Intervals Triggered:** `1023`
- **AI Updates Accepted / Rejected:** `7024` accepted, `0` rejected (100.0% acceptance)
- **AI Mean NIS:** `0.00`
- **GNSS Availability State:** `CAN_REFERENCE_VALID`
- **Blackout Duration:** `0.00 s`
