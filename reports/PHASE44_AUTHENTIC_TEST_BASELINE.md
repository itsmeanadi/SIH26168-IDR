# Experiment Report: authentic_baseline_test_drive_y1

## 1. Provenance & Metadata
- **Provenance Type:** `AUTHENTIC_DATASET`
- **Dataset Role:** `TEST`
- **Dataset / Session:** `Y1`
- **Source Path:** `data\raw\categorised_authentic\Y (Driver D)\Y1`
- **Timestamp (UTC):** `2026-09-08T19:28:20.525779+00:00`
- **Model Checkpoint:** `models\velocity_net.pt`
- **Model SHA256:** `eb6c50d9b964df205ae85916550b26af47284a5434d05d2774aabe1a39bd49b6`
- **Model Authenticity:** `historical_synthetic`
- **Scientific Research Valid:** `NO`
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
| **AI VelocityNet** | 17.27 m/s | 17.27 m/s | 22.05 m/s | 8.33 m/s | 9.089 m/s (32.72 km/h) | 10.396 m/s (37.43 km/h) | +8.943 m/s |
| **ES-EKF Velocity** | 17.27 m/s | 17.27 m/s | 22.05 m/s | 8.33 m/s | 9.089 m/s (32.72 km/h) | 10.396 m/s (37.43 km/h) | +8.943 m/s |

## 4. Dead Reckoning & Navigation Filter State
- **Total DR Distance:** `121337.76 m`
- **Reference Ground-Truth Distance:** `58525.40 m`
- **Endpoint Position Error:** `N/A`
- **Final Heading:** `0.0°`
- **ZUPT Intervals Triggered:** `1023`
- **AI Updates Accepted / Rejected:** `7024` accepted, `0` rejected (100.0% acceptance)
- **AI Mean NIS:** `0.00`
- **GNSS Availability State:** `CAN_REFERENCE_VALID`
- **Blackout Duration:** `0.00 s`
