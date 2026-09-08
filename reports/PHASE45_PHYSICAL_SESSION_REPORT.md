# Experiment Report: phase45_physical_replay_exp_20260908_172204

## 1. Provenance & Metadata
- **Provenance Type:** `REAL_DEVICE_OBSERVATION`
- **Dataset Role:** `FIELD_TRIAL`
- **Dataset / Session:** `exp_20260908_172204_two_wheeler`
- **Source Path:** `data\sessions\exp_20260908_172204_two_wheeler\telemetry.csv`
- **Timestamp (UTC):** `2026-09-08T21:37:47.367284+00:00`
- **Model Checkpoint:** `models\velocity_net.pt`
- **Model SHA256:** `ef1f004f77e18223e5524932435083ec70fa1be27db981b3ba170e8245110b87`
- **Model Authenticity:** `authentic_iovnbd`
- **Scientific Research Valid:** `YES`
- **Leakage Audit:** `PASSED`
- **Notes:** Qualitative out-of-domain evaluation on real Android GPS-denied walk with authentic checkpoint.

## 2. Sampling & Temporal Breakdown
- **Total Duration:** `298.90 s` (5850 samples)
- **Effective Sample Rate:** `29.52 Hz`
- **$\Delta t$ Statistics:** Mean: `33.88 ms`, Median: `33.00 ms`, Std: `10.23 ms` (Min: `19.0 ms`, Max: `401.0 ms`)
- **Stationary Duration:** `93.90 s`
- **Movement Duration:** `104.29 s`

## 3. Velocity & Motion Estimation Metrics
| Subsystem | Peak Speed | Mean Speed | Reference Peak | Reference Mean | MAE | RMSE | Mean Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AI VelocityNet** | 11.70 m/s | 1.72 m/s | N/A | N/A | N/A | N/A | N/A |
| **ES-EKF Velocity** | 28.54 m/s | 1.52 m/s | N/A | N/A | N/A | N/A | N/A |

## 4. Dead Reckoning & Navigation Filter State
- **Total DR Distance:** `35.10 m`
- **Reference Ground-Truth Distance:** `Not Available (Field Trial)`
- **Endpoint Position Error:** `N/A`
- **Final Heading:** `127.8°`
- **ZUPT Intervals Triggered:** `52`
- **AI Updates Accepted / Rejected:** `4506` accepted, `1344` rejected (77.0% acceptance)
- **AI Mean NIS:** `24.21`
- **GNSS Availability State:** `GPS_PERMISSION_DENIED`
- **Blackout Duration:** `298.90 s`
