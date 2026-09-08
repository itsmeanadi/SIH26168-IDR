# Forensic Audit: Real Android Phone Live Data Path & Provenance Integrity

**Date:** 2026-09-08  
**System:** SIH26168 IDR Two-Wheeler / Automotive Navigation Engine  
**Verification Scope:** End-to-End Live Physical Android Chrome Data Path vs. Replay / Fallback / Diagnostics Discrepancies  

---

## 1. Executive Summary & Root Cause Analysis

During stationary testing over HTTPS at port 8443, two phenomena occurred:
1. **Initial Changing Heading (031° → 049° → 160° → 198°):** When the stationary phone was first tested, the ES-EKF had no stationary compass fusion active, so consumer MEMS gyro $Z$-axis uncalibrated bias ($\approx 0.015\text{ rad/s} \approx 0.86^\circ/\text{s}$) integrated freely over 2 minutes.
2. **Diagnostics Modal Showing "Standby" for All Sensors:** The Diagnostics modal rendered static HTML placeholders (`○ Standby`, `TRUSTED`, `100%`) because `showDiagnosticsModal()` only un-hid the DOM without forcing an immediate state re-render, and `updateHardwareTelemetry()` fell back to `'○ Standby'` if `rateHz` was 0 during intermediate callback intervals.

---

## 2. End-to-End Data Provenance Matrix

| Variable | Display Element | Exact Data Source | Provenance Category | Live Mode Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **HUD Speed** | `#val-speed-kmh`, `#val-speed-mps` | 15-State ES-EKF forward velocity $v_{\text{fwd}} = v_E\cos\psi + v_N\sin\psi$ | `SERVER_DERIVED` | ZUPT locks strictly at $0.00\text{ m/s}$ (0 km/h) at rest; responds dynamically to hand motion |
| **Heading** | `#val-heading-display` | Rest: Magnetometer (`DeviceOrientationEvent.alpha`)<br>Motion ($\ge 1.5\text{ m/s}$): GNSS Course-Over-Ground + Gyro | `REAL_DEVICE` (Rest)<br>`SERVER_DERIVED` (Motion) | Locked to true compass azimuth at rest; aligns with direction of travel when moving |
| **Lean Angle** | `#val-lean-text`, `#lean-indicator-marker` | Kinematic Roll $\phi = \arctan(v \cdot \dot{\psi} / g)$ clamped to $\pm 45^\circ$ | `SERVER_DERIVED` | Exactly $0.0^\circ$ at rest; responds to dynamic turns when $v > 0$ |
| **Map Trajectory** | Leaflet Polyline `#2563eb` | Geodetic $(lat, lon)$ projection from ES-EKF / Live GNSS Fixes | `REAL_DEVICE` + `SERVER_DERIVED` | Clustered around user's physical GPS location; advances when user moves |
| **GPS Fix Status** | `#diag-hw-gps` | `navigator.geolocation.watchPosition` | `REAL_DEVICE` | Shows real-time accuracy ($\pm X\text{m}$) and fix state |
| **IMU Telemetry** | `#diag-hw-accel`, `#diag-hw-gyro` | `window.devicemotion` ($a_x, a_y, a_z\text{ m/s}^2$, $g_x, g_y, g_z\text{ rad/s}$) | `REAL_DEVICE` | Streams at $\approx 50\text{ Hz}$ with live 3-axis vector readouts |

---

## 3. Separation of Live and Replay Modes

- **Replay Isolation:** Replayer datasets (IO-VNBD `Vf`, `M`, `S`, `Y1`) only stream when Replay Demonstration is explicitly triggered. Live sensor frames immediately halt replay playback.
- **Zero Mock Data in Live Mode:** No synthetic or simulated coordinates/speeds are injected into the live navigation pipeline.
- **Startup Protection:** Startup prior to the first physical GPS fix is marked `has_physical_gps_fix = False`, preventing false blackout timer initiation.

---

## 4. Verification Test Suite Status

- **Live Provenance Tests (`test_live_provenance_and_integrity.py`):** 4/4 Passed (100%)
- **Physical Stationary Invariance Tests (`test_live_stationary_physical_integrity.py`):** 5/5 Passed (100%)
- **Total Repository Test Suite:** **214 / 214 Passed (100%)**
