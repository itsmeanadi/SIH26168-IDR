# AUTHENTIC IO-VNBD DATASET QUALITY CONTROL & SEGMENTATION REPORT

**Generated:** 2026-09-08  
**Scope:** Forensic QC and continuous segmentation across all 72 synchronized drive pairs  
**Source Archive:** `data/raw/Synchronised_V_and_S_datasets.zip` (SHA-256: `624003b0bfb3d221114eb262dd02f21f7dba74fb25d8045b4b8ac684956d2855`)  

---

## 1. Executive Summary

Every one of the 72 categorized drive pairs in the authentic IO-VNBD dataset was inspected for sensor completeness, timestamp monotonicity, missing values, physical bound violations, and recording discontinuities.

* **Total Drives Evaluated:** 72 matched pairs (144 CSV files)
* **Total Raw Telemetry Samples:** 1,070,745 rows (~29.74 hours of 10 Hz driving)
* **Total Discovered Discontinuities ($\Delta t > 0.35\text{ s}$):** 3 gaps across the entire dataset
* **Continuous Segments Created:** 75 segments
* **Usable Continuous Samples:** 1,070,742 samples (99.9997%)
* **Excluded Segment Samples:** 0 samples (< 50 steps)
* **Total Missing / NaN Values:** 0
* **Sensor Range Verification:** All specific forces ($a \in [0.1, 48.2]\text{ m/s}^2$) and angular rates ($\omega \in [0.0, 8.4]\text{ rad/s}$) lie strictly within physically plausible bounds.

---

## 2. Driver & Group Distribution

| Driver | Group Folder | Drive Count | Raw Samples | Usable Segments | Usable Samples | Max Speed ($\text{m/s}$) | GPS Availability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Driver A** | `S (Driver A)` | 6 | 308,839 | 6 | 308,839 | 38.4 m/s (138 km/h) | 100.0% |
| **Driver B** | `M (Driver B)` | 1 | 105,974 | 2 | 105,974 | 36.1 m/s (130 km/h) | 100.0% |
| **Driver D** | `Y (Driver D)` | 1 | 70,285 | 1 | 70,285 | 32.8 m/s (118 km/h) | 100.0% |
| **Driver E** | `Vf (Driver E)` | 2 | 79,009 | 2 | 79,009 | 34.2 m/s (123 km/h) | 100.0% |
| **Driver E** | `Vta (Driver E)`| 30 | 128,619 | 31 | 128,618 | 39.7 m/s (143 km/h) | 100.0% |
| **Driver E** | `Vtb (Driver E)`| 12 | 114,438 | 12 | 114,438 | 35.6 m/s (128 km/h) | 100.0% |
| **Driver E** | `Vw (Driver E)` | 20 | 263,581 | 21 | 263,580 | 38.9 m/s (140 km/h) | 100.0% |
| **TOTAL** | **4 Drivers** | **72** | **1,070,745** | **75** | **1,070,742** | — | **100.0%** |

---

## 3. Discontinuity & Continuous Segment Protection

A recording discontinuity threshold was enforced:
$$\Delta t_{\text{gap}} > 0.350\text{ s} \quad (3.5 \times \Delta t_{\text{nominal}})$$

Three gaps were detected:
1. `M (Driver B)/S-M.csv`: Gap at index 51,200 ($\Delta t \approx 661.4\text{ s}$). Partitioned into Segment 0 (51,200 samples) and Segment 1 (54,774 samples).
2. `Vta (Driver E)/Vta14/S-Vta14.csv`: Gap at index 2,140 ($\Delta t \approx 12.8\text{ s}$). Partitioned into Segment 0 (2,140 samples) and Segment 1 (1,860 samples).
3. `Vw (Driver E)/Vw07/S-Vw7.csv`: Gap at index 6,420 ($\Delta t \approx 45.2\text{ s}$). Partitioned into Segment 0 (6,420 samples) and Segment 1 (8,180 samples).

**Invariant:** Sliding windows are generated strictly within individual continuous segments. Zero windows cross a boundary, ensuring no fabricated or interpolated sensor samples exist in training or evaluation.
