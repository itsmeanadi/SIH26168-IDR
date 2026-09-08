# AUTHENTIC IO-VNBD DATASET PREPROCESSING SUMMARY

**Generated:** 2026-09-08  
**Scope:** Preprocessing, continuous segment protection, training-only scaling, and driver-disjoint tensor generation  
**Source Root:** `data/raw/categorised_authentic/`  
**Output Root:** `data/processed/authentic/`  

---

## 1. Preprocessing Configuration

* **Window Size:** $W = 50$ steps (5.0 seconds at nominal 10.0 Hz)
* **Window Stride:** $S = 10$ steps (1.0 second stride, 80% temporal overlap)
* **Gap Threshold:** $\Delta t_{\text{gap}} = 0.350\text{ s}$ ($> 3.5 \times \Delta t_{\text{nominal}}$)
* **Input Feature Channels (6):** `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`
  * Specific force in $\text{m/s}^2$
  * Angular velocity in $\text{rad/s}$
* **Target Channels:**
  * `targets_vel`: Forward vehicle reference speed in $\text{m/s}$ at the window endpoint ($t_{\text{end}-1}$).
  * `targets_imu`: Inertial sensor bias residual ($\vec{u}_{\text{noisy}} - \vec{u}_{\text{clean}}$) in $\text{m/s}^2$ and $\text{rad/s}$.

---

## 2. Driver-Disjoint Split Accounting

| Split | Assigned Drivers | Directory Groups | Drive Count | Raw Samples | Continuous Segments | Generated Windows | Window Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | Driver A + Driver E | `S`, `Vf`, `Vta`, `Vtb`, `Vw` | 70 | 894,486 | 72 | **89,130** | 83.50% |
| **Validation** | Driver B | `M` | 1 | 105,974 | 2 | **10,588** | 9.92% |
| **Test** | Driver D | `Y` (`Y1`) | 1 | 70,285 | 1 | **7,024** | 6.58% |
| **TOTAL** | **4 Drivers** | **7 Groups** | **72** | **1,070,745** | **75** | **106,742** | **100.00%** |

### Mathematical Driver-Disjointness Proof:
$$\text{Drivers}(\text{Train}) = \{\text{Driver A, Driver E}\}$$
$$\text{Drivers}(\text{Val}) = \{\text{Driver B}\}$$
$$\text{Drivers}(\text{Test}) = \{\text{Driver D}\}$$
$$\text{Drivers}(\text{Train}) \cap \text{Drivers}(\text{Val}) = \emptyset, \quad \text{Drivers}(\text{Train}) \cap \text{Drivers}(\text{Test}) = \emptyset, \quad \text{Drivers}(\text{Val}) \cap \text{Drivers}(\text{Test}) = \emptyset$$

---

## 3. Continuous Segment Protection Invariant

* **Discontinuity Detection:** For any sample pair where $t_{i+1} - t_i > 0.350\text{ s}$, the time-series is partitioned into discrete sub-segments $[0, i]$ and $[i+1, N]$.
* **No Gap Crossing:** Sliding windows $[s, s+W)$ are generated strictly within individual continuous segments:
  $$s \ge \text{seg\_start} \quad \text{and} \quad s+W \le \text{seg\_end}$$
* **No Synthetic Samples:** Zero linear interpolation or artificial sensor fabrication was introduced across recording gaps.

---

## 4. Normalization & Scaler Provenance

* **Training-Only Scaler:** Fitted strictly on the 89,130 training windows ($89,130 \times 50 = 4,456,500$ sample vectors).
* **Saved Path:** `data/processed/authentic/train_scaler.json`
* **Fitted Parameter Values:**
  * `acc_x`: Mean = $0.0051\text{ m/s}^2$, Std = $1.5218\text{ m/s}^2$
  * `acc_y`: Mean = $1.7244\text{ m/s}^2$, Std = $2.1480\text{ m/s}^2$
  * `acc_z`: Mean = $9.4582\text{ m/s}^2$, Std = $1.7931\text{ m/s}^2$
  * `gyro_x`: Mean = $0.0002\text{ rad/s}$, Std = $0.0841\text{ rad/s}$
  * `gyro_y`: Mean = $-0.0011\text{ rad/s}$, Std = $0.0927\text{ rad/s}$
  * `gyro_z`: Mean = $0.0004\text{ rad/s}$, Std = $0.1105\text{ rad/s}$
* **Zero Leakage:** Validation and test splits were strictly excluded from scaler computation.

---

## 5. Target Speed Physical Statistics

| Split | Min Speed | Max Speed | Mean Speed | Median Speed | Std Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | $0.00\text{ m/s}$ ($0.0\text{ km/h}$) | $36.52\text{ m/s}$ ($131.5\text{ km/h}$) | $13.17\text{ m/s}$ ($47.4\text{ km/h}$) | $11.80\text{ m/s}$ | $9.64\text{ m/s}$ |
| **Validation** | $0.00\text{ m/s}$ ($0.0\text{ km/h}$) | $27.95\text{ m/s}$ ($100.6\text{ km/h}$) | $9.92\text{ m/s}$ ($35.7\text{ km/h}$) | $8.45\text{ m/s}$ | $8.12\text{ m/s}$ |
| **Test** | $0.00\text{ m/s}$ ($0.0\text{ km/h}$) | $22.05\text{ m/s}$ ($79.4\text{ km/h}$) | $8.33\text{ m/s}$ ($30.0\text{ km/h}$) | $6.90\text{ m/s}$ | $6.75\text{ m/s}$ |

Target speeds are sourced directly from reference vehicle ECU transmission telemetry ($v_{\text{ref}} = v_{\text{km/h}} / 3.6$), completely independent of EKF outputs or model predictions.
