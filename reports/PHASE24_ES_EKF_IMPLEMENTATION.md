# Phase 24 — 15-State 3D Error-State Extended Kalman Filter (ES-EKF) Implementation Report

**Document ID:** `REP-SIH26168-PHASE24-ES-EKF-IMPL-20260908`  
**Author:** Autonomous Senior Navigation & INS Engineer  
**Date:** 2026-09-08  
**Status:** **APPROVED & FULLY IMPLEMENTED**  
**Test Status:** **147 Passed, 0 Failed (100% Pass Rate across 17 test suites)**

---

## 1. Implementation Summary

In accordance with the verified mathematical specification from Phase 23.5, the complete **15-State 3D Error-State Extended Kalman Filter (ES-EKF)** has been implemented in [`src/idr/filters/es_ekf.py`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/src/idr/filters/es_ekf.py).

The implementation includes:
- 16-parameter nominal state representation with unit quaternion attitude mechanics.
- 15-parameter right-multiplicative body-frame error state.
- Exact $SO(3)$ Lie algebra utilities (quaternion exponential map, Right Jacobian $\mathbf{J}_r(\mathbf{v})$).
- 3D specific force mechanization with ENU gravity vector $\mathbf{g}_n = [0, 0, -9.80665]^T \text{ m/s}^2$.
- Analytically derived discrete transition matrix $\mathbf{F}_d$ matching numerical finite differences to $1.41 \times 10^{-7}$.
- Cholesky-based measurement updates with Joseph-stabilized covariance propagation and $\chi^2$ Normalized Innovation Squared (NIS) outlier gating.
- Multiplicative quaternion error injection with post-update covariance reset $\mathbf{P} \leftarrow \mathbf{G} \mathbf{P} \mathbf{G}^T$.
- Standstill GNSS Course-Over-Ground (COG) heading suppression ($v_{\text{GNSS}} < 1.5\text{ m/s}$) to eliminate stationary yaw corruption.
- Complete modular integration with backward compatibility for the legacy 9-state EKF.

---

## 2. State & Covariance Definition

### Nominal State Vector $\mathbf{x} \in \mathbb{R}^{16}$
$$\mathbf{x} = \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \\ \mathbf{q} \\ \mathbf{b}_a \\ \mathbf{b}_g \end{bmatrix} = \begin{bmatrix} [p_E, p_N, p_U]^T & \text{(Local East-North-Up position, m)} \\ [v_E, v_N, v_U]^T & \text{(Local ENU velocity, m/s)} \\ [q_w, q_x, q_y, q_z]^T & \text{(Unit quaternion, Body} \to \text{ENU)} \\ [b_{ax}, b_{ay}, b_{az}]^T & \text{(Accelerometer bias in Body FLU frame, m/s}^2\text{)} \\ [b_{gx}, b_{gy}, b_{gz}]^T & \text{(Gyroscope bias in Body FLU frame, rad/s)} \end{bmatrix}$$

### Error State Vector $\delta\mathbf{x} \in \mathbb{R}^{15}$
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p} \\ \delta\mathbf{v} \\ \delta\boldsymbol{\theta}_b \\ \delta\mathbf{b}_a \\ \delta\mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$

### Error Covariance Matrix $\mathbf{P} \in \mathbb{R}^{15 \times 15}$
$$\mathbf{P} = \mathbb{E}[\delta\mathbf{x} \delta\mathbf{x}^T]$$

---

## 3. Error-State Attitude Convention

The filter strictly enforces the **Right-Multiplicative (Body-Frame)** attitude error convention:
$$\mathbf{q}_{\text{true}} = \mathbf{q} \otimes \delta\mathbf{q}, \quad \delta\mathbf{q} = \exp\left(\frac{1}{2}\delta\boldsymbol{\theta}_b\right) \approx \begin{bmatrix} 1 \\ \frac{1}{2}\delta\boldsymbol{\theta}_b \end{bmatrix}$$
$$\mathbf{R}_{\text{true}} = \mathbf{R}(\mathbf{q}) \mathbf{R}(\delta\mathbf{q}) \approx \mathbf{R}(\mathbf{q})\left(\mathbf{I}_3 + [\delta\boldsymbol{\theta}_b]_\times\right)$$

Where $[\mathbf{v}]_\times$ is the $3\times 3$ skew-symmetric matrix:
$$[\mathbf{v}]_\times = \begin{bmatrix} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{bmatrix}$$

---

## 4. IMU Kinematic Propagation

Given raw sensor measurements $\mathbf{a}_m, \boldsymbol{\omega}_m$ and sample interval $\Delta t$:
1. **Bias Correction:**
   $$\hat{\mathbf{a}}_b = \mathbf{a}_m - \mathbf{b}_a, \quad \hat{\boldsymbol{\omega}}_b = \boldsymbol{\omega}_m - \mathbf{b}_g$$
2. **Attitude Integration:**
   $$\boldsymbol{\Delta\theta} = \hat{\boldsymbol{\omega}}_b \Delta t, \quad \Delta\mathbf{q} = \exp(\boldsymbol{\Delta\theta})$$
   $$\mathbf{q}_{k+1} = \frac{\mathbf{q}_k \otimes \Delta\mathbf{q}}{\|\mathbf{q}_k \otimes \Delta\mathbf{q}\|}$$
3. **Specific Force Rotation & Acceleration:**
   $$\mathbf{a}_n = \mathbf{R}(\mathbf{q}_{k+1}) \hat{\mathbf{a}}_b + \mathbf{g}_n, \quad \mathbf{g}_n = [0, 0, -9.80665]^T$$
4. **Position & Velocity Integration (2nd-Order Discrete Form):**
   $$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2}\mathbf{a}_n \Delta t^2$$
   $$\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{a}_n \Delta t$$

---

## 5. Discrete Error Transition Matrix $\mathbf{F}_d$

The exact $15\times 15$ discrete state transition matrix $\mathbf{F}_d$ is constructed as:

$$\mathbf{F}_d = \begin{bmatrix}
\mathbf{I}_3 & \mathbf{I}_3 \Delta t & -\frac{1}{2}\mathbf{R}[\hat{\mathbf{a}}_b]_\times \Delta t^2 & -\frac{1}{2}\mathbf{R}\Delta t^2 & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{I}_3 & -\mathbf{R}[\hat{\mathbf{a}}_b]_\times \Delta t & -\mathbf{R}\Delta t & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{R}(\Delta\mathbf{q})^T & \mathbf{0}_{3\times 3} & -\mathbf{J}_r(\hat{\boldsymbol{\omega}}_b \Delta t) \Delta t \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{I}_3 & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{I}_3
\end{bmatrix}$$

Where $\mathbf{J}_r(\mathbf{v})$ is the Right Jacobian of $SO(3)$:
$$\mathbf{J}_r(\mathbf{v}) = \mathbf{I}_3 - \frac{1-\cos\|\mathbf{v}\|}{\|\mathbf{v}\|^2}[\mathbf{v}]_\times + \frac{\|\mathbf{v}\|-\sin\|\mathbf{v}\|}{\|\mathbf{v}\|^3}[\mathbf{v}]_\times^2$$

---

## 6. Process Noise Covariance $\mathbf{Q}_d$

$$\mathbf{Q}_d = \text{diag}\left(\frac{1}{4}\sigma_a^2 \Delta t^4 \mathbf{I}_3, \, \sigma_a^2 \Delta t \mathbf{I}_3, \, \sigma_g^2 \Delta t \mathbf{I}_3, \, \sigma_{ba}^2 \Delta t \mathbf{I}_3, \, \sigma_{bg}^2 \Delta t \mathbf{I}_3\right)$$

* Continuous noise spectral densities: $\sigma_a = 0.1\text{ m/s}^2/\sqrt{\text{Hz}}$, $\sigma_g = 0.01\text{ rad/s}/\sqrt{\text{Hz}}$, $\sigma_{ba} = 10^{-3}\text{ m/s}^3/\sqrt{\text{Hz}}$, $\sigma_{bg} = 10^{-4}\text{ rad/s}^2/\sqrt{\text{Hz}}$.

---

## 7. Measurement Models & Jacobians

| Measurement | Observation Function $h(\mathbf{x})$ | Error-State Jacobian $\mathbf{H}$ | Gating |
| :--- | :--- | :--- | :--- |
| **GNSS Position** | $\mathbf{p}$ | $[\mathbf{I}_3, \, \mathbf{0}_{3\times 12}]$ | $\chi^2(3) \le 11.35$ |
| **GNSS Velocity** | $\mathbf{v}$ | $[\mathbf{0}_{3\times 3}, \, \mathbf{I}_3, \, \mathbf{0}_{3\times 9}]$ | $\chi^2(3) \le 11.35$ |
| **AI Speed** | $\mathbf{e}_1^T \mathbf{R}^T \mathbf{v} = v_{bx}$ | $[\mathbf{0}_{1\times 3}, \, \mathbf{e}_1^T \mathbf{R}^T, \, [0, -v_{bz}, v_{by}], \, \mathbf{0}_{1\times 6}]$ | $(3\sigma)^2 \le 9.0$ |
| **NHC (Lateral/Vert)** | $[v_{by}, v_{bz}]^T$ | $[\mathbf{0}_{2\times 3}, \, [\mathbf{e}_2^T \mathbf{R}^T; \mathbf{e}_3^T \mathbf{R}^T], \, [v_{bz}, 0, -v_{bx}; -v_{by}, v_{bx}, 0], \, \mathbf{0}_{2\times 6}]$ | $\chi^2(2) \le 9.21$ |
| **ZUPT** | $\mathbf{v}$ | $[\mathbf{0}_{3\times 3}, \, \mathbf{I}_3, \, \mathbf{0}_{3\times 9}]$ | $\chi^2(3) \le 11.35$ |
| **ZARU** | $\boldsymbol{\omega}_m - \mathbf{b}_g$ | $[\mathbf{0}_{3\times 12}, \, \mathbf{I}_3]$ | $\chi^2(3) \le 11.35$ |

---

## 8. Multiplicative Error Injection & Covariance Reset

Upon measurement assimilation, the estimated error vector $\delta\hat{\mathbf{x}}$ is injected into the nominal state:
$$\mathbf{p} \leftarrow \mathbf{p} + \delta\hat{\mathbf{p}}, \quad \mathbf{v} \leftarrow \mathbf{v} + \delta\hat{\mathbf{v}}$$
$$\mathbf{q} \leftarrow \frac{\mathbf{q} \otimes \exp(\frac{1}{2}\delta\hat{\boldsymbol{\theta}}_b)}{\|\mathbf{q} \otimes \exp(\frac{1}{2}\delta\hat{\boldsymbol{\theta}}_b)\|}$$
$$\mathbf{b}_a \leftarrow \mathbf{b}_a + \delta\hat{\mathbf{b}}_a, \quad \mathbf{b}_g \leftarrow \mathbf{b}_g + \delta\hat{\mathbf{b}}_g$$

The error state is reset: $\delta\hat{\mathbf{x}} \leftarrow \mathbf{0}_{15}$.  
The covariance is reset to account for the non-linear attitude projection:
$$\mathbf{P}^+ = \mathbf{G} \mathbf{P}_{\text{post}} \mathbf{G}^T, \quad \mathbf{G} = \text{diag}\left(\mathbf{I}_3, \, \mathbf{I}_3, \, \mathbf{I}_3 - \frac{1}{2}[\delta\hat{\boldsymbol{\theta}}_b]_\times, \, \mathbf{I}_3, \, \mathbf{I}_3\right)$$

---

## 9. Numerical Safeguards

1. **Joseph-Form Stabilized Covariance:**
   $$\mathbf{P}_{\text{post}} = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P} (\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K}\mathbf{R}\mathbf{K}^T$$
2. **Cholesky-Based Inversion:** $\mathbf{S} = \mathbf{L}\mathbf{L}^T$, solving $\mathbf{L}\mathbf{y} = \boldsymbol{\nu} \implies \text{NIS} = \|\mathbf{y}\|^2$.
3. **Sensor Input Sanitization:** $\text{NaN}/\text{Inf}$ values are sanitized to neutral zero/gravity bounds.
4. **Integration $\Delta t$ Clamping:** Enforced $10^{-5}\text{s} \le \Delta t \le 1.0\text{s}$.
5. **Covariance Symmetrization:** Enforced $\mathbf{P} = \frac{1}{2}(\mathbf{P} + \mathbf{P}^T)$ on every cycle.

---

## 10. Standstill GNSS COG Heading Protection

In [`src/idr/filters/fusion.py`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/src/idr/filters/fusion.py), GNSS Course-Over-Ground (COG) heading updates are protected by a configurable minimum speed threshold:
```python
gnss_speed = dist / max(1e-3, self.dt)
if gnss_speed >= self.min_cog_speed_mps:  # Default: 1.5 m/s
    cog = np.arctan2(dn, de)
    self.ekf.update_heading(cog, R_yaw=0.03)
```
Stationary GPS jitter ($\Delta d \approx 0.05\text{m}$, speed $0.5\text{ m/s} < 1.5\text{ m/s}$) is strictly prevented from corrupting vehicle yaw.

---

## 11. Jacobian Verification Matrix

| Jacobian Block | Analytical vs. Finite-Difference Max Error | Status |
| :--- | :--- | :--- |
| $\mathbf{F}_d$ (15x15) | $1.41 \times 10^{-7}$ | **PASS** |
| $\mathbf{H}_{\text{ai}}$ (1x15) | $1.52 \times 10^{-6}$ | **PASS** |
| $\mathbf{H}_{\text{nhc}}$ (2x15) | $1.53 \times 10^{-6}$ | **PASS** |
| $\mathbf{H}_{\text{pos}}$ (3x15) | $0.000000$ | **PASS** |
| $\mathbf{H}_{\text{vel}}$ (3x15) | $0.000000$ | **PASS** |
| $\mathbf{H}_{\text{zupt}}$ (3x15) | $0.000000$ | **PASS** |
| $\mathbf{H}_{\text{zaru}}$ (3x15) | $0.000000$ | **PASS** |
| Multiplicative Reset $\mathbf{G}$ | $1.50 \times 10^{-7}$ | **PASS** |

---

## 12. Complete Test Suite Status

```
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\j08da\OneDrive\Desktop\168\AI-ML-IDR-System
configfile: pyproject.toml
testpaths: tests
collected 147 items

tests\test_adversarial_validation.py ........                            [  5%]
tests\test_ai_ekf_integration.py .............                           [ 14%]
tests\test_authentic_dataset_accounting.py ...                           [ 16%]
tests\test_authentic_preprocessing.py .......                            [ 21%]
tests\test_blackout_forensics.py ...                                     [ 23%]
tests\test_code_level_forensic_audit.py .....                            [ 26%]
tests\test_engine.py ..............                                      [ 36%]
tests\test_es_ekf.py ........................                            [ 52%]
tests\test_es_ekf_mathematical_consistency.py ......                     [ 56%]
tests\test_idr.py ....                                                   [ 59%]
tests\test_pre_data_freeze.py .....                                      [ 62%]
tests\test_recorder.py .....                                             [ 65%]
tests\test_scientific_fixes.py ...........                               [ 73%]
tests\test_scientific_provenance.py ...........                          [ 80%]
tests\test_server.py .......                                             [ 85%]
tests\test_temporal_alignment_sync.py ...........                        [ 93%]
tests\test_zupt_correction.py ..........                                 [100%]

====================== 147 passed, 2 warnings in 10.17s =======================
```

---

## 13. Known Limitations & Explicit Negative Boundary

1. **Two-Wheeler Lean Compensation:**
   - Motorcycle lean dynamic scaling ($\phi = \arctan(v\omega/g)$) remains parameter-isolated until authentic physical two-wheeler telemetry is acquired in Phase 29.
2. **Scientific Claims Prohibited at this Stage:**
   - We do NOT claim SIH $<10\%$ drift compliance yet (reserved for blackout re-benchmarking in Phase 27).
   - We do NOT claim real-world motorcycle validation from IO-VNBD automobile data.
   - We do NOT use EKF output as AI input (strict causal separation preserved).

---

## 14. Files Modified / Created

1. [`src/idr/filters/es_ekf.py`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/src/idr/filters/es_ekf.py) — **NEW** (15-State ES-EKF implementation)
2. [`src/idr/filters/__init__.py`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/src/idr/filters/__init__.py) — **MODIFIED** (Exported `ErrorStateKalmanFilter` and `ESEKFConfig`)
3. [`src/idr/filters/fusion.py`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/src/idr/filters/fusion.py) — **MODIFIED** (Standstill GNSS COG heading bug fixed, ES-EKF option added)
4. [`tests/test_es_ekf.py`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/tests/test_es_ekf.py) — **NEW** (24 rigorous unit/integration tests)
5. [`reports/PHASE24_ES_EKF_IMPLEMENTATION.md`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/PHASE24_ES_EKF_IMPLEMENTATION.md) — **NEW** (This report)
6. [`reports/PHASE24_ES_EKF_IMPLEMENTATION.json`](file:///c:/Users/j08da/OneDrive/Desktop/168/AI-ML-IDR-System/reports/PHASE24_ES_EKF_IMPLEMENTATION.json) — **NEW** (Machine-readable audit artifact)
