# System Architecture & Design Decisions (SIH 26168)

> [!WARNING]
> **PROVENANCE AUDIT NOTICE: HISTORICAL SYNTHETIC DEVELOPMENT BASELINE**  
> All numerical metrics referenced in these decision records (e.g., drift percentages, MAE values) reflect historical software pipeline simulation baselines on synthetic data. Performance on authentic road data has not yet been established.

---
# Architecture & Engineering Decisions (DECISIONS.md)
*SIH PS 26168 (ISRO): AI-ML Intelligent Dead Reckoning (IDR)*

---

### DEC-001: IO-VNBD Dataset Subset & Split Strategy
- **Decision**: Focus on the **Synchronised Categorised** subset (drives: `M`, `S`, `Vf`, `Vta`, `Vtb`, `Vw`, `Y1`).
- **Rationale**: The uncategorised/unsynchronised datasets contain variable time-drifts and unlabelled drive contexts. Categorised drives provide clear scenario labels (motorway, roundabout, urban, hard-braking).
- **Split Strategy**: Strictly **drive-level (track-level)** partitioning to prevent temporal leakage:
  - **Train**: `M`, `S`, `Vta`, `Vtb`, `Vw`
  - **Val**: `Y1`
  - **Test**: `Vf` (motorway trajectory with extended high-speed straight-line segments ideal for assessing pure dead-reckoning drift).

---

### DEC-002: Cross-Modal IMU & Speed Supervision
- **Decision**: Use smartphone IMU (`S-*.csv`) as primary sensor input during inference; use vehicle ECU signals (`V-*.csv`) during training as supervision targets.
- **Rationale**:
  - In deployed operation, the user only has a smartphone mounted in the vehicle (no CAN/OBD access).
  - Vehicle wheel speeds provide low-noise, slip-aware ground truth for training the deep learning forward-velocity estimator.
  - Vehicle tactical IMU provides clean ground truth for training the IMU Denoise / Bias Net.

---

### DEC-003: Model Architectures & Edge/Mobile Constraints
- **IMU Denoise Net**:
  - **Choice**: Lightweight 1D Dilated Temporal Convolutional Network (TCN) or 1D-CNN (receptive field ~1.0 s / 10 samples).
  - **Output**: Regressed bias and high-frequency noise residual $\Delta a, \Delta \omega$.
- **Forward-Velocity Estimator**:
  - **Choice**: Compact TCN with causal convolutions + global pooling or 1-layer GRU (hidden dim 32-64, params < 100k).
  - **Motion Rejection**: Input window features explicit motion energy thresholding to reject phone re-mount shocks, road bumps (potholes), and idle engine vibration ($v=0$).
- **Export Compatibility**:
  - All layers strictly maintain ONNX opset 14+ and standard TFLite ops (Conv1d, Linear, ReLU/GELU, LayerNorm/BatchNorm). Avoid dynamic control flow inside model graph.

---

### DEC-004: Sensor Fusion Architecture (EKF + NHC)
- **Decision**: 9-state Extended Kalman Filter operating in Local ENU (East-North-Up) / Vehicle Frame:
  - State vector: $x = [p_e, p_n, p_u, v_e, v_n, v_u, \psi, b_a, b_\omega]^T$.
- **Non-Holonomic Constraints (NHC)**:
  - Ground vehicles satisfy no lateral slip ($v_y \approx 0$) and no vertical bounce ($v_z \approx 0$) in the vehicle body frame during regular driving.
  - Formulated as virtual pseudo-measurements updated in the EKF at 10 Hz when GNSS is lost:
    $$z_{nhc} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} + v_{noise}, \quad H_{nhc} = R_{body \to ENU}^T$$
- **Re-convergence on GNSS recovery**:
  - Continuous covariance update prevents discontinuous trajectory jumps ("teleportation") when GNSS signals re-acquire after an outage.

---

### DEC-005: Map-Matching via HMM / Viterbi Snapping
- **Decision**: Hidden Markov Model (HMM) Viterbi algorithm matching dead-reckoned trajectory to candidate road segments extracted from local OpenStreetMap graph.
- **Offline Guarantee**: Pre-cached OSM network (`.graphml` or `.pbf`) stored locally. Emission probability models orthogonal distance to road centerline; transition probability models route distance consistency vs Euclidean distance.

---

### DEC-006: Allan-Variance MEMS Process Noise & Heading Observability
- **Decision**: Calibrate EKF process noise $Q$ and initial covariance $P$ using smartphone MEMS Allan variance parameters ($Q_{gyro} = (10^{-4})^2, P_{gyro} = 0.02^2$). Update NHC measurement Jacobian to explicitly include heading sensitivity:
  $$\frac{\partial v_{lat}}{\partial \psi} = -\cos(\psi) v_E - \sin(\psi) v_N$$
- **Measured Impact**: Restores heading observability during outage without artificial subspace zeroing. Reduces dead-reckoning core drift from 11.95% to 4.02% without map-matching.

---

### DEC-007: 2D Inertial Odometry with Heteroscedastic NLL Loss
- **Decision**: Deploy `InertialOdomNet` (ResNet1D + GRU) predicting body-frame 2D displacement $[dx, dy]$ and aleatoric log-variance $[\log\sigma_x^2, \log\sigma_y^2]$, trained via Gaussian Negative Log-Likelihood.
- **Augmentation**: 5x physics-consistent data augmentation (speed scaling $0.5\times-2.0\times$, bias perturbations $\pm 0.3$ m/s$^2$, noise scaling, planar rotations) to prevent shortcut collapse on single-trajectory datasets.
- **Validation**: Achieves 1.721 m displacement MAE across held-out drives over 5.0 s windows (0.34 m/s equivalent).

---

### DEC-008: Neural Kalman Gain Estimation (KalmanNet)
- **Decision**: Integrate `KalmanNet` learned Kalman gain estimator alongside classical EKF. A lightweight GRU learns $K_t$ from innovation and state features, with a deterministic fallback to classical Riccati gain if condition checks or norm thresholds are exceeded.

---

### DEC-009: Multi-Scenario Statistical Study (443 Outage Segments)
- **Decision**: Reject single-segment cherry-picking. Evaluate dead-reckoning performance across 443 distinct GNSS outages across all 7 IO-VNBD drives, varying outage lengths (100m–1500m) and dynamic regimes (motorway straight, curve, speed variations, high-noise canyon).
- **Outcome**: Confirms median drift of 9.18% (<10%) for pure DR core without map-matching, and 1.16% median drift (73.1% pass rate) with OSM snapping.

---

### DEC-010: Vehicle Kinematic Profiles (CarProfile vs TwoWheelerProfile)
- **Decision**: Abstract vehicle kinematics into dynamic profiles. `TwoWheelerProfile` models roll-lean angle $\phi = \arctan(v \omega / g)$ and dynamically widens lateral tolerance $\sigma_{lat}$ during cornering up to 45°, preventing false slip penalties on motorcycles.

---

### DEC-011: C1 Continuous Reacquisition Smoothing
- **Decision**: Replace discrete state reset on GNSS recovery with a 1.5-second cosine-bell transition filter:
  $$w(t) = \frac{1}{2} \left(1 - \cos\left(\frac{\pi t}{T}\right)\right)$$
- **Impact**: Completely eliminates the 20–50 m position teleport discontinuity upon tunnel exit, delivering smooth navigation.

