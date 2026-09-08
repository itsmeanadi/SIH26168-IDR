"""Scientific and Mathematical Fixes Test Suite for SIH 26168 IDR.

Covers:
1. Exact Analytical vs Numerical Finite-Difference EKF Jacobian Validation
2. IMUDenoiseNet Signal Preservation and Bounded Residual Verification
3. Hard-Failure Prevention on Missing Datasets (Zero Silent Fallbacks)
4. Motorcycle Roll Lean Kinematics and Pitch Slope Gravity Compensation
5. Genuine HMM Viterbi Map-Matching and Ground-Truth Safety
6. Motorcycle Engine Idle Vibration-Robust Stationary Detection (ZUPT/ZARU)
7. Soft Vertical Dynamics and Slope Handling
"""

from pathlib import Path
import numpy as np
import networkx as nx
from shapely.geometry import LineString
import pytest
import torch

from idr.filters.ekf import ExtendedKalmanFilter
from idr.filters.fusion import GNSSINSFusion
from idr.filters.vehicle_profiles import TwoWheelerProfile, CarProfile
from idr.filters.zupt import StationaryDetector
from idr.mapmatch.hmm_matcher import HMMMapMatcher, GeometricRoadSnapper
from idr.mapmatch.osm_graph import OSMGraphLoader
from idr.models.imu_denoise import IMUDenoiseNet
from idr.models.train_odom import prepare_all_drives, train_inertial_odom
from idr.data.provenance import UnsafeEvaluationError, SyntheticDataBlockedError
from idr.io.preprocess import create_sliding_windows


# =====================================================================
# 1. EXACT ANALYTICAL VS NUMERICAL FINITE-DIFFERENCE EKF JACOBIAN
# =====================================================================

def test_ekf_analytical_vs_numerical_jacobian_straight_motion():
    """Verify analytical Jacobian matches finite-difference Jacobian during straight driving."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    
    # State: East=100m, North=200m, Up=0m, vE=12m/s, vN=8m/s, vU=0, psi=33 deg, ba=0.05, bw=0.005
    x = np.array([100.0, 200.0, 0.0, 12.0, 8.0, 0.0, float(np.deg2rad(33.0)), 0.05, 0.005], dtype=np.float64)
    fwd_acc = 1.5   # 1.5 m/s^2 accel
    yaw_rate = 0.0  # Straight line

    F_analytical = ekf.compute_analytical_jacobian(x, fwd_acc, yaw_rate)
    F_numerical = ekf.compute_numerical_jacobian(x, fwd_acc, yaw_rate, eps=1e-6)

    max_abs_diff = float(np.max(np.abs(F_analytical - F_numerical)))
    assert max_abs_diff < 1e-4, f"Analytical Jacobian mismatch in straight motion: max diff = {max_abs_diff:.2e}"


def test_ekf_analytical_vs_numerical_jacobian_high_yaw_rate_turning():
    """Verify analytical Jacobian matches finite-difference Jacobian during high-yaw turning."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    
    # High-speed turn with significant centripetal acceleration and yaw rate
    x = np.array([50.0, -80.0, 5.0, 15.0, 10.0, 0.0, float(np.deg2rad(120.0)), -0.1, 0.02], dtype=np.float64)
    fwd_acc = 2.0
    yaw_rate = 0.25  # ~14 deg/s sharp turn
    pitch = float(np.deg2rad(4.0))  # 4 deg uphill slope

    F_analytical = ekf.compute_analytical_jacobian(x, fwd_acc, yaw_rate, pitch_rad=pitch)
    F_numerical = ekf.compute_numerical_jacobian(x, fwd_acc, yaw_rate, pitch_rad=pitch, eps=1e-6)

    max_abs_diff = float(np.max(np.abs(F_analytical - F_numerical)))
    assert max_abs_diff < 1e-4, f"Analytical Jacobian mismatch in turning motion: max diff = {max_abs_diff:.2e}"


def test_ekf_analytical_vs_numerical_jacobian_negative_yaw_turning():
    """Verify analytical Jacobian with negative yaw rate and varying headings."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    
    headings = [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3]
    for h in headings:
        x = np.array([0.0, 0.0, 0.0, 18.0, -5.0, 0.0, h, 0.15, -0.01], dtype=np.float64)
        F_ana = ekf.compute_analytical_jacobian(x, fwd_accel=0.5, yaw_rate=-0.18)
        F_num = ekf.compute_numerical_jacobian(x, fwd_accel=0.5, yaw_rate=-0.18, eps=1e-6)
        
        diff = float(np.max(np.abs(F_ana - F_num)))
        assert diff < 1e-4, f"Jacobian mismatch at heading {h}: {diff:.2e}"


# =====================================================================
# 2. IMUDenoiseNet SIGNAL PRESERVATION & TARGET DEFINITION
# =====================================================================

def test_imu_denoise_net_preserves_signal_dynamics():
    """Verify IMUDenoiseNet outputs bounded residuals and does not annihilate signal."""
    model = IMUDenoiseNet()
    model.eval()

    # Input: 50-step window with 5 m/s^2 forward acceleration
    x_input = torch.zeros((1, 6, 50), dtype=torch.float32)
    x_input[0, 0, :] = 5.0  # 5 m/s^2 forward acceleration
    x_input[0, 2, :] = 9.81 # gravity vertical

    with torch.no_grad():
        residual = model(x_input)

    # Residuals must be strictly bounded in realistic bias range (<= 1.0 m/s^2)
    assert residual.shape == (1, 6)
    assert float(torch.max(torch.abs(residual[0, :3]))) <= 1.0
    assert float(torch.max(torch.abs(residual[0, 3:]))) <= 0.1

    # Calibrated IMU preserves the primary physical acceleration
    u_corrected = x_input[0, :, -1] - residual[0]
    assert float(u_corrected[0]) >= 4.0, "Legitimate forward acceleration was destroyed by denoiser!"


def test_preprocessing_imu_target_defaults_to_zero_bias_without_clean_reference():
    """Verify preprocessing yields zero bias targets when no clean reference IMU exists."""
    imu_raw = np.ones((100, 6), dtype=np.float32) * 2.0
    speeds = np.ones(100, dtype=np.float32) * 10.0

    # No clean reference provided
    X, y_vel, y_imu = create_sliding_windows(imu_raw, speeds, window_size=50, clean_imu_ref=None)
    assert X.shape[0] > 0
    assert np.allclose(y_imu, 0.0), "y_imu should default to zero bias residual without clean reference"


# =====================================================================
# 3. REMOVAL OF SILENT RANDOM FALLBACKS
# =====================================================================

def test_train_odom_hard_fails_when_data_missing(tmp_path):
    """Verify train_inertial_odom hard-fails instead of creating random Gaussian arrays."""
    empty_dir = tmp_path / "empty_raw"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        prepare_all_drives(empty_dir)
    assert "Silent fallback to random tensors has been removed" in str(exc_info.value)


# =====================================================================
# 4. MOTORCYCLE ATTITUDE, LEAN, AND SLOPE PHYSICS
# =====================================================================

def test_motorcycle_lean_angle_kinematics():
    """Verify roll angle estimation phi = arctan(v * omega / g)."""
    profile = TwoWheelerProfile()
    g = 9.80665

    # 1. Straight driving -> 0 roll
    phi_straight = profile.estimate_roll_angle(forward_speed=15.0, yaw_rate=0.0)
    assert abs(phi_straight) < 1e-5

    # 2. Moderate turn: 15 m/s @ 0.1 rad/s -> a_c = 1.5 m/s^2 -> phi ≈ arctan(1.5 / 9.80665) ≈ 8.7 deg
    phi_turn = profile.estimate_roll_angle(forward_speed=15.0, yaw_rate=0.1)
    expected_phi = np.arctan(1.5 / g)
    assert np.isclose(phi_turn, expected_phi, atol=1e-3)

    # 3. Dynamic NHC sigma relaxes during lean
    sigma_straight, _ = profile.compute_nhc_sigmas(yaw_rate=0.0, forward_speed=15.0)
    sigma_turn, _ = profile.compute_nhc_sigmas(yaw_rate=0.25, forward_speed=18.0)
    assert sigma_turn > sigma_straight, "NHC lateral sigma should relax during sharp turns to avoid over-constraining"


def test_pitch_slope_gravity_compensation_in_ekf():
    """Verify that uphill pitch slope properly compensates longitudinal gravity."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    
    # Driving on 5 deg uphill slope with 0 m/s^2 net acceleration
    # Accelerometer measures +g * sin(5 deg) due to gravity tilt
    pitch_slope = float(np.deg2rad(5.0))
    g_meas = 9.80665 * np.sin(pitch_slope)  # ~0.854 m/s^2

    # Without pitch compensation, filter would accelerate
    x_uncomp = ekf.state_transition_function(ekf.x, fwd_accel=g_meas, yaw_rate=0.0, pitch_rad=0.0)
    v_uncomp = float(np.hypot(x_uncomp[3], x_uncomp[4]))

    # With pitch compensation, net forward acceleration is 0 -> velocity stays 0
    x_comp = ekf.state_transition_function(ekf.x, fwd_accel=g_meas, yaw_rate=0.0, pitch_rad=pitch_slope)
    v_comp = float(np.hypot(x_comp[3], x_comp[4]))

    assert v_uncomp > 0.05, "Uncompensated slope should induce false acceleration"
    assert np.isclose(v_comp, 0.0, atol=1e-4), "Compensated slope must maintain zero net acceleration"


# =====================================================================
# 5. GENUINE HMM VITERBI MAP-MATCHING & GT-LEAKAGE SAFETY
# =====================================================================

def test_hmm_map_matcher_viterbi_trellis_decoding():
    """Verify true HMM Viterbi Trellis correctly snaps trajectory to road network."""
    G = nx.MultiDiGraph()
    G.add_node(1, x=-200.0, y=0.0)
    G.add_node(2, x=200.0, y=0.0)
    line = LineString([(-200.0, 0.0), (200.0, 0.0)])
    G.add_edge(1, 2, 0, geometry=line, length=400.0)

    matcher = HMMMapMatcher(G, sigma_z=15.0, beta=10.0)

    # Simulated drifting trajectory along road at y=0 from x=-100 to x=+100
    noisy_traj = [
        (-100.0, 4.0),
        (-50.0, -5.0),
        (0.0, 6.0),
        (50.0, -3.0),
        (100.0, 5.0),
    ]

    snapped = matcher.snap_trajectory(noisy_traj)
    assert len(snapped) == len(noisy_traj)

    # Snapped points should lie exactly on the y=0 road centerline
    for pt in snapped:
        assert abs(pt[1] - 0.0) < 1e-3, f"Point {pt} was not snapped to road centerline y=0"


def test_map_matching_prohibits_ground_truth_graph_in_scientific_mode():
    """Verify that constructing map from ground truth trajectory waypoints raises error."""
    loader = OSMGraphLoader()
    gt_waypoints = np.random.randn(20, 2)

    with pytest.raises(UnsafeEvaluationError):
        loader.build_from_waypoints(gt_waypoints, unsafe_allow_ground_truth_graph=False)


# =====================================================================
# 6. MOTORCYCLE VIBRATION-ROBUST STATIONARY DETECTION
# =====================================================================

def test_stationary_detector_rejects_single_cylinder_engine_vibration():
    """Verify stationary detector remains stationary under high-frequency engine vibration."""
    detector = StationaryDetector(window_size=10, persistence_frames=1)

    # 1. Feed stationary baseline with 25 Hz single-cylinder engine idle vibration
    t = np.arange(25) * 0.1
    for i in range(25):
        vib_acc = np.array([
            0.8 * np.sin(2.0 * np.pi * 25.0 * t[i]),
            0.6 * np.cos(2.0 * np.pi * 25.0 * t[i]),
            9.81 + 1.2 * np.sin(2.0 * np.pi * 25.0 * t[i]),
        ])
        vib_gyro = np.array([
            0.02 * np.sin(2.0 * np.pi * 25.0 * t[i]),
            0.02 * np.cos(2.0 * np.pi * 25.0 * t[i]),
            0.01 * np.sin(2.0 * np.pi * 25.0 * t[i]),
        ])
        is_stat = detector.update(vib_acc, vib_gyro)

    assert is_stat is True, "Engine idle vibration should be recognized as stationary"

    # 2. Vehicle starts moving: translational acceleration and angular turn
    for i in range(15):
        move_acc = np.array([2.5, 0.5, 9.81])  # 2.5 m/s^2 forward pull
        move_gyro = np.array([0.05, 0.05, 0.25]) # Turn
        is_stat = detector.update(move_acc, move_gyro)

    assert is_stat is False, "Translational motion must promptly exit stationary state"
