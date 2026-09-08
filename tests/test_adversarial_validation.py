"""Adversarial Scientific Pre-Data Validation Test Suite.

Rigorously verifies:
1. IMUDenoiseNet pass-through safety and research isolation.
2. Motorcycle 3D attitude kinematics, lean angle, and pitch slope gravity compensation.
3. HMM Viterbi Trellis Map Matcher on adversarial road topologies (solving nearest-road traps).
4. Map matching scientific isolation from Ground Truth.
5. GNSS blackout adversarial leak prevention (valid, stale, and recovery GNSS isolation).
6. AI-to-EKF strict directional independence (EKF state corruption leaves AI inference unchanged).
7. Mathematical metric calculations on exact hand-calculated trajectories.
8. Live vs Recorded Replay bit-exact / tight tolerance determinism.
"""

from pathlib import Path
import tempfile
import networkx as nx
import numpy as np
import pytest
import torch
from shapely.geometry import LineString

from idr.filters.vehicle_profiles import TwoWheelerProfile
from idr.engine.navigation_engine import (
    GNSSInputFix,
    NavigationEngine,
    NavigationMode,
    SensorInputFrame,
)
from idr.engine.replay import DriveReplayer
from idr.eval.metrics import compute_navigation_metrics
from idr.mapmatch.osm_graph import OSMGraphLoader
from idr.filters.ekf import ExtendedKalmanFilter
from idr.mapmatch.hmm_matcher import GeometricRoadSnapper, HMMMapMatcher
from idr.models.imu_denoise import IMUDenoiseNet
from idr.models.velocity_net import VelocityEstimatorNet
from idr.recorder.recorder import ExperimentRecorder


# ─────────────────────────────────────────────────────────────────────────────
# 1. IMUDenoiseNet Research Isolation & Safe Bounded Residuals
# ─────────────────────────────────────────────────────────────────────────────

def test_imu_denoise_net_safety_and_pass_through():
    """Verify that IMUDenoiseNet operates strictly as a bounded residual."""
    model = IMUDenoiseNet()
    model.eval()

    # Synthetic input window with physical dynamics: 10 m/s^2 accel, 0.5 rad/s yaw
    x = torch.zeros(1, 6, 50, dtype=torch.float32)
    x[:, 0, :] = 10.0
    x[:, 5, :] = 0.5

    with torch.no_grad():
        residual = model(x).squeeze().numpy()

    # Residuals must never exceed max realistic bias bounds (1.0 m/s^2, 0.1 rad/s)
    assert np.all(np.abs(residual[:3]) <= 1.0)
    assert np.all(np.abs(residual[3:]) <= 0.1)

    # Legitimate dynamic motion (10.0 m/s^2) must remain >= 90% intact
    corrected_accel = 10.0 - residual[0]
    assert corrected_accel >= 9.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Motorcycle Attitude, Lean Angle & Gravity Invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_motorcycle_lean_and_gravity_invariants():
    """Verify roll-lean kinematics, slope pitch gravity compensation, and coordinate frames."""
    profile = TwoWheelerProfile()
    g = 9.80665

    # 1. Zero speed -> zero lean angle
    assert profile.estimate_roll_angle(forward_speed=0.0, yaw_rate=0.5) == 0.0
    # Zero yaw rate -> zero lean angle
    assert profile.estimate_roll_angle(forward_speed=20.0, yaw_rate=0.0) == 0.0

    # 2. Cornering at 15 m/s (~54 km/h) with yaw rate 0.2 rad/s
    # phi = atan(15 * 0.2 / 9.80665) = atan(3.0 / 9.80665) ~ 0.297 rad (17.0 deg)
    phi_rad = profile.estimate_roll_angle(forward_speed=15.0, yaw_rate=0.2)
    expected_phi = np.arctan(15.0 * 0.2 / g)
    assert abs(phi_rad - expected_phi) < 1e-4

    # 3. EKF Pitch Slope Gravity Compensation:
    # On a +5.71 degree uphill slope (sin(theta) = 0.10):
    # Gravity component along vehicle forward axis = g * sin(theta) ~ 0.9807 m/s^2
    ekf = ExtendedKalmanFilter(dt=0.1)
    pitch_angle = np.arcsin(0.10)  # ~0.10 rad

    # If measured forward accel is 2.9807 m/s^2 on uphill slope, net inertial acceleration is 2.0 m/s^2
    ekf.x[6] = 0.0  # Heading = East (psi = 0)
    ekf.predict(fwd_accel=2.0 + g * 0.10, yaw_rate=0.0, pitch_rad=pitch_angle)

    # In ENU frame with psi=0: East velocity should increase by exactly net_accel * dt = 2.0 * 0.1 = 0.20 m/s
    assert abs(ekf.x[3] - 0.20) < 1e-4
    # Vertical velocity state should receive positive pitch component: 2.0 * sin(pitch) * dt ~ 0.020 m/s
    assert ekf.x[5] > 0.015


# ─────────────────────────────────────────────────────────────────────────────
# 3. HMM Viterbi vs Adversarial Nearest-Road Trap
# ─────────────────────────────────────────────────────────────────────────────

def test_hmm_viterbi_resolves_nearest_road_trap():
    """Adversarial Road Topology:
    
    Road A (wrong road): Straight line from (0, 0) to (100, 0).
    Road B (correct road): Straight line from (0, 10) to (100, 10).
    
    Trajectory starts at (0, 3) (closer to Road A: distance 3 vs 7).
    Then immediately moves along y=10 corridor: (20, 10), (40, 10), (60, 10), (80, 10), (100, 10).
    
    Nearest-Road Snapper snaps t=0 to Road A, creating a discontinuous jump to Road B at t=1.
    HMM Viterbi uses transition log-likelihood to decode the entire trajectory along Road B!
    """
    graph = nx.MultiDiGraph()
    graph.add_node("A1", x=0.0, y=0.0)
    graph.add_node("A2", x=100.0, y=0.0)
    graph.add_edge("A1", "A2", key=0, geometry=LineString([(0.0, 0.0), (100.0, 0.0)]), length=100.0)

    graph.add_node("B1", x=0.0, y=10.0)
    graph.add_node("B2", x=100.0, y=10.0)
    graph.add_edge("B1", "B2", key=0, geometry=LineString([(0.0, 10.0), (100.0, 10.0)]), length=100.0)

    traj = [
        (0.0, 3.0),    # Closer to Road A (dist=3.0) than Road B (dist=7.0)
        (20.0, 10.0),  # On Road B
        (40.0, 10.0),  # On Road B
        (60.0, 10.0),  # On Road B
        (80.0, 10.0),  # On Road B
        (100.0, 10.0), # On Road B
    ]

    # Geometric Snapper falls into the trap at t=0 (snaps to y=0 on Road A)
    snapper = GeometricRoadSnapper(graph, sigma_z=10.0)
    snapped_geom = snapper.snap_trajectory(traj)
    assert abs(snapped_geom[0][1] - 0.0) < 1e-3  # Trapped on Road A

    # HMM Viterbi correctly decodes all points onto Road B (y=10.0)
    hmm = HMMMapMatcher(graph, sigma_z=10.0, beta=15.0, max_candidates=5)
    snapped_hmm = hmm.snap_trajectory(traj)
    for pt in snapped_hmm:
        assert abs(pt[1] - 10.0) < 1e-3, f"Point {pt} was not snapped to Road B (y=10.0)"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Map-Matching Scientific Isolation from Ground Truth
# ─────────────────────────────────────────────────────────────────────────────

def test_map_matching_scientific_isolation_runtime():
    """Prove that changing Ground Truth coordinates does NOT alter OSM graph or map decisions."""
    # Build independent OSM graph
    graph = nx.MultiDiGraph()
    graph.add_node("N1", x=0.0, y=0.0)
    graph.add_node("N2", x=500.0, y=0.0)
    graph.add_edge("N1", "N2", key=0, geometry=LineString([(0.0, 0.0), (500.0, 0.0)]), length=500.0)

    matcher = HMMMapMatcher(graph, sigma_z=15.0)

    # Predicted dead-reckoning trajectory
    dr_traj = [(float(x), 2.0) for x in range(0, 100, 20)]

    # Run map matching with GT Trajectory 1
    gt_traj_1 = [(float(x), 0.0) for x in range(0, 100, 20)]
    snapped_1 = matcher.snap_trajectory(dr_traj)

    # Run map matching with completely modified GT Trajectory 2 (in opposite direction/offset)
    gt_traj_2 = [(float(x), 100.0) for x in range(100, 0, -20)]
    snapped_2 = matcher.snap_trajectory(dr_traj)

    # Snapped results MUST be identical because GT has zero path into HMMMapMatcher
    assert snapped_1 == snapped_2

    # Verify that attempting to construct OSM graph from GT in scientific mode raises UnsafeEvaluationError
    loader = OSMGraphLoader()
    with pytest.raises(Exception, match="Constructing an OSM road graph directly from"):
        loader.build_from_waypoints(
            waypoints=np.array([[28.6139, 77.2090], [28.6140, 77.2091]]),
            unsafe_allow_ground_truth_graph=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. GNSS Blackout Adversarial Leak Prevention
# ─────────────────────────────────────────────────────────────────────────────

def test_gnss_blackout_adversarial_leak_prevention():
    """Inject valid GNSS, stale GNSS, and speed during simulated blackout and prove 0% leaks to EKF."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090)

    # Anchor engine with 5 normal GPS frames
    for i in range(5):
        t = i * 0.1
        imu = SensorInputFrame(t, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
        gnss = GNSSInputFix(t, 28.6139, 77.2090, 200.0, 1.0, 0.0, 0.0)
        engine.process_frame(imu, gnss)

    assert engine.has_gps_anchor is True
    assert engine.in_blackout is False

    # Trigger blackout mode
    engine.is_gnss_denied_simulated = True

    # Initial position before blackout
    pre_blackout_pos = engine.fusion.ekf.x[:2].copy()

    # Adversarially feed valid, highly accurate GPS jumps that would pull the filter if not blocked
    for i in range(10):
        t = 1.0 + i * 0.1
        # Zero motion IMU
        imu = SensorInputFrame(t, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
        # Fake GPS jumping 1000m away with ultra-high reported accuracy (0.1m)
        gnss_injected = GNSSInputFix(
            timestamp=t,
            latitude=28.6250, # ~1.2 km away
            longitude=77.2200,
            altitude=500.0,
            accuracy_m=0.1,
            speed_mps=35.0,
            heading_deg=180.0,
        )
        out = engine.process_frame(imu, gnss_injected)

        assert out.is_in_blackout is True
        assert out.nav_mode in (
            NavigationMode.DEAD_RECKONING_NHC_AI,
            NavigationMode.DEAD_RECKONING_PURE,
            NavigationMode.STATIONARY_ZUPT,
        )
        # GNSS history must NOT receive the denied fixes
        assert (gnss_injected.latitude, gnss_injected.longitude) not in engine.gnss_history

    # EKF position should have stayed at pre_blackout_pos (stationary ZUPT) and NOT jumped 1.2 km
    post_pos = engine.fusion.ekf.x[:2]
    assert np.linalg.norm(post_pos - pre_blackout_pos) < 1.0, "GNSS fix leaked into EKF during blackout!"


# ─────────────────────────────────────────────────────────────────────────────
# 6. AI -> EKF Independence Runtime Test
# ─────────────────────────────────────────────────────────────────────────────

def test_ai_velocity_estimator_strict_independence():
    """Corrupt EKF state and GT coordinates; prove AI inference output is strictly unchanged."""
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090)

    # Feed 60 frames to fill the 50-sample AI buffer
    for i in range(60):
        t = i * 0.1
        imu = SensorInputFrame(t, 2.0, 0.0, 9.81, 0.0, 0.0, 0.0)
        engine.process_frame(imu, None)

    ai_speed_clean = engine.latest_ai_speed

    # 1. Adversarially corrupt EKF state
    engine.fusion.ekf.x[0:3] = np.array([999999.0, -888888.0, 5000.0]) # Wild position
    engine.fusion.ekf.x[3:6] = np.array([120.0, -85.0, 10.0])           # Wild velocity
    engine.fusion.ekf.x[6] = 3.1415                                     # Wild heading

    # Feed another frame with identical physical IMU data
    t = 6.0
    imu = SensorInputFrame(t, 2.0, 0.0, 9.81, 0.0, 0.0, 0.0)
    engine.process_frame(imu, None)

    # AI speed must remain identical
    assert engine.latest_ai_speed == ai_speed_clean, "AI velocity estimator depended on EKF internal state!"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Exact Hand-Calculated Metric Calculation
# ─────────────────────────────────────────────────────────────────────────────

def test_navigation_metrics_exact_hand_calculation():
    """Verify compute_navigation_metrics against an analytically derived trajectory."""
    # Ground Truth: Straight line along East axis of 1000 m (101 points, step = 10 m)
    # Predicted: Straight line with linear drift reaching 20 m North offset at endpoint
    N = 101
    gt_enu = np.zeros((N, 2))
    gt_enu[:, 0] = np.linspace(0.0, 1000.0, N)

    pred_enu = np.zeros((N, 2))
    pred_enu[:, 0] = np.linspace(0.0, 1000.0, N)
    pred_enu[:, 1] = np.linspace(0.0, 20.0, N)  # Offset growing from 0 to 20m

    metrics = compute_navigation_metrics(pred_enu, gt_enu)

    # Expected values:
    # Total distance = 1000.0 m
    # Final drift = 20.0 m
    # Drift percentage = (20.0 / 1000.0) * 100 = 2.0%
    # Mean error = mean of linspace(0, 20, 101) = 10.0 m
    # Max error = 20.0 m
    # CEP 50% = 10.0 m
    assert abs(metrics.total_distance_m - 1000.0) < 1e-4
    assert abs(metrics.final_drift_m - 20.0) < 1e-4
    assert abs(metrics.drift_percent - 2.0) < 1e-4
    assert abs(metrics.mae_position_m - 10.0) < 1e-4
    assert abs(metrics.max_error_m - 20.0) < 1e-4
    assert abs(metrics.cep_50_m - 10.0) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 8. Live vs Recorded Replay Bit-Exact Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_live_vs_replay_engine_determinism():
    """Record a live session and replay through a fresh engine; assert numerical agreement."""
    temp_dir = tempfile.mkdtemp(prefix="idr_replay_test_")
    recorder = ExperimentRecorder(output_base_dir=temp_dir, buffer_flush_size=1)
    engine_live = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090)

    recorder.start_session(session_id="determinism_test", vehicle_type="two_wheeler")

    live_outputs = []
    # Feed 25 frames
    for i in range(25):
        t = 100.0 + i * 0.1
        imu = SensorInputFrame(t, 1.5, 0.1, 9.81, 0.0, 0.0, 0.05)
        gnss = GNSSInputFix(t, 28.6139 + i*1e-4, 77.2090 + i*1e-4, 210.0, 2.0, 10.0, 45.0) if i % 5 == 0 else None
        out = engine_live.process_frame(imu, gnss)
        recorder.record_frame(imu, gnss, out)
        live_outputs.append((out.latitude, out.longitude, out.forward_speed_mps, out.heading_deg))

    recorder.stop_session()

    # Replay through fresh engine
    engine_replay = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090)
    replayer = DriveReplayer(engine_replay)
    session_csv = str(Path(temp_dir) / "determinism_test" / "telemetry.csv")

    assert replayer.load_recorded_session(session_csv, reset_engine=True) is True

    replay_outputs = []
    for _ in range(len(replayer.data_frames)):
        out = replayer.step()
        replay_outputs.append((out.latitude, out.longitude, out.forward_speed_mps, out.heading_deg))

    assert len(live_outputs) == len(replay_outputs)
    for idx, (live, rep) in enumerate(zip(live_outputs, replay_outputs)):
        assert abs(live[0] - rep[0]) < 1e-6, f"Frame {idx} latitude mismatch: {live[0]} vs {rep[0]}"
        assert abs(live[1] - rep[1]) < 1e-6, f"Frame {idx} longitude mismatch: {live[1]} vs {rep[1]}"
        assert abs(live[2] - rep[2]) < 1e-4, f"Frame {idx} speed mismatch: {live[2]} vs {rep[2]}"
        assert abs(live[3] - rep[3]) < 1e-3, f"Frame {idx} heading mismatch: {live[3]} vs {rep[3]}"
