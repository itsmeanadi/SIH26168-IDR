"""Unified Real-Time Navigation Engine for Mobile / PWA Deployment.

Integrates:
1. Sensor Ingestion & Phone-to-Vehicle Alignment
2. AI Speed / Residual Estimation (PyTorch / ONNX / TorchScript)
3. 9-State EKF + Vehicle Kinematics (TwoWheelerProfile lean adaptation, NHC, ZUPT/ZARU)
4. GNSS Trust Engine (USP 1)
5. Navigation Health & Environment Diagnostics (USP 2)
6. GNSS Blackspot Outage Tracker (USP 3)
7. Explainable Crash Detection Engine (USP 4)
8. C1 Continuous Reacquisition Smoother
"""

from dataclasses import asdict, dataclass
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from idr.calib.alignment import PhoneToVehicleAligner
from idr.filters.ekf import ExtendedKalmanFilter
from idr.filters.fusion import GNSSINSFusion
from idr.filters.nhc import apply_nhc_update
from idr.filters.vehicle_profiles import CarProfile, TwoWheelerProfile, VehicleProfile
from idr.filters.zupt import StationaryDetector, apply_zaru, apply_zupt
from idr.eval.transition import ReacquisitionSmoother
from idr.models.velocity_net import VelocityEstimatorNet

from .blackspot import BlackspotTracker
from .crash_detector import CrashAlert, CrashDetector
from .gnss_trust import GNSSTrustEngine, GNSSTrustResult, GNSSTrustStatus
from .health import HealthDiagnosticEngine, HealthDiagnostics, NavigationMode
from .resampler import TimestampAwareAIResampler


@dataclass
class SensorInputFrame:
    timestamp: float
    # Accelerometer in m/s^2 (phone frame)
    acc_x: float
    acc_y: float
    acc_z: float
    # Gyroscope in rad/s (phone frame)
    gyro_x: float
    gyro_y: float
    gyro_z: float
    # Magnetometer (uT) or orientation (Euler angles / quaternion)
    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    orientation_yaw: Optional[float] = None
    orientation_pitch: Optional[float] = None
    orientation_roll: Optional[float] = None


@dataclass
class GNSSInputFix:
    timestamp: float
    latitude: float
    longitude: float
    altitude: float = 0.0
    accuracy_m: float = 3.0
    speed_mps: Optional[float] = None
    heading_deg: Optional[float] = None


@dataclass
class NavigationOutputState:
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    # Vehicle frame / world velocity
    forward_speed_mps: float
    velocity_east: float
    velocity_north: float
    heading_deg: float  # Compass heading (0=North, 90=East)
    heading_rad: float  # Mathematical ENU yaw (counter-clockwise from East)
    lean_angle_deg: float
    # Mode & Quality
    nav_mode: NavigationMode
    gnss_trust_score: float
    gnss_status: str
    pos_uncertainty_m: float
    # Flags
    is_in_blackout: bool
    is_stationary: bool
    is_aligned: bool
    # Diagnostics & USPs
    diagnostics: HealthDiagnostics
    active_blackspot_id: Optional[str]
    active_crash_alert: Optional[CrashAlert]


class NavigationEngine:
    """Production-ready real-time navigation engine coordinating all modules."""

    def __init__(
        self,
        ref_lat: float = 28.6139,
        ref_lon: float = 77.2090,
        vehicle_type: str = "two_wheeler",
        model_path: Optional[str] = None,
        dt: float = 0.1,
        gnss_stale_timeout_sec: float = 2.0,
    ):
        self.dt = dt
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.vehicle_type = vehicle_type
        self.gnss_stale_timeout_sec = gnss_stale_timeout_sec

        # 1. Profiles & Core Fusion
        self.two_wheeler_profile = TwoWheelerProfile()
        self.car_profile = CarProfile()
        self.current_profile: VehicleProfile = (
            self.two_wheeler_profile if vehicle_type == "two_wheeler" else self.car_profile
        )

        self.fusion = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=dt)
        self.aligner = PhoneToVehicleAligner()
        self.stationary_detector = StationaryDetector(window_size=10, acc_var_threshold=0.15)
        self.reacquisition_smoother = ReacquisitionSmoother(blend_duration_sec=1.5, dt=dt)

        # 2. USP Engines
        self.trust_engine = GNSSTrustEngine()
        self.blackspot_tracker = BlackspotTracker(min_duration_sec=2.0)
        self.crash_detector = CrashDetector()
        self.health_engine = HealthDiagnosticEngine()

        # 3. AI Velocity Model & Timestamp-Aware Resampler (10 Hz target, 5.0s temporal context)
        self.ai_model: Optional[VelocityEstimatorNet] = None
        self._init_ai_model(model_path)
        self.ai_resampler = TimestampAwareAIResampler(target_rate_hz=10.0, window_size=50)
        self.window_size = 50

        # 4. State & Buffers
        self.is_gnss_denied_simulated = False
        self.in_blackout = False
        self.blackout_start_time: Optional[float] = None
        self.total_dr_distance = 0.0
        self.prev_dr_pos_enu = np.zeros(2)
        self.latest_ai_speed = 0.0
        self.current_lean_angle = 0.0
        self.has_gps_anchor = False
        self.last_gnss_fix: Optional[GNSSInputFix] = None
        self.last_gnss_arrival_time: Optional[float] = None
        self.last_gnss_trust_res: Optional[GNSSTrustResult] = None
        self.last_gnss_valid: bool = False

        # Trajectory storage for rendering
        self.gnss_history: List[Tuple[float, float]] = []
        self.dr_history: List[Tuple[float, float]] = []
        self.fused_history: List[Tuple[float, float]] = []

    def _init_ai_model(self, model_path: Optional[str]):
        """Load trained AI speed estimator if available."""
        paths_to_try = [
            model_path,
            "models/velocity_net.pt",
            "../models/velocity_net.pt",
            os.path.join(os.path.dirname(__file__), "../../../models/velocity_net.pt"),
        ]
        for p in paths_to_try:
            if p and os.path.exists(p):
                try:
                    model = VelocityEstimatorNet(in_channels=6, hidden_dim=64, num_layers=2)
                    state = torch.load(p, map_location="cpu", weights_only=True)
                    model.load_state_dict(state)
                    model.eval()
                    self.ai_model = model
                    break
                except Exception:
                    pass

    @property
    def imu_window_buffer(self) -> List[np.ndarray]:
        """Sliding 10 Hz window buffer of 50 vehicle-frame IMU samples (5.0 seconds)."""
        return self.ai_resampler.ai_window_buffer

    def set_vehicle_type(self, vehicle_type: str):
        """Toggle between two-wheeler (motorcycle/scooter) and car dynamics."""
        self.vehicle_type = vehicle_type
        if vehicle_type == "two_wheeler":
            self.current_profile = self.two_wheeler_profile
        else:
            self.current_profile = self.car_profile

    def set_simulated_blackout(self, denied: bool):
        """Simulate GNSS tunnel or urban canyon blackout for testing."""
        self.is_gnss_denied_simulated = denied

    def process_frame(
        self,
        imu: SensorInputFrame,
        gnss: Optional[GNSSInputFix] = None,
    ) -> NavigationOutputState:
        """Single real-time integration step at IMU / 10 Hz rate."""
        t = imu.timestamp
        acc_raw = np.array([imu.acc_x, imu.acc_y, imu.acc_z], dtype=np.float32)
        gyro_raw = np.array([imu.gyro_x, imu.gyro_y, imu.gyro_z], dtype=np.float32)
        if not np.all(np.isfinite(acc_raw)):
            acc_raw = np.nan_to_num(acc_raw, nan=0.0, posinf=9.81, neginf=-9.81)
            if np.linalg.norm(acc_raw) < 1e-3:
                acc_raw[2] = 9.81
        if not np.all(np.isfinite(gyro_raw)):
            gyro_raw = np.nan_to_num(gyro_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Update Diagnostics Sensor Stats
        self.health_engine.update_sensor_stats(acc_raw, gyro_raw)

        # 2. Stationary / ZUPT Detection
        is_stationary = self.stationary_detector.update(acc_raw, gyro_raw)

        # 3. Dynamic Phone-to-Vehicle Alignment
        if not self.aligner.is_calibrated:
            current_speed = (
                float(gnss.speed_mps)
                if (gnss is not None and gnss.speed_mps is not None and np.isfinite(gnss.speed_mps))
                else None
            )
            self.aligner.update(
                acc_raw=acc_raw,
                gyro_raw=gyro_raw,
                is_stationary=is_stationary,
                speed_mps=current_speed,
            )

        # Rotate IMU measurements into vehicle frame [Forward X_v, Lateral Y_v, Vertical Z_v]
        if self.aligner.is_calibrated or self.aligner.z_phone is not None:
            acc_vehicle, gyro_vehicle = self.aligner.transform_imu(
                acc_raw.reshape(1, 3), gyro_raw.reshape(1, 3)
            )
            fwd_accel = float(acc_vehicle[0, 0])
            lat_accel = float(acc_vehicle[0, 1])
            vert_accel = float(acc_vehicle[0, 2])
            roll_rate = float(gyro_vehicle[0, 0])
            pitch_rate = float(gyro_vehicle[0, 1])
            yaw_rate = float(gyro_vehicle[0, 2])
        else:
            # Fallback when uncalibrated and vertical axis is not yet estimated
            fwd_accel = float(acc_raw[0])
            lat_accel = float(acc_raw[1])
            vert_accel = float(acc_raw[2])
            roll_rate = float(gyro_raw[0])
            pitch_rate = float(gyro_raw[1])
            yaw_rate = float(gyro_raw[2])

        # 4. Slide IMU window for AI velocity inference (Resampled to 10 Hz / 5.0s physical context)
        # Explicit Coordinate-Frame Contract:
        # [0]: fwd_accel  (a_v_x, m/s^2, Vehicle Forward)
        # [1]: lat_accel  (a_v_y, m/s^2, Vehicle Lateral)
        # [2]: vert_accel (a_v_z, m/s^2, Vehicle Vertical)
        # [3]: roll_rate  (g_v_x, rad/s, Vehicle Roll)
        # [4]: pitch_rate (g_v_y, rad/s, Vehicle Pitch)
        # [5]: yaw_rate   (g_v_z, rad/s, Vehicle Yaw)
        imu_vehicle_6d = np.array(
            [fwd_accel, lat_accel, vert_accel, roll_rate, pitch_rate, yaw_rate],
            dtype=np.float32,
        )
        new_10hz_sample = self.ai_resampler.add_sample(t, imu_vehicle_6d)

        # AI inference evaluates only when a new 10 Hz temporal sample is emitted (and buffer has 50 samples)
        if new_10hz_sample is not None and self.ai_model is not None and self.ai_resampler.is_ready:
            with torch.no_grad():
                win_tensor = torch.tensor(
                    np.stack(self.ai_resampler.ai_window_buffer).T[np.newaxis, :, :],
                    dtype=torch.float32,
                )
                pred_speed = self.ai_model(win_tensor)
                self.latest_ai_speed = float(pred_speed.squeeze().item())

        ai_speed = self.latest_ai_speed

        # 5. Two-Wheeler Lean Dynamics
        if isinstance(self.current_profile, TwoWheelerProfile):
            cur_speed = float(self.fusion.ekf.x[3] * np.cos(self.fusion.ekf.x[6]) + self.fusion.ekf.x[4] * np.sin(self.fusion.ekf.x[6]))
            self.current_lean_angle = self.current_profile.estimate_roll_angle(cur_speed, yaw_rate)
        else:
            self.current_lean_angle = 0.0

        # 6. EKF Prediction Step
        self.fusion.ekf.predict(fwd_accel, yaw_rate)

        # 7. Apply ZUPT & ZARU if stationary
        if is_stationary:
            apply_zupt(self.fusion.ekf, sigma_v=0.01)
            apply_zaru(self.fusion.ekf, gyro_z_raw=yaw_rate)

        # 8. GNSS Freshness Check & Fusion Decision
        is_new_gnss = False
        if gnss is not None and not self.is_gnss_denied_simulated:
            if np.isfinite(gnss.latitude) and np.isfinite(gnss.longitude):
                if self.last_gnss_fix is None:
                    is_new_gnss = True
                else:
                    # Deterministic freshness check: timestamp progression or coordinate delta
                    if gnss.timestamp is not None and self.last_gnss_fix.timestamp is not None:
                        if abs(float(gnss.timestamp) - float(self.last_gnss_fix.timestamp)) > 1e-4:
                            is_new_gnss = True
                    if not is_new_gnss:
                        if (abs(float(gnss.latitude) - float(self.last_gnss_fix.latitude)) > 1e-9 or
                            abs(float(gnss.longitude) - float(self.last_gnss_fix.longitude)) > 1e-9):
                            is_new_gnss = True

        if is_new_gnss:
            self.last_gnss_arrival_time = t
            # Auto-anchor ENU tangent plane to first valid GNSS fix if not already anchored
            if not self.has_gps_anchor:
                self.ref_lat = float(gnss.latitude)
                self.ref_lon = float(gnss.longitude)
                self.fusion = GNSSINSFusion(ref_lat=self.ref_lat, ref_lon=self.ref_lon, dt=self.dt)
                self.has_gps_anchor = True
                self.prev_dr_pos_enu = np.zeros(2)

            gnss_east, gnss_north, _ = self.fusion.latlon_to_enu(gnss.latitude, gnss.longitude)
            gnss_enu = np.array([gnss_east, gnss_north, gnss.altitude])

            # Evaluate GNSS Trust (USP 1)
            gnss_trust_res = self.trust_engine.evaluate_fix(
                gnss_enu=gnss_enu,
                pred_dr_enu=self.fusion.ekf.x[0:3],
                pos_covariance=self.fusion.ekf.P[0:2, 0:2],
                timestamp=t,
                reported_accuracy_m=gnss.accuracy_m,
                inertial_speed_mps=float(np.hypot(self.fusion.ekf.x[3], self.fusion.ekf.x[4])),
            )

            if gnss_trust_res.is_trusted:
                effective_gnss_valid = True
                self.gnss_history.append((gnss.latitude, gnss.longitude))

                # If exiting blackout, handle reacquisition transition
                if self.in_blackout:
                    jump_m = self.reacquisition_smoother.trigger_reacquisition(
                        self.fusion.ekf.x[:2], gnss_enu[:2]
                    )
                    self.blackspot_tracker.on_outage_end(
                        exit_lat=gnss.latitude,
                        exit_lon=gnss.longitude,
                        reacquisition_jump_m=jump_m,
                        timestamp=t,
                    )
                    self.in_blackout = False
                    self.blackout_start_time = None

                self.last_gnss_enu = gnss_enu[:2].copy()

                # Update EKF with GNSS measurement
                r_cov = np.eye(3) * ((gnss.accuracy_m or 3.0) ** 2)
                self.fusion.ekf.update_gnss_pos(gnss_enu, R_cov=r_cov)

                # Optional GNSS Course / Velocity update
                if gnss.heading_deg is not None and gnss.speed_mps is not None and gnss.speed_mps > 1.0:
                    # Convert compass heading to ENU angle psi: psi = 90 - heading
                    psi_gnss = np.deg2rad(90.0 - gnss.heading_deg)
                    self.fusion.ekf.update_heading(psi_gnss, R_yaw=0.05)
            else:
                effective_gnss_valid = False

            self.last_gnss_valid = effective_gnss_valid
            self.last_gnss_trust_res = gnss_trust_res
            self.last_gnss_fix = gnss

        elif gnss is not None and not self.is_gnss_denied_simulated and self.last_gnss_trust_res is not None:
            # Reusing cached GNSS fix on intermediate high-rate IMU frame without redundant EKF measurement updates
            gnss_age = max(0.0, t - self.last_gnss_arrival_time) if self.last_gnss_arrival_time is not None else 0.0
            if gnss_age <= self.gnss_stale_timeout_sec and self.last_gnss_valid:
                effective_gnss_valid = True
                gnss_trust_res = self.last_gnss_trust_res
            else:
                effective_gnss_valid = False
                gnss_trust_res = GNSSTrustResult(
                    status=GNSSTrustStatus.BLACKOUT,
                    trust_score=0.0,
                    innovation_dist_m=0.0,
                    implied_speed_mps=0.0,
                    is_trusted=False,
                    rejection_reason=f"Stale GNSS: no fix for {gnss_age:.1f}s (> {self.gnss_stale_timeout_sec:.1f}s)",
                )
                self.last_gnss_valid = False
        else:
            effective_gnss_valid = False
            gnss_trust_res = GNSSTrustResult(
                status=GNSSTrustStatus.BLACKOUT,
                trust_score=0.0,
                innovation_dist_m=0.0,
                implied_speed_mps=0.0,
                is_trusted=False,
            )
            self.last_gnss_valid = False

        cur_lat, cur_lon = self.fusion.enu_to_latlon(self.fusion.ekf.x[0], self.fusion.ekf.x[1])
        nav_mode = NavigationMode.GNSS_INS_FULL if effective_gnss_valid and gnss_trust_res.trust_score > 0.7 else (NavigationMode.GNSS_DEGRADED if effective_gnss_valid else NavigationMode.DEAD_RECKONING_NHC_AI)

        if not effective_gnss_valid:
            # Inside Blackout or Rejected GNSS
            if not self.in_blackout:
                self.in_blackout = True
                self.blackout_start_time = t
                self.prev_dr_pos_enu = self.fusion.ekf.x[:2].copy()
                self.blackspot_tracker.on_outage_start(cur_lat, cur_lon, timestamp=t)

            # Apply Non-Holonomic Constraints (NHC) with dynamic vehicle profile
            sigma_lat, sigma_vert = self.current_profile.compute_nhc_sigmas(
                yaw_rate=yaw_rate,
                forward_speed=float(np.hypot(self.fusion.ekf.x[3], self.fusion.ekf.x[4])),
            )
            apply_nhc_update(self.fusion.ekf, sigma_lat=sigma_lat, sigma_vert=sigma_vert)

            # Apply AI Velocity pseudo-measurement update
            if self.latest_ai_speed > 0.1:
                self.fusion.ekf.update_velocity(self.latest_ai_speed, R_speed=0.5)

            # Track DR distance and Blackspot analytics
            delta_dr = float(np.linalg.norm(self.fusion.ekf.x[:2] - self.prev_dr_pos_enu))
            self.total_dr_distance += delta_dr
            self.prev_dr_pos_enu = self.fusion.ekf.x[:2].copy()
            
            pos_unc = float(np.sqrt(self.fusion.ekf.P[0, 0] + self.fusion.ekf.P[1, 1]))
            self.blackspot_tracker.on_dr_update(
                lat=cur_lat,
                lon=cur_lon,
                delta_dist_m=delta_dr,
                pos_uncertainty_m=pos_unc,
                timestamp=t,
            )

            nav_mode = NavigationMode.DEAD_RECKONING_NHC_AI if self.latest_ai_speed > 0.1 else NavigationMode.DEAD_RECKONING_PURE

        if is_stationary:
            nav_mode = NavigationMode.STATIONARY_ZUPT

        # 9. Apply C1 Continuous Reacquisition Smoothing if active
        if self.reacquisition_smoother.blend_counter < self.reacquisition_smoother.total_blend_steps and hasattr(self, 'last_gnss_enu'):
            smoothed_enu = self.reacquisition_smoother.apply_smoothing(
                self.fusion.ekf.x[:2], self.last_gnss_enu
            )
            lat_out, lon_out = self.fusion.enu_to_latlon(smoothed_enu[0], smoothed_enu[1])
            nav_mode = NavigationMode.REACQUISITION_SMOOTHING
        else:
            lat_out, lon_out = self.fusion.enu_to_latlon(self.fusion.ekf.x[0], self.fusion.ekf.x[1])

        self.fused_history.append((lat_out, lon_out))
        if self.in_blackout:
            self.dr_history.append((lat_out, lon_out))

        # 10. Check Crash Detection (USP 4)
        cur_fwd_speed = float(
            self.fusion.ekf.x[3] * np.cos(self.fusion.ekf.x[6])
            + self.fusion.ekf.x[4] * np.sin(self.fusion.ekf.x[6])
        )
        crash_alert = self.crash_detector.update(
            acc_3d=acc_raw,
            gyro_3d=gyro_raw,
            current_speed_mps=max(0.0, cur_fwd_speed),
            current_lat=lat_out,
            current_lon=lon_out,
            vehicle_type=self.vehicle_type,
            timestamp=t,
        )

        # 11. Compute Diagnostic Health State (USP 2)
        blackout_elapsed = (t - self.blackout_start_time) if self.blackout_start_time else 0.0
        blend_prog = min(1.0, self.reacquisition_smoother.blend_counter / self.reacquisition_smoother.total_blend_steps)

        diagnostics = self.health_engine.compute_diagnostics(
            nav_mode=nav_mode,
            gnss_trust_score=gnss_trust_res.trust_score,
            gnss_status=gnss_trust_res.status.value,
            ekf_covariance=self.fusion.ekf.P,
            is_stationary=is_stationary,
            is_phone_calibrated=self.aligner.is_calibrated,
            vehicle_type=self.vehicle_type,
            lean_angle_rad=self.current_lean_angle,
            ai_speed_mps=self.latest_ai_speed,
            ekf_forward_speed_mps=cur_fwd_speed,
            total_dr_distance_m=self.total_dr_distance,
            blackout_elapsed_sec=blackout_elapsed,
            reacquisition_blend_progress=blend_prog,
        )

        # Compass heading (0 deg = North, 90 deg = East): heading = 90 - rad2deg(psi)
        psi_deg = float(np.rad2deg(self.fusion.ekf.x[6]))
        compass_heading = (90.0 - psi_deg) % 360.0

        pos_unc_1s = float(np.sqrt(self.fusion.ekf.P[0, 0] + self.fusion.ekf.P[1, 1]))

        return NavigationOutputState(
            timestamp=t,
            latitude=float(lat_out),
            longitude=float(lon_out),
            altitude=float(self.fusion.ekf.x[2]),
            forward_speed_mps=round(cur_fwd_speed, 2),
            velocity_east=round(float(self.fusion.ekf.x[3]), 2),
            velocity_north=round(float(self.fusion.ekf.x[4]), 2),
            heading_deg=round(compass_heading, 1),
            heading_rad=round(float(self.fusion.ekf.x[6]), 3),
            lean_angle_deg=round(float(np.rad2deg(self.current_lean_angle)), 1),
            nav_mode=nav_mode,
            gnss_trust_score=round(gnss_trust_res.trust_score, 2),
            gnss_status=gnss_trust_res.status.value,
            pos_uncertainty_m=round(pos_unc_1s, 2),
            is_in_blackout=bool(self.in_blackout),
            is_stationary=bool(is_stationary),
            is_aligned=bool(self.aligner.is_calibrated),
            diagnostics=diagnostics,
            active_blackspot_id=self.blackspot_tracker.active_outage.id if self.blackspot_tracker.active_outage else None,
            active_crash_alert=crash_alert,
        )

    def reset(self, ref_lat: Optional[float] = None, ref_lon: Optional[float] = None):
        """Reset filter and state."""
        if ref_lat is not None and ref_lon is not None:
            self.ref_lat = float(ref_lat)
            self.ref_lon = float(ref_lon)
            self.has_gps_anchor = True
        else:
            self.has_gps_anchor = False
        self.fusion = GNSSINSFusion(ref_lat=self.ref_lat, ref_lon=self.ref_lon, dt=self.dt)
        self.aligner = PhoneToVehicleAligner()
        self.stationary_detector = StationaryDetector(window_size=10, acc_var_threshold=0.15)
        self.reacquisition_smoother = ReacquisitionSmoother(blend_duration_sec=1.5, dt=self.dt)
        self.trust_engine.reset()
        self.blackspot_tracker = BlackspotTracker(min_duration_sec=2.0)
        self.crash_detector.reset()
        self.ai_resampler.reset()
        self.latest_ai_speed = 0.0
        self.in_blackout = False
        self.blackout_start_time = None
        self.total_dr_distance = 0.0
        self.prev_dr_pos_enu = np.zeros(2)
        self.last_gnss_fix = None
        self.last_gnss_arrival_time = None
        self.last_gnss_trust_res = None
        self.last_gnss_valid = False
        self.gnss_history = []
        self.dr_history = []
        self.fused_history = []
