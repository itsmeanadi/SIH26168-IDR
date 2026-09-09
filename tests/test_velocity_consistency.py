import numpy as np
import pytest
from idr.filters.es_ekf import ErrorStateKalmanFilter, ESEKFConfig
from idr.filters.ekf import ExtendedKalmanFilter
from idr.engine.navigation_engine import NavigationEngine, SensorInputFrame, GNSSInputFix

def test_es_ekf_ai_velocity_3sigma_gating():
    """
    Verify ErrorStateKalmanFilter innovation gating strictly enforces 3-sigma gate.
    """
    cfg = ESEKFConfig()
    es_ekf = ErrorStateKalmanFilter(config=cfg)
    es_ekf.set_state(pos=np.zeros(3), vel=np.array([5.0, 0.0, 0.0]))

    # Small innovation (5.5 m/s vs 5.0 m/s) -> accepted
    accepted_norm, metrics_norm = es_ekf.update_ai_velocity(
        speed_mps=5.5,
        sigma_v=1.0,
        max_innovation_sigma=3.0,
    )
    assert accepted_norm is True
    assert metrics_norm["accepted"] is True

    # Extreme unphysical innovation (45.0 m/s vs ~5.0 m/s) -> rejected by 3-sigma gate
    accepted_outlier, metrics_outlier = es_ekf.update_ai_velocity(
        speed_mps=45.0,
        sigma_v=1.0,
        max_innovation_sigma=3.0,
    )
    assert accepted_outlier is False
    assert metrics_outlier["accepted"] is False
    assert "exceeded gate" in metrics_outlier.get("rejection_reason", "").lower()

def test_dr_distance_integration():
    """
    Verify that DR distance correctly accumulates displacement during dead reckoning.
    """
    engine = NavigationEngine(dt=0.1)
    engine.gnss_stale_timeout_sec = 0.0
    engine.stationary_detector.update = lambda *args, **kwargs: False

    def get_imu_frame(t):
        return SensorInputFrame(
            timestamp=t,
            acc_x=0.0, acc_y=0.0, acc_z=9.81,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            orientation_pitch=0.0, orientation_roll=0.0,
            is_absolute=True, orientation_yaw=0.0
        )

    gnss_fix = GNSSInputFix(
        timestamp=0.0, latitude=28.6139, longitude=77.2090,
        altitude=0.0, accuracy_m=1.0, speed_mps=5.0, heading_deg=0.0
    )
    engine.process_frame(get_imu_frame(0.0), gnss_fix)

    # Set initial velocity to 5.0 m/s
    engine.fusion.es_ekf.v = np.array([5.0, 0.0, 0.0])

    # Propagate 50 steps (5.0 seconds at 5 m/s => ~25 meters)
    for i in range(1, 51):
        engine.process_frame(get_imu_frame(i * 0.1), gnss=None)

    # Expected distance ~ 25 m
    assert 20.0 <= engine.total_dr_distance <= 30.0, f"DR distance {engine.total_dr_distance:.2f}m expected ~25m"
