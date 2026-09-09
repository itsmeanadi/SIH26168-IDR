import numpy as np
import torch
import pytest
from idr.engine.navigation_engine import NavigationEngine, SensorInputFrame, GNSSInputFix

class MockAIModel:
    def __call__(self, x):
        return torch.tensor([[7.0]])

def test_ai_velocity_recovery_from_drift():
    """
    Regression test for the AI velocity innovation gate relaxation.
    Ensures that if the EKF is drifted, the AI velocity can eventually pull the state back.
    """
    # 1. Setup engine
    engine = NavigationEngine(dt=0.1)
    engine.gnss_stale_timeout_sec = 0.0 # Force immediate blackout
    engine.stationary_detector.update = lambda *args, **kwargs: False
    engine.ai_model = MockAIModel() # Use mock model

    # Warm up resampler to be ready
    for i in range(50):
        engine.ai_resampler.add_sample(i * 0.1, np.zeros(6))

    engine.ai_resampler._total_emitted_count = 10 # Trigger inference

    # Mock IMU frame
    def get_imu_frame(t):
        return SensorInputFrame(
            timestamp=t,
            acc_x=0.0, acc_y=0.0, acc_z=9.81,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            orientation_pitch=0.0, orientation_roll=0.0,
            is_absolute=True, orientation_yaw=0.0
        )

    # Use a GNSS fix to anchor the engine initially
    gnss_fix = GNSSInputFix(
        timestamp=0.0, latitude=28.6139, longitude=77.2090,
        altitude=0.0, accuracy_m=1.0, speed_mps=2.0, heading_deg=0.0
    )

    # First frame to initialize
    engine.process_frame(get_imu_frame(0.0), gnss_fix)

    # 2. Artificially drift the EKF velocity state
    engine.fusion.es_ekf.v = np.array([2.0, 0.0, 0.0]) # East velocity = 2.0
    engine.fusion.es_ekf.P[3:6, 3:6] = np.eye(3) * 0.1 # Very confident

    # 3. Simulate blackout frames
    initial_ekf_speed = engine.fusion.es_ekf.v[0]

    for i in range(1, 101):
        engine.process_frame(get_imu_frame(i * 0.1), gnss=None)

    # 4. Verification
    final_ekf_speed = engine.fusion.es_ekf.v[0]
    print(f"Initial EKF Speed: {initial_ekf_speed}, Final EKF Speed: {final_ekf_speed}")

    assert final_ekf_speed > initial_ekf_speed, "EKF velocity should increase towards AI velocity after gate relaxation"
    assert engine.ai_accepted_count > 0, "AI velocity should eventually be accepted"

def test_dr_distance_consistency():
    """
    Verify that DR distance integrates the consistent EKF velocity.
    """
    engine = NavigationEngine(dt=0.1)
    engine.gnss_stale_timeout_sec = 0.0 # Force immediate blackout
    engine.stationary_detector.update = lambda *args, **kwargs: False
    engine.ai_model = MockAIModel()

    # Warm up resampler
    for i in range(50):
        engine.ai_resampler.add_sample(i * 0.1, np.zeros(6))

    engine.ai_resampler._total_emitted_count = 10

    def get_imu_frame(t):
        return SensorInputFrame(
            timestamp=t,
            acc_x=0.0, acc_y=0.0, acc_z=9.81,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            orientation_pitch=0.0, orientation_roll=0.0,
            is_absolute=True, orientation_yaw=0.0
        )

    # Anchor
    gnss_fix = GNSSInputFix(
        timestamp=0.0, latitude=28.6139, longitude=77.2090,
        altitude=0.0, accuracy_m=1.0, speed_mps=5.0, heading_deg=0.0
    )
    engine.process_frame(get_imu_frame(0.0), gnss_fix)

    # Simulate movement
    for i in range(1, 101):
        engine.latest_ai_speed = 5.0
        engine.has_new_ai_estimate = True
        # Ensure AI is accepted
        engine.fusion.es_ekf.P[3:6, 3:6] = np.eye(3) * 10.0

        engine.process_frame(get_imu_frame(i * 0.1), gnss=None)
        engine.has_new_ai_estimate = True

    # Approx dist: 5.0 m/s * 10.0 s = 50.0m
    assert 40.0 < engine.total_dr_distance < 60.0, f"DR distance {engine.total_dr_distance} inconsistent with velocity"
