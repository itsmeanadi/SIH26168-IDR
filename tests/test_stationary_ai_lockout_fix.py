"""Regression test for AI-induced ZUPT lockout.
Ensures that a false AI speed estimate does not prevent the system from
latching stationary when the IMU is decisively quiet.
"""

import numpy as np
import pytest
from idr.engine.navigation_engine import (
    NavigationEngine,
    SensorInputFrame,
)

def test_ai_speed_does_not_block_stationary_latch():
    """
    Verify that if the AI predicts movement but the IMU is quiet,
    the system still latches stationary and applies ZUPT.
    """
    engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")

    # Force a 'false positive' AI speed that would have previously blocked ZUPT (threshold is 0.8)
    engine.latest_ai_speed = 1.5
    engine.ai_accepted_count = 1

    # 30 seconds of perfectly quiet IMU
    for i in range(300):
        t = i * 0.1
        imu = SensorInputFrame(
            timestamp=t,
            acc_x=0.0,
            acc_y=0.0,
            acc_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
        )
        out = engine.process_frame(imu, None)

    # After the persistence window (2 frames), it should be stationary
    assert out.is_stationary is True, "System failed to latch stationary despite quiet IMU"

    # DR distance should be near zero because is_stationary is True
    assert engine.total_dr_distance < 0.01, f"DR distance leaked: {engine.total_dr_distance}m"

    # EKF velocity should have been zeroed by ZUPT
    assert np.linalg.norm(engine.fusion.velocity_enu) < 0.01, "Velocity not zeroed by ZUPT"

if __name__ == "__main__":
    pytest.main([__file__])
