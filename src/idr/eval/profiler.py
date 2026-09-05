"""Real-Time Latency and Throughput Profiler for Dead Reckoning and Sensor Fusion.

Measures per-step execution time for:
- IMU Mechanization (predict step)
- Kalman Measurement Updates (position, heading, velocity)
- Non-Holonomic Constraints (NHC)
- AI Model Inference (InertialOdomNet, VelocityNet)
- Multi-rate simulation: 200 Hz IMU mechanization + 10 Hz AI inference
"""

import time
from typing import Dict
import numpy as np
import torch

from ..filters.ekf import ExtendedKalmanFilter
from ..filters.nhc import apply_nhc_update
from ..models.inertial_odom import InertialOdomNet


def profile_pipeline(num_iterations: int = 500) -> Dict[str, float]:
    """Profile runtime of individual components and full step pipeline."""
    ekf = ExtendedKalmanFilter(dt=0.1)
    odom_model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    odom_model.eval()

    test_win = torch.randn(1, 6, 50, dtype=torch.float32)
    acc = 0.5
    omega = 0.02
    pos_meas = np.array([10.0, 20.0, 0.0])

    # Warmup
    for _ in range(50):
        ekf.predict(acc, omega)
        apply_nhc_update(ekf)
        with torch.no_grad():
            _ = odom_model(test_win)

    # 1. Profile IMU Predict
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        ekf.predict(acc, omega)
    t_predict_ms = (time.perf_counter() - t0) / num_iterations * 1000.0

    # 2. Profile NHC Update
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        apply_nhc_update(ekf)
    t_nhc_ms = (time.perf_counter() - t0) / num_iterations * 1000.0

    # 3. Profile GNSS Position + Velocity Update
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        ekf.update_gnss_pos(pos_meas)
        ekf.update_gnss_vel(np.array([15.0, 0.0, 0.0]))
    t_update_ms = (time.perf_counter() - t0) / num_iterations * 1000.0

    # 4. Profile AI Inference
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            _ = odom_model(test_win)
        t_ai_ms = (time.perf_counter() - t0) / num_iterations * 1000.0

    t_total_step_ms = t_predict_ms + t_nhc_ms + t_ai_ms

    # 5. Multi-rate simulation (200 Hz IMU path + 10 Hz AI inference)
    # In 1 second: 200 predict steps + 10 AI steps
    t_1sec_load_ms = 200 * t_predict_ms + 10 * t_ai_ms
    cpu_util_pct_200hz = (t_1sec_load_ms / 1000.0) * 100.0

    results = {
        "imu_predict_ms": round(t_predict_ms, 4),
        "nhc_update_ms": round(t_nhc_ms, 4),
        "gnss_update_ms": round(t_update_ms, 4),
        "ai_inference_ms": round(t_ai_ms, 4),
        "total_step_10hz_ms": round(t_total_step_ms, 3),
        "budget_10hz_ms": 100.0,
        "headroom_10hz_pct": round(100.0 - (t_total_step_ms / 100.0 * 100.0), 1),
        "multi_rate_200hz_cpu_load_pct": round(cpu_util_pct_200hz, 2),
    }
    return results


if __name__ == "__main__":
    res = profile_pipeline()
    print("=== LATENCY AND PERFORMANCE PROFILE ===")
    for k, v in res.items():
        print(f"{k:35s}: {v}")
