"""Two-Wheeler Dynamics Dead Reckoning Evaluation.

Evaluates performance of CarProfile vs TwoWheelerProfile under motorcycle lean dynamics.
Demonstrates how roll-compensated NHC adapts lateral constraint tolerances during cornering.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from ..config import set_seed
from ..filters.vehicle_profiles import CarProfile, TwoWheelerProfile
from ..filters.fusion import GNSSINSFusion
from ..eval.metrics import compute_navigation_metrics


def synthesize_twowheeler_trajectory(N: int = 1000, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize motorcycle trajectory with sharp cornering and roll-lean dynamics."""
    set_seed(42)
    t = np.arange(N) * dt
    # Motorcycle speed: 12 m/s (~43 km/h) with acceleration/braking
    speed = 12.0 + 3.0 * np.sin(0.04 * t)
    # Series of S-curves with alternating yaw rates reaching 0.12 rad/s (~7 deg/s)
    yaw_rate = 0.08 * np.sin(0.06 * t) + 0.04 * np.cos(0.03 * t)
    yaw = np.cumsum(yaw_rate * dt)

    gt_x = np.cumsum(speed * np.cos(yaw) * dt)
    gt_y = np.cumsum(speed * np.sin(yaw) * dt)
    gt_enu = np.column_stack([gt_x, gt_y])

    # Motorcycle lean angle: phi = arctan(v * omega / g)
    g = 9.80665
    phi_lean = np.arctan2(speed * yaw_rate, g)

    # Smartphone IMU mounted on motorcycle handlebars (experiences lean + centripetal forces)
    imu_data = np.zeros((N, 6))
    fwd_acc = np.gradient(speed, dt)
    lat_acc = speed * yaw_rate

    # In leaned body frame:
    imu_data[:, 0] = fwd_acc + 0.12 + np.random.randn(N) * 0.2
    # Lateral accelerometer measures residual acceleration + gravity component
    imu_data[:, 1] = lat_acc * np.cos(phi_lean) - g * np.sin(phi_lean) + np.random.randn(N) * 0.15
    imu_data[:, 5] = yaw_rate + 0.012 + np.random.randn(N) * 0.025

    return imu_data, gt_enu, phi_lean


def evaluate_twowheeler_performance() -> Dict[str, Dict[str, float]]:
    N = 1000
    dt = 0.1
    imu_data, gt_enu, phi_lean = synthesize_twowheeler_trajectory(N, dt)

    blackout_start = 250
    blackout_end = 850  # 60 seconds outage = ~720m
    ref_lat, ref_lon = 52.4068, -1.5197

    profiles = {
        "Car Rigid NHC (Fixed sigma_lat=0.05m/s)": CarProfile(sigma_lat=0.05),
        "Two-Wheeler Adaptive NHC (Lean-Aware)": TwoWheelerProfile(sigma_lat=0.35),
    }

    results = {}
    for name, profile in profiles.items():
        fusion = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=dt)
        fusion.ekf.x[:2] = gt_enu[0]
        init_v = (gt_enu[1] - gt_enu[0]) / dt
        fusion.ekf.x[3:5] = init_v
        fusion.ekf.x[6] = np.arctan2(init_v[1], init_v[0])

        pred_enu = np.zeros((N, 2))
        prev_enu = None

        for i in range(N):
            is_denied = (blackout_start <= i < blackout_end)
            fwd_acc = float(imu_data[i, 0])
            yaw_rate = float(imu_data[i, 5])

            fusion.ekf.predict(fwd_acc, yaw_rate)

            if not is_denied:
                east, north = gt_enu[i, 0], gt_enu[i, 1]
                fusion.ekf.update_gnss_pos(np.array([east, north, 0.0]))
                if prev_enu is not None:
                    de = east - prev_enu[0]
                    dn = north - prev_enu[1]
                    if np.hypot(de, dn) > 0.1:
                        cog = np.arctan2(dn, de)
                        fusion.ekf.update_heading(cog, R_yaw=0.03)
                        fusion.ekf.update_gnss_vel(np.array([de/dt, dn/dt, 0.0]))
                prev_enu = (east, north)
            else:
                prev_enu = None
                # Adapt NHC noise based on vehicle profile
                v_est = float(np.hypot(fusion.ekf.x[3], fusion.ekf.x[4]))
                sig_lat, sig_vert = profile.compute_nhc_sigmas(yaw_rate, v_est)
                
                # Apply NHC with profile sigmas
                from ..filters.nhc import apply_nhc_update
                apply_nhc_update(fusion.ekf, sigma_lat=sig_lat, sigma_vert=sig_vert)
                fusion.ekf.update_velocity(v_est, R_speed=0.5)

            pred_enu[i] = fusion.ekf.x[:2]

        m = compute_navigation_metrics(pred_enu[blackout_start:blackout_end], gt_enu[blackout_start:blackout_end])
        results[name] = {
            "outage_distance_m": round(m.total_distance_m, 2),
            "final_drift_m": round(m.final_drift_m, 2),
            "drift_pct": round(m.drift_percent, 2),
            "cep50_m": round(m.cep_50_m, 2),
            "rmse_m": round(m.rmse_position_m, 2),
            "max_lean_angle_deg": round(float(np.rad2deg(np.max(np.abs(phi_lean)))), 1),
        }

    return results


if __name__ == "__main__":
    res = evaluate_twowheeler_performance()
    print("=== TWO-WHEELER KINEMATIC EVALUATION RESULTS ===")
    for k, v in res.items():
        print(f"{k:45s} | Drift: {v['final_drift_m']:5.2f}m ({v['drift_pct']:5.2f}%) | CEP50: {v['cep50_m']:5.2f}m")
