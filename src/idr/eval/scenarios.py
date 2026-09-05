"""Multi-Scenario Generator for Statistical Dead Reckoning Evaluation.

Generates ≥200 distinct GNSS outage segments across varied:
1. Trajectory segments (7 real drives)
2. Outage lengths (100 m, 500 m, 1000 m, 1500 m)
3. Blackout onset locations (early, mid, late drive)
4. Dynamic regimes (straight highway, highway curves, speed transitions, high noise)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from ..io.loader import load_drive_pair


@dataclass
class OutageScenario:
    """Encapsulates a single blackout evaluation segment."""
    scenario_id: str
    drive_id: str
    scenario_type: str  # 'straight', 'curve', 'speed_variation', 'high_noise'
    blackout_start: int
    blackout_end: int
    blackout_length_m: float
    imu_data: np.ndarray        # (N, 6)
    gps_latlon: np.ndarray      # (N, 2)
    gt_enu: np.ndarray          # (N, 2)
    ref_lat: float
    ref_lon: float


def build_scenario_library(
    raw_dir: Path,
    target_count: int = 200,
    outage_steps_list: Tuple[int, ...] = (100, 250, 400, 600),
) -> List[OutageScenario]:
    """Generate ≥200 realistic outage evaluation scenarios from IO-VNBD drives."""
    raw_dir = Path(raw_dir)
    drives = ["M", "S", "Vf", "Vta", "Vtb", "Vw", "Y1"]
    loaded_drives = {}

    R_earth = 6378137.0

    # Load all drives
    for d in drives:
        drive_path = raw_dir / d
        if not drive_path.exists():
            continue
        try:
            drive_data = load_drive_pair(drive_path, d)
            phone_imu, phone_gps, v_speed, t = drive_data.get_synced_data()
            ref_lat, ref_lon = phone_gps[0, 0], phone_gps[0, 1]

            # Convert to local ENU
            lat_rad = np.deg2rad(ref_lat)
            d_lat = np.deg2rad(phone_gps[:, 0] - ref_lat)
            d_lon = np.deg2rad(phone_gps[:, 1] - ref_lon)
            north = d_lat * R_earth
            east = d_lon * R_earth * np.cos(lat_rad)
            gt_enu = np.column_stack([east, north])

            loaded_drives[d] = {
                "imu": phone_imu,
                "gps": phone_gps[:, :2],
                "enu": gt_enu,
                "speed": v_speed,
                "ref_lat": ref_lat,
                "ref_lon": ref_lon,
            }
        except Exception:
            continue

    scenarios = []
    scenario_idx = 0

    # Generate diverse scenarios by sliding outage windows
    for d_name, d_dict in loaded_drives.items():
        N = len(d_dict["imu"])
        imu_base = d_dict["imu"]
        enu_base = d_dict["enu"]
        gps_base = d_dict["gps"]

        for outage_len in outage_steps_list:
            # Step size along drive
            step_stride = 50
            for start in range(50, N - outage_len - 20, step_stride):
                end = start + outage_len
                # Calculate distance traversed during outage
                dist_traversed = float(np.sum(np.hypot(np.diff(enu_base[start:end, 0]), np.diff(enu_base[start:end, 1]))))
                if dist_traversed < 30.0:
                    continue

                # Classify dynamics
                segment_yaw_rates = imu_base[start:end, 5]
                mean_yaw_rate = float(np.mean(np.abs(segment_yaw_rates)))
                segment_speeds = np.hypot(np.diff(enu_base[start:end, 0]), np.diff(enu_base[start:end, 1])) / 0.1
                speed_std = float(np.std(segment_speeds)) if len(segment_speeds) > 0 else 0.0

                if mean_yaw_rate > 0.02:
                    stype = "motorway_curve"
                elif speed_std > 2.0:
                    stype = "speed_variation"
                else:
                    stype = "motorway_straight"

                scenario = OutageScenario(
                    scenario_id=f"SCEN_{scenario_idx:04d}_{d_name}_L{int(dist_traversed)}m",
                    drive_id=d_name,
                    scenario_type=stype,
                    blackout_start=start,
                    blackout_end=end,
                    blackout_length_m=dist_traversed,
                    imu_data=imu_base.copy(),
                    gps_latlon=gps_base.copy(),
                    gt_enu=enu_base.copy(),
                    ref_lat=d_dict["ref_lat"],
                    ref_lon=d_dict["ref_lon"],
                )
                scenarios.append(scenario)
                scenario_idx += 1

                # Generate a high-noise / severe-bias perturbation scenario
                if len(scenarios) < target_count:
                    perturbed_imu = imu_base.copy()
                    perturbed_imu[:, 0] += np.random.uniform(-0.3, 0.3)  # extra accel bias
                    perturbed_imu[:, 5] += np.random.uniform(-0.015, 0.015)  # extra gyro bias
                    perturbed_imu += np.random.randn(*perturbed_imu.shape) * 0.05  # sensor noise

                    scen_noise = OutageScenario(
                        scenario_id=f"SCEN_{scenario_idx:04d}_{d_name}_noise_L{int(dist_traversed)}m",
                        drive_id=d_name,
                        scenario_type="high_noise_canyon",
                        blackout_start=start,
                        blackout_end=end,
                        blackout_length_m=dist_traversed,
                        imu_data=perturbed_imu,
                        gps_latlon=gps_base.copy(),
                        gt_enu=enu_base.copy(),
                        ref_lat=d_dict["ref_lat"],
                        ref_lon=d_dict["ref_lon"],
                    )
                    scenarios.append(scen_noise)
                    scenario_idx += 1

    return scenarios[:max(target_count, len(scenarios))]
