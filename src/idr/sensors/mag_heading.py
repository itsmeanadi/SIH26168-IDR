"""Tilt-Compensated Magnetic Heading Calculation.

Computes horizontal magnetic heading in East-North-Up (ENU) navigation frame convention
from 3D magnetic field vector and current attitude (roll and pitch).
"""

import numpy as np


def compute_tilt_compensated_heading(
    mag_vehicle: np.ndarray,
    roll_rad: float,
    pitch_rad: float,
    declination_rad: float = 0.0,
) -> float:
    """Calculate tilt-compensated magnetic heading in ENU frame convention.
    
    ENU Frame Convention:
    - 0 deg (0 rad)   : East
    - 90 deg (pi/2)   : North
    - 180 deg (pi)    : West
    - -90 deg (-pi/2) : South
    
    When Vehicle Body-X points North, yaw = pi/2.
    When Vehicle Body-X points East, yaw = 0.
    
    Args:
        mag_vehicle: (3,) Calibrated magnetic field vector in vehicle body frame [mx, my, mz].
        roll_rad: Estimated body roll angle phi in radians.
        pitch_rad: Estimated body pitch angle theta in radians.
        declination_rad: Local magnetic declination angle in radians (default 0.0).
        
    Returns:
        psi_enu: Tilt-compensated heading angle in ENU frame [-pi, pi].
    """
    mx = float(mag_vehicle[0])
    my = float(mag_vehicle[1])
    mz = float(mag_vehicle[2])

    cos_roll = np.cos(roll_rad)
    sin_roll = np.sin(roll_rad)
    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)

    # 1. Project 3D magnetometer vector onto horizontal plane (level frame)
    mx_h = mx * cos_pitch + my * sin_roll * sin_pitch + mz * cos_roll * sin_pitch
    my_h = my * cos_roll - mz * sin_roll

    # 2. Compute ENU heading angle:
    # In ENU: Magnetic North is +Y, Magnetic East is +X.
    # When Vehicle points North: mx_h > 0, my_h = 0 -> psi = arctan2(mx_h, -my_h) = pi/2 (North)
    # When Vehicle points East:  mx_h = 0, my_h = -B -> psi = arctan2(0, B) = 0 (East)
    # When Vehicle points West:  mx_h = 0, my_h = +B -> psi = arctan2(0, -B) = pi (West)
    # When Vehicle points South: mx_h < 0, my_h = 0 -> psi = arctan2(mx_h, 0) = -pi/2 (South)
    psi_enu = np.arctan2(mx_h, -my_h) + declination_rad

    # 3. Wrap to [-pi, pi]
    return float(np.arctan2(np.sin(psi_enu), np.cos(psi_enu)))
