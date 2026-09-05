"""Vehicle Kinematic Profiles for Ground Vehicles and Two-Wheelers.

Defines kinematic constraint parameters and roll dynamics for:
- 4-Wheeled Passenger Vehicles (Cars, Vans, Trucks)
- 2-Wheeled Vehicles (Motorcycles, Scooters, E-Bikes)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class VehicleProfile:
    """Base kinematic profile for dead reckoning."""
    vehicle_type: str = "car"
    sigma_lat: float = 0.05       # Lateral constraint noise (m/s)
    sigma_vert: float = 0.05      # Vertical constraint noise (m/s)
    max_roll_deg: float = 5.0     # Maximum expected roll (degrees)
    has_roll_coupling: bool = False

    def compute_nhc_sigmas(self, yaw_rate: float, forward_speed: float) -> Tuple[float, float]:
        """Dynamically adapt NHC standard deviation based on vehicle state."""
        return self.sigma_lat, self.sigma_vert


class CarProfile(VehicleProfile):
    """Car Profile: Rigid chassis, negligible roll angle, tight non-holonomic constraints."""

    def __init__(self, sigma_lat: float = 0.05, sigma_vert: float = 0.05):
        super().__init__(
            vehicle_type="car",
            sigma_lat=sigma_lat,
            sigma_vert=sigma_vert,
            max_roll_deg=4.0,
            has_roll_coupling=False,
        )


class TwoWheelerProfile(VehicleProfile):
    """Two-Wheeler Profile (Motorcycle / Scooter):

    In two-wheelers, the vehicle rolls/leans into turns to balance centripetal acceleration:
        tan(phi) ≈ a_centripetal / g = (v * omega) / g

    Because of leaning, the body lateral axis experiences gravity projection:
        a_meas,lat = v * omega * cos(phi) - g * sin(phi) ≈ 0 (when balanced)

    Consequently, the lateral velocity constraint is softer (sigma_lat = 0.35 m/s),
    and roll angle can be estimated directly from yaw rate and forward speed.
    """

    def __init__(self, sigma_lat: float = 0.35, sigma_vert: float = 0.10):
        super().__init__(
            vehicle_type="two_wheeler",
            sigma_lat=sigma_lat,
            sigma_vert=sigma_vert,
            max_roll_deg=45.0,
            has_roll_coupling=True,
        )

    def estimate_roll_angle(self, forward_speed: float, yaw_rate: float) -> float:
        """Estimate roll/lean angle phi in radians from steady-state turn kinematics."""
        g = 9.80665
        a_centripetal = forward_speed * yaw_rate
        # phi = arctan(v * omega / g), clamped to [-45 deg, +45 deg]
        phi = np.arctan2(a_centripetal, g)
        max_rad = np.deg2rad(self.max_roll_deg)
        return float(np.clip(phi, -max_rad, max_rad))

    def compute_nhc_sigmas(self, yaw_rate: float, forward_speed: float) -> Tuple[float, float]:
        """Scale lateral tolerance dynamically during sharp turns."""
        lean_angle = abs(self.estimate_roll_angle(forward_speed, yaw_rate))
        # When leaning hard, increase lateral tolerance to avoid over-constraining
        scaled_sigma_lat = self.sigma_lat * (1.0 + 2.0 * np.sin(lean_angle))
        return float(scaled_sigma_lat), self.sigma_vert
