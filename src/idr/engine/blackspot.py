"""GNSS Blackspot / Outage Tracker & Analytics (USP 3).

Tracks GNSS outage zones, entry/exit coordinates, blackout duration,
accumulated dead-reckoning distance, and estimated drift upon reacquisition.
Generates GeoJSON layers for map visualization.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BlackspotRecord:
    id: str
    start_timestamp: float
    end_timestamp: Optional[float]
    duration_sec: float
    entry_lat: float
    entry_lon: float
    exit_lat: Optional[float]
    exit_lon: Optional[float]
    dr_distance_m: float
    max_drift_uncertainty_m: float
    final_reacquisition_error_m: Optional[float]
    severity: str  # "MILD", "MODERATE", "SEVERE"
    trajectory_points: List[Tuple[float, float]]


class BlackspotTracker:
    """Tracks and logs GNSS outage blackspots for infrastructure reliability maps."""

    def __init__(self, min_duration_sec: float = 2.0):
        self.min_duration_sec = min_duration_sec
        self.history: List[BlackspotRecord] = []
        self.active_outage: Optional[BlackspotRecord] = None
        self._outage_counter = 0

    def on_outage_start(
        self,
        lat: float,
        lon: float,
        timestamp: Optional[float] = None,
    ):
        """Called when GNSS signal is lost or rejected."""
        if self.active_outage is not None:
            return  # Already in an outage

        t = timestamp if timestamp is not None else time.time()
        self._outage_counter += 1
        self.active_outage = BlackspotRecord(
            id=f"BS-{self._outage_counter:04d}",
            start_timestamp=t,
            end_timestamp=None,
            duration_sec=0.0,
            entry_lat=float(lat),
            entry_lon=float(lon),
            exit_lat=None,
            exit_lon=None,
            dr_distance_m=0.0,
            max_drift_uncertainty_m=0.0,
            final_reacquisition_error_m=None,
            severity="MILD",
            trajectory_points=[(float(lat), float(lon))],
        )

    def on_dr_update(
        self,
        lat: float,
        lon: float,
        delta_dist_m: float,
        pos_uncertainty_m: float,
        timestamp: Optional[float] = None,
    ):
        """Track dead-reckoned progress inside the active blackout."""
        if self.active_outage is None:
            return

        t = timestamp if timestamp is not None else time.time()
        self.active_outage.duration_sec = max(0.0, t - self.active_outage.start_timestamp)
        self.active_outage.dr_distance_m += float(delta_dist_m)
        self.active_outage.max_drift_uncertainty_m = max(
            self.active_outage.max_drift_uncertainty_m, float(pos_uncertainty_m)
        )
        self.active_outage.trajectory_points.append((float(lat), float(lon)))

        # Update severity
        if self.active_outage.duration_sec > 30.0 or self.active_outage.dr_distance_m > 300.0:
            self.active_outage.severity = "SEVERE"
        elif self.active_outage.duration_sec > 10.0 or self.active_outage.dr_distance_m > 80.0:
            self.active_outage.severity = "MODERATE"
        else:
            self.active_outage.severity = "MILD"

    def on_outage_end(
        self,
        exit_lat: float,
        exit_lon: float,
        reacquisition_jump_m: float,
        timestamp: Optional[float] = None,
    ) -> Optional[BlackspotRecord]:
        """Called upon successful GNSS reacquisition."""
        if self.active_outage is None:
            return None

        t = timestamp if timestamp is not None else time.time()
        self.active_outage.end_timestamp = t
        self.active_outage.duration_sec = max(0.0, t - self.active_outage.start_timestamp)
        self.active_outage.exit_lat = float(exit_lat)
        self.active_outage.exit_lon = float(exit_lon)
        self.active_outage.final_reacquisition_error_m = float(reacquisition_jump_m)

        record = self.active_outage
        self.active_outage = None

        if record.duration_sec >= self.min_duration_sec or record.dr_distance_m >= 10.0:
            self.history.append(record)
            return record
        return None

    def get_all_records(self) -> List[Dict[str, Any]]:
        return [asdict(rec) for rec in self.history]

    def to_geojson(self) -> Dict[str, Any]:
        """Export blackspots as GeoJSON FeatureCollection for map rendering."""
        features = []
        for rec in self.history:
            # Polyline of dead-reckoned path inside blackspot
            coords = [[lon, lat] for lat, lon in rec.trajectory_points]
            if len(coords) < 2 and rec.exit_lon is not None and rec.exit_lat is not None:
                coords = [[rec.entry_lon, rec.entry_lat], [rec.exit_lon, rec.exit_lat]]

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "id": rec.id,
                    "duration_sec": round(rec.duration_sec, 1),
                    "dr_distance_m": round(rec.dr_distance_m, 1),
                    "max_drift_uncertainty_m": round(rec.max_drift_uncertainty_m, 1),
                    "reacquisition_jump_m": round(rec.final_reacquisition_error_m or 0.0, 1),
                    "severity": rec.severity,
                    "entry": [rec.entry_lat, rec.entry_lon],
                    "exit": [rec.exit_lat, rec.exit_lon] if rec.exit_lat is not None else None,
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }
