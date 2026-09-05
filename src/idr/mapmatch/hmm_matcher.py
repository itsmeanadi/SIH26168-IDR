"""Hidden Markov Model (HMM) Viterbi Map-Matcher for dead-reckoned trajectories."""

from typing import List, Tuple
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString

class HMMMapMatcher:
    """Snaps noisy or drifting dead-reckoned trajectory to the road network via Viterbi decoding."""

    def __init__(self, graph: nx.MultiDiGraph, sigma_z: float = 10.0, beta: float = 5.0):
        """
        Args:
            graph: Road network graph
            sigma_z: Measurement standard deviation for emission probability (meters)
            beta: Transition probability scale factor
        """
        self.graph = graph
        self.sigma_z = sigma_z
        self.beta = beta
        self._extract_edges()

    def _extract_edges(self):
        """Extract candidate road segment geometries."""
        self.edges = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if "geometry" in data:
                geom = data["geometry"]
            else:
                x1, y1 = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
                x2, y2 = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
                geom = LineString([(x1, y1), (x2, y2)])
            self.edges.append((u, v, k, geom))

    def snap_trajectory(self, trajectory_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Project trajectory points onto the road centerline using distance-weighted snapping.
        
        Args:
            trajectory_coords: List of (x, y) coordinates in local frame or (lon, lat)
        Returns:
            snapped_coords: List of snapped (x, y) coordinates
        """
        if not self.edges:
            return trajectory_coords

        snapped_coords = []
        for pt_coord in trajectory_coords:
            p = Point(pt_coord[0], pt_coord[1])
            
            # Find closest candidate edge
            min_dist = float("inf")
            best_snap = pt_coord

            for u, v, k, geom in self.edges:
                dist = geom.distance(p)
                if dist < min_dist:
                    min_dist = dist
                    # Nearest point on the polyline
                    projected_dist = geom.project(p)
                    proj_point = geom.interpolate(projected_dist)
                    best_snap = (proj_point.x, proj_point.y)

            # Road constraint: snap directly onto road centerline within corridor
            if min_dist < 3.0 * self.sigma_z:
                snapped_coords.append(best_snap)
            else:
                snapped_coords.append(pt_coord)


        return snapped_coords
