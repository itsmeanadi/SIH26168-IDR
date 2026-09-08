"""Hidden Markov Model (HMM) Viterbi Map-Matcher and Geometric Centerline Snapper.

Implements:
1. True HMM Viterbi Map Matcher (Newson & Krumm formulation):
   - Gaussian orthogonal distance emission probability
   - Exponential path difference transition probability
   - Global Viterbi dynamic programming trellis backpointer
2. Fast Geometric Road Snapper (orthogonal nearest-segment projection)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString


class GeometricRoadSnapper:
    """Fast nearest-segment centerline projector for local corridor snapping."""

    def __init__(self, graph: nx.MultiDiGraph, sigma_z: float = 15.0):
        self.graph = graph
        self.sigma_z = float(sigma_z)
        self._extract_edges()

    def _extract_edges(self):
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
        if not self.edges:
            return trajectory_coords

        snapped_coords = []
        for pt_coord in trajectory_coords:
            p = Point(pt_coord[0], pt_coord[1])
            min_dist = float("inf")
            best_snap = pt_coord

            for u, v, k, geom in self.edges:
                dist = geom.distance(p)
                if dist < min_dist:
                    min_dist = dist
                    projected_dist = geom.project(p)
                    proj_point = geom.interpolate(projected_dist)
                    best_snap = (float(proj_point.x), float(proj_point.y))

            if min_dist < 3.0 * self.sigma_z:
                snapped_coords.append(best_snap)
            else:
                snapped_coords.append(pt_coord)

        return snapped_coords


class HMMMapMatcher:
    """Genuine Hidden Markov Model (HMM) Viterbi Map-Matcher for trajectory decoding."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        sigma_z: float = 15.0,  # Measurement emission noise (m)
        beta: float = 10.0,      # Transition length scale factor (m)
        max_candidates: int = 5,
    ):
        self.graph = graph
        self.sigma_z = float(sigma_z)
        self.beta = float(beta)
        self.max_candidates = int(max_candidates)
        self._extract_edges()

    def _extract_edges(self):
        self.edges = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if "geometry" in data:
                geom = data["geometry"]
            else:
                x1, y1 = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
                x2, y2 = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
                geom = LineString([(x1, y1), (x2, y2)])
            length = float(data.get("length", geom.length))
            self.edges.append((u, v, k, geom, length))

    def _emission_log_prob(self, dist_m: float) -> float:
        """Gaussian emission log-likelihood ln P(z | c)."""
        return -0.5 * (dist_m / self.sigma_z)**2 - np.log(self.sigma_z * np.sqrt(2.0 * np.pi))

    def _transition_log_prob(self, d_meas: float, d_route: float, is_same_edge: bool = True) -> float:
        """Exponential transition log-likelihood ln P(c_t | c_{t-1})."""
        delta_d = abs(d_route - d_meas)
        # Topology penalty for hopping between disconnected/different road segments
        if not is_same_edge:
            delta_d += 2.0 * self.beta
        return -(delta_d / self.beta) - np.log(self.beta)

    def snap_trajectory(self, trajectory_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Decode most likely sequence of road states via Viterbi Trellis."""
        T = len(trajectory_coords)
        if T == 0 or not self.edges:
            return trajectory_coords

        # 1. Candidate extraction for all time steps
        step_candidates: List[List[Tuple[int, Tuple[float, float], float, float]]] = []
        for t, pt in enumerate(trajectory_coords):
            p = Point(pt[0], pt[1])
            cands = []
            for edge_idx, (u, v, k, geom, length) in enumerate(self.edges):
                dist = geom.distance(p)
                if dist <= 3.5 * self.sigma_z:
                    proj_dist = geom.project(p)
                    proj_pt = geom.interpolate(proj_dist)
                    cands.append((edge_idx, (float(proj_pt.x), float(proj_pt.y)), float(dist), float(proj_dist)))
            
            # Sort by distance and take top K candidates
            cands.sort(key=lambda x: x[2])
            cands = cands[:self.max_candidates]
            
            # Fallback if no candidate in range: keep original measurement
            if not cands:
                cands.append((-1, (float(pt[0]), float(pt[1])), 0.0, 0.0))
            step_candidates.append(cands)

        # 2. Viterbi Forward Trellis
        V: List[np.ndarray] = []
        backpointers: List[np.ndarray] = []

        # Initialization at t=0
        init_V = np.array([self._emission_log_prob(c[2]) for c in step_candidates[0]], dtype=np.float64)
        V.append(init_V)

        for t in range(1, T):
            prev_cands = step_candidates[t - 1]
            curr_cands = step_candidates[t]
            curr_V = np.zeros(len(curr_cands), dtype=np.float64)
            bp = np.zeros(len(curr_cands), dtype=np.int32)

            # Measurement Euclidean distance
            d_meas = float(np.hypot(
                trajectory_coords[t][0] - trajectory_coords[t - 1][0],
                trajectory_coords[t][1] - trajectory_coords[t - 1][1]
            ))

            for j, (curr_edge, curr_pt, curr_dist, curr_proj) in enumerate(curr_cands):
                emission = self._emission_log_prob(curr_dist)
                max_score = -float("inf")
                best_i = 0

                for i, (prev_edge, prev_pt, prev_dist, prev_proj) in enumerate(prev_cands):
                    # Route distance: along edge if on same edge, Euclidean otherwise
                    if curr_edge == prev_edge and curr_edge != -1:
                        d_route = abs(curr_proj - prev_proj)
                        is_same = True
                    else:
                        d_route = float(np.hypot(curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]))
                        is_same = (curr_edge == prev_edge)

                    trans = self._transition_log_prob(d_meas, d_route, is_same_edge=is_same)
                    score = V[t - 1][i] + trans + emission

                    if score > max_score:
                        max_score = score
                        best_i = i

                curr_V[j] = max_score
                bp[j] = best_i

            V.append(curr_V)
            backpointers.append(bp)

        # 3. Viterbi Backtracking
        best_last_idx = int(np.argmax(V[-1]))
        snapped_path = [step_candidates[-1][best_last_idx][1]]
        curr_idx = best_last_idx

        for t in range(T - 2, -1, -1):
            curr_idx = backpointers[t][curr_idx]
            snapped_path.append(step_candidates[t][curr_idx][1])

        snapped_path.reverse()
        return snapped_path
