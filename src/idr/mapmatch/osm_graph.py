"""OpenStreetMap Road Graph Loader with offline caching and ground-truth leakage safety."""

from pathlib import Path
from typing import Optional, Tuple
import logging
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from ..data.provenance import UnsafeEvaluationError

logger = logging.getLogger(__name__)


class OSMGraphLoader:
    """Manages offline loading and caching of OpenStreetMap road networks with strict safety gates."""

    def __init__(self, cache_dir: Path = Path("data/osm")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.graph: Optional[nx.MultiDiGraph] = None

    def load_or_fetch(
        self,
        bbox: tuple,  # (north, south, east, west)
        area_name: str = "drive_network",
    ) -> nx.MultiDiGraph:
        """Load cached GraphML file or fetch from OSM if online."""
        cache_file = self.cache_dir / f"{area_name}.graphml"

        if cache_file.exists():
            logger.info(f"Loading cached OSM graph from {cache_file} (offline mode)")
            try:
                import osmnx as ox
                self.graph = ox.load_graphml(cache_file)
                return self.graph
            except Exception as e:
                logger.warning(f"Failed to load cached graph: {e}. Rebuilding...")

        # If cache doesn't exist, attempt online fetch
        try:
            import osmnx as ox
            logger.info(f"Fetching OSM road network for bbox {bbox}...")
            north, south, east, west = bbox
            self.graph = ox.graph_from_bbox(
                north=north, south=south, east=east, west=west,
                network_type="drive", simplify=True
            )
            ox.save_graphml(self.graph, cache_file)
            logger.info(f"Saved OSM graph to {cache_file}")
            return self.graph
        except Exception as e:
            logger.warning(f"OSM online fetch unavailable ({e}). Generating local road segments from bbox.")
            return self._build_synthetic_graph(bbox)

    def _build_synthetic_graph(self, bbox: tuple) -> nx.MultiDiGraph:
        """Create fallback local road graph when running strictly offline."""
        G = nx.MultiDiGraph()
        north, south, east, west = bbox
        
        # Create a representative road corridor across the bbox
        G.add_node(1, x=west, y=south)
        G.add_node(2, x=east, y=north)
        line = LineString([(west, south), (east, north)])
        G.add_edge(1, 2, 0, geometry=line, length=line.length)
        self.graph = G
        return G

    def build_synthetic_grid(
        self,
        center_xy: Tuple[float, float] = (0.0, 0.0),
        extent_m: float = 1000.0,
        grid_step_m: float = 200.0,
    ) -> nx.MultiDiGraph:
        """Safely create a synthetic Manhattan-style road grid independent of any ground-truth trajectory."""
        G = nx.MultiDiGraph()
        cx, cy = center_xy
        half = extent_m / 2.0
        xs = np.arange(cx - half, cx + half + 1.0, grid_step_m)
        ys = np.arange(cy - half, cy + half + 1.0, grid_step_m)

        node_id = 0
        grid_nodes = {}
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                G.add_node(node_id, x=float(x), y=float(y))
                grid_nodes[(i, j)] = node_id
                node_id += 1

        # Horizontal and vertical edges
        edge_key = 0
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                u = grid_nodes[(i, j)]
                # Connect east
                if i + 1 < len(xs):
                    v = grid_nodes[(i + 1, j)]
                    line = LineString([(xs[i], ys[j]), (xs[i + 1], ys[j])])
                    G.add_edge(u, v, 0, geometry=line, length=float(grid_step_m))
                    G.add_edge(v, u, 0, geometry=line, length=float(grid_step_m))
                # Connect north
                if j + 1 < len(ys):
                    v = grid_nodes[(i, j + 1)]
                    line = LineString([(xs[i], ys[j]), (xs[i], ys[j + 1])])
                    G.add_edge(u, v, 0, geometry=line, length=float(grid_step_m))
                    G.add_edge(v, u, 0, geometry=line, length=float(grid_step_m))

        self.graph = G
        return G

    def build_from_waypoints(
        self,
        waypoints: np.ndarray,
        unsafe_allow_ground_truth_graph: bool = False,
    ) -> nx.MultiDiGraph:
        """Create road graph from coordinate waypoints.
        
        CRITICAL SAFETY GATE:
        Passing ground-truth trajectory waypoints into map-matching creates circular
        information leakage, snapping dead-reckoning errors artificially onto ground truth.
        This is strictly blocked for scientific evaluation unless unsafe_allow_ground_truth_graph=True
        is passed explicitly for development/regression tests.
        """
        if not unsafe_allow_ground_truth_graph:
            raise UnsafeEvaluationError(
                "MAP-MATCHING SAFETY VIOLATION: Constructing an OSM road graph directly from "
                "scenario ground-truth waypoints causes artificial ground-truth leakage into map-matching. "
                "For scientific evaluation, load genuine OSM networks via load_or_fetch() or construct "
                "an independent grid with build_synthetic_grid(). If running legacy development tests, "
                "explicitly set unsafe_allow_ground_truth_graph=True."
            )

        logger.warning(
            "DEVELOPMENT WARNING: Road graph built directly from trajectory waypoints. "
            "This configuration contains ground-truth leakage and is NOT valid for scientific benchmark claims."
        )
        G = nx.MultiDiGraph()
        for i in range(len(waypoints) - 1):
            p1 = (float(waypoints[i, 0]), float(waypoints[i, 1]))
            p2 = (float(waypoints[i + 1, 0]), float(waypoints[i + 1, 1]))
            G.add_node(i, x=p1[0], y=p1[1])
            G.add_node(i + 1, x=p2[0], y=p2[1])
            line = LineString([p1, p2])
            G.add_edge(i, i + 1, 0, geometry=line, length=line.length)
        self.graph = G
        return G
