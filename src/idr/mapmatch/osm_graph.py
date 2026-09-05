"""OpenStreetMap Road Graph Loader with offline caching."""

from pathlib import Path
from typing import Optional
import logging
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)

class OSMGraphLoader:
    """Manages offline loading and caching of OpenStreetMap road networks."""

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
            logger.warning(f"OSM online fetch unavailable ({e}). Generating local road segments from trajectory.")
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

    def build_from_waypoints(self, waypoints: np.ndarray) -> nx.MultiDiGraph:
        """Create road graph from local coordinate waypoints (e.g. ENU road centerline)."""
        G = nx.MultiDiGraph()
        for i in range(len(waypoints) - 1):
            p1 = (float(waypoints[i, 0]), float(waypoints[i, 1]))
            p2 = (float(waypoints[i+1, 0]), float(waypoints[i+1, 1]))
            G.add_node(i, x=p1[0], y=p1[1])
            G.add_node(i+1, x=p2[0], y=p2[1])
            line = LineString([p1, p2])
            G.add_edge(i, i+1, 0, geometry=line, length=line.length)
        self.graph = G
        return G
