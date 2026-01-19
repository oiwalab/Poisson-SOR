"""GDS file loading and polygon rasterization for electrode masks

Loads electrode patterns from GDSII files and converts them to simulation grid masks.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy imports to avoid startup overhead
_gdspy = None
_shapely = None


def _ensure_gdspy():
    """Lazy import of gdspy"""
    global _gdspy
    if _gdspy is None:
        import gdspy

        _gdspy = gdspy
    return _gdspy


def _ensure_shapely():
    """Lazy import of shapely"""
    global _shapely
    if _shapely is None:
        import shapely

        _shapely = shapely
    return _shapely


class GDSLoader:
    """Load and process GDS files for electrode mask generation

    Parameters
    ----------
    gds_path : str or Path
        Path to GDS file
    gds_unit : float, optional
        Conversion factor from GDS units to meters (default: 1e-6 for um)
    origin : Tuple[float, float], optional
        Origin offset in simulation coordinates (x, y) in meters
    cell_name : str, optional
        Name of cell to use (default: top-level cell)
    """

    def __init__(
        self,
        gds_path: str | Path,
        gds_unit: float = 1e-6,
        origin: Tuple[float, float] = (0.0, 0.0),
        cell_name: Optional[str] = None,
    ):
        self.gds_path = Path(gds_path)
        self.gds_unit = gds_unit
        self.origin = origin
        self.cell_name = cell_name

        self._gds_lib = None
        self._cell = None
        self._polygons_by_layer: Dict[Tuple[int, int], List[np.ndarray]] = {}

        self._load_gds()

    def _load_gds(self) -> None:
        """Load GDS file and extract polygons"""
        gdspy = _ensure_gdspy()

        if not self.gds_path.exists():
            raise FileNotFoundError(f"GDS file not found: {self.gds_path}")

        self._gds_lib = gdspy.GdsLibrary()
        self._gds_lib.read_gds(str(self.gds_path))

        # Get cell
        if self.cell_name:
            if self.cell_name not in self._gds_lib.cells:
                available = list(self._gds_lib.cells.keys())
                raise ValueError(
                    f"Cell '{self.cell_name}' not found. Available: {available}"
                )
            self._cell = self._gds_lib.cells[self.cell_name]
        else:
            top_cells = self._gds_lib.top_level()
            if not top_cells:
                raise ValueError("No top-level cells found in GDS file")
            self._cell = top_cells[0]

        # Extract polygons including referenced cells (depth=None means all levels)
        self._polygons_by_layer = self._cell.get_polygons(by_spec=True, depth=None)

    def get_available_layers(self) -> List[Tuple[int, int]]:
        """Get list of (layer, datatype) tuples available in the GDS

        Returns
        -------
        layers : List[Tuple[int, int]]
            List of (layer, datatype) tuples
        """
        return list(self._polygons_by_layer.keys())

    def get_polygons(self, layer: int, datatype: int = 0) -> List[np.ndarray]:
        """Get polygons for a specific layer, converted to simulation coordinates

        Parameters
        ----------
        layer : int
            GDS layer number
        datatype : int, optional
            GDS datatype (default: 0)

        Returns
        -------
        polygons : List[np.ndarray]
            List of polygon vertex arrays, each shape (N, 2) in meters
        """
        key = (layer, datatype)
        if key not in self._polygons_by_layer:
            return []

        polygons = []
        for poly in self._polygons_by_layer[key]:
            # Convert to meters and apply origin offset
            converted = poly.astype(np.float64) * self.gds_unit
            converted[:, 0] += self.origin[0]
            converted[:, 1] += self.origin[1]
            polygons.append(converted)

        return polygons

    def rasterize_layer(
        self,
        layer: int,
        datatype: int,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
    ) -> np.ndarray:
        """Rasterize a GDS layer onto the computational grid

        Uses shapely.contains_xy for efficient point-in-polygon testing.

        Parameters
        ----------
        layer : int
            GDS layer number
        datatype : int
            GDS datatype
        x_coords : np.ndarray
            X coordinates of grid points (1D array in meters)
        y_coords : np.ndarray
            Y coordinates of grid points (1D array in meters)

        Returns
        -------
        mask : np.ndarray
            Boolean mask of shape (nx, ny), True where electrode present
        """
        shapely = _ensure_shapely()

        nx, ny = len(x_coords), len(y_coords)
        mask = np.zeros((nx, ny), dtype=bool)

        polygons = self.get_polygons(layer, datatype)
        if not polygons:
            return mask

        # Create coordinate arrays for vectorized operations
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")

        for poly_vertices in polygons:
            if len(poly_vertices) < 3:
                continue

            shapely_poly = shapely.Polygon(poly_vertices)
            if not shapely_poly.is_valid:
                # Fix invalid polygons (self-intersecting, etc.)
                shapely_poly = shapely_poly.buffer(0)

            if shapely_poly.is_empty:
                continue

            # Vectorized point-in-polygon check (shapely 2.0+)
            contains = shapely.contains_xy(shapely_poly, xx, yy)
            mask |= contains

        return mask

    def get_bounding_box(
        self, layer: int, datatype: int = 0
    ) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box of all polygons in a layer

        Parameters
        ----------
        layer : int
            GDS layer number
        datatype : int, optional
            GDS datatype (default: 0)

        Returns
        -------
        bbox : Tuple[float, float, float, float] or None
            (x_min, y_min, x_max, y_max) in meters, or None if layer is empty
        """
        polygons = self.get_polygons(layer, datatype)
        if not polygons:
            return None

        all_vertices = np.vstack(polygons)
        x_min, y_min = all_vertices.min(axis=0)
        x_max, y_max = all_vertices.max(axis=0)

        return (x_min, y_min, x_max, y_max)
