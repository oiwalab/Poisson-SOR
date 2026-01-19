"""Test cases for GDSLoader

Tests for GDS file loading, coordinate conversion, and polygon rasterization
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gds_loader import GDSLoader


def create_test_gds_file(output_path: str, cell_name: str = "TEST"):
    """Create a simple test GDS file with known polygons

    Creates:
    - Layer 1: Rectangle from (0, 0) to (10, 5) um
    - Layer 2: L-shaped polygon
    """
    import gdspy

    # Create a fresh library to avoid global state issues
    lib = gdspy.GdsLibrary(name="test_lib")

    # Create cell directly without using current_library
    cell = gdspy.Cell(cell_name, exclude_from_current=True)
    lib.add(cell)

    # Layer 1: Simple rectangle (in um)
    rect = gdspy.Rectangle((0, 0), (10, 5), layer=1, datatype=0)
    cell.add(rect)

    # Layer 2: L-shaped polygon
    l_shape = gdspy.Polygon(
        [(20, 0), (30, 0), (30, 10), (25, 10), (25, 5), (20, 5)],
        layer=2,
        datatype=0,
    )
    cell.add(l_shape)

    lib.write_gds(output_path)


class TestGDSLoader:
    """Tests for GDSLoader class"""

    def test_load_gds_file(self, tmp_path):
        """Test loading a GDS file"""
        gds_file = tmp_path / "test.gds"
        create_test_gds_file(str(gds_file))

        loader = GDSLoader(gds_file, gds_unit=1e-6)
        layers = loader.get_available_layers()

        assert (1, 0) in layers
        assert (2, 0) in layers

    def test_file_not_found(self, tmp_path):
        """Test FileNotFoundError for missing GDS file"""
        gds_file = tmp_path / "nonexistent.gds"

        with pytest.raises(FileNotFoundError):
            GDSLoader(gds_file)

    def test_coordinate_conversion(self, tmp_path):
        """Test GDS to simulation coordinate conversion"""
        gds_file = tmp_path / "test.gds"
        create_test_gds_file(str(gds_file))

        # Default: 1 um = 1e-6 m
        loader = GDSLoader(gds_file, gds_unit=1e-6)
        polygons = loader.get_polygons(1, 0)

        assert len(polygons) == 1
        # Rectangle corners should be at (0, 0) and (10e-6, 5e-6) in meters
        assert np.allclose(polygons[0].min(axis=0), [0, 0])
        assert np.allclose(polygons[0].max(axis=0), [10e-6, 5e-6])

    def test_coordinate_conversion_with_origin(self, tmp_path):
        """Test coordinate conversion with origin offset"""
        gds_file = tmp_path / "test.gds"
        create_test_gds_file(str(gds_file))

        loader = GDSLoader(gds_file, gds_unit=1e-6, origin=(5e-6, 2e-6))
        polygons = loader.get_polygons(1, 0)

        assert len(polygons) == 1
        # Rectangle should be shifted by origin
        assert np.allclose(polygons[0].min(axis=0), [5e-6, 2e-6])
        assert np.allclose(polygons[0].max(axis=0), [15e-6, 7e-6])

    def test_get_empty_layer(self, tmp_path):
        """Test getting polygons from non-existent layer"""
        gds_file = tmp_path / "test.gds"
        create_test_gds_file(str(gds_file))

        loader = GDSLoader(gds_file, gds_unit=1e-6)
        polygons = loader.get_polygons(99, 0)

        assert polygons == []

    def test_rasterize_rectangle(self, tmp_path):
        """Test rasterizing a rectangle layer to grid"""
        gds_file = tmp_path / "test.gds"
        create_test_gds_file(str(gds_file))

        loader = GDSLoader(gds_file, gds_unit=1e-6)

        # Create grid: 0-15 um in x, 0-10 um in y, 1 um spacing
        x_coords = np.arange(0, 16e-6, 1e-6)
        y_coords = np.arange(0, 11e-6, 1e-6)

        mask = loader.rasterize_layer(1, 0, x_coords, y_coords)

        # Check shape
        assert mask.shape == (16, 11)
        assert mask.dtype == bool

        # Rectangle spans 0-10 um in x, 0-5 um in y
        # Point at (5 um, 2 um) should be inside
        assert mask[5, 2]

        # Point at (12 um, 2 um) should be outside
        assert not mask[12, 2]

        # Point at (5 um, 8 um) should be outside (y > 5)
        assert not mask[5, 8]

    def test_rasterize_empty_layer(self, tmp_path):
        """Test rasterizing an empty layer"""
        gds_file = tmp_path / "test.gds"
        create_test_gds_file(str(gds_file))

        loader = GDSLoader(gds_file, gds_unit=1e-6)

        x_coords = np.arange(0, 10e-6, 1e-6)
        y_coords = np.arange(0, 10e-6, 1e-6)

        mask = loader.rasterize_layer(99, 0, x_coords, y_coords)

        # Shape should match coordinate array lengths
        assert mask.shape == (len(x_coords), len(y_coords))
        assert not mask.any()

    def test_bounding_box(self, tmp_path):
        """Test getting bounding box of a layer"""
        gds_file = tmp_path / "test.gds"
        create_test_gds_file(str(gds_file))

        loader = GDSLoader(gds_file, gds_unit=1e-6)
        bbox = loader.get_bounding_box(1, 0)

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        assert np.isclose(x_min, 0)
        assert np.isclose(y_min, 0)
        assert np.isclose(x_max, 10e-6)
        assert np.isclose(y_max, 5e-6)

    def test_bounding_box_empty_layer(self, tmp_path):
        """Test bounding box for empty layer returns None"""
        gds_file = tmp_path / "test.gds"
        create_test_gds_file(str(gds_file))

        loader = GDSLoader(gds_file, gds_unit=1e-6)
        bbox = loader.get_bounding_box(99, 0)

        assert bbox is None


class TestGDSLoaderCellSelection:
    """Tests for cell selection in GDS files"""

    def create_multi_cell_gds(self, output_path: str):
        """Create a GDS file with multiple cells"""
        import gdspy

        lib = gdspy.GdsLibrary(name="multi_lib")

        # Create a child cell
        child = gdspy.Cell("CHILD", exclude_from_current=True)
        lib.add(child)
        child.add(gdspy.Rectangle((0, 0), (5, 5), layer=1))

        # Create top cell that references child
        top = gdspy.Cell("TOP", exclude_from_current=True)
        lib.add(top)
        top.add(gdspy.Rectangle((10, 10), (20, 15), layer=1))
        top.add(gdspy.CellReference(child, (0, 0)))

        lib.write_gds(output_path)

    def test_select_specific_cell(self, tmp_path):
        """Test selecting a specific cell by name"""
        gds_file = tmp_path / "multi.gds"
        self.create_multi_cell_gds(str(gds_file))

        # Load child cell only
        loader = GDSLoader(gds_file, gds_unit=1e-6, cell_name="CHILD")
        polygons = loader.get_polygons(1, 0)

        # Child has one rectangle
        assert len(polygons) == 1

    def test_invalid_cell_name(self, tmp_path):
        """Test error when cell name doesn't exist"""
        gds_file = tmp_path / "multi.gds"
        self.create_multi_cell_gds(str(gds_file))

        with pytest.raises(ValueError, match="not found"):
            GDSLoader(gds_file, cell_name="NONEXISTENT")


class TestStructureManagerGDSIntegration:
    """Integration tests for GDS electrodes in StructureManager"""

    def test_gds_electrode_mask_generation(self, tmp_path):
        """Test generating electrode mask from GDS file"""
        from structure import StructureManager

        # Create test GDS file
        gds_file = tmp_path / "electrodes.gds"
        create_test_gds_file(str(gds_file))

        # Create YAML config with GDS electrodes
        config = {
            "domain": {
                "size": [20e-6, 15e-6, 50e-9],  # 20x15 um x 50 nm
                "grid_spacing": 1e-6,  # 1 um grid
            },
            "layers": [
                {"material": "SiO2", "z_range": [0, -25e-9]},
                {"material": "Si", "z_range": [-25e-9, -50e-9]},
            ],
            "electrodes": [
                {
                    "source": "gds",
                    "gds_file": str(gds_file),
                    "gds_unit": 1e-6,
                    "origin": [0, 0],
                    "layer_mapping": [
                        {
                            "layer": 1,
                            "datatype": 0,
                            "name": "gate_1",
                            "voltage": 0.5,
                            "z_position": -15e-9,
                        }
                    ],
                }
            ],
            "boundary_conditions": {
                "z_top": {"type": "neumann", "value": 0.0},
                "z_bottom": {"type": "dirichlet", "value": 0.0},
                "x_sides": {"type": "neumann", "value": 0.0},
                "y_sides": {"type": "neumann", "value": 0.0},
            },
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        # Load structure
        structure = StructureManager(str(yaml_file))

        # Check electrode mask
        assert structure.electrode_mask is not None
        assert structure.electrode_mask.any()

        # Check electrode voltages
        assert structure.electrode_voltages is not None
        # Voltage should be 0.5 where mask is True
        masked_voltages = structure.electrode_voltages[structure.electrode_mask]
        assert np.allclose(masked_voltages, 0.5)

    def test_mixed_gds_and_rectangle_electrodes(self, tmp_path):
        """Test mixing GDS and rectangle electrodes"""
        from structure import StructureManager

        # Create test GDS file
        gds_file = tmp_path / "electrodes.gds"
        create_test_gds_file(str(gds_file))

        # Create YAML config with both GDS and rectangle electrodes
        config = {
            "domain": {
                "size": [50e-6, 20e-6, 50e-9],
                "grid_spacing": 1e-6,
            },
            "layers": [
                {"material": "SiO2", "z_range": [0, -25e-9]},
                {"material": "Si", "z_range": [-25e-9, -50e-9]},
            ],
            "electrodes": [
                {
                    "source": "gds",
                    "gds_file": str(gds_file),
                    "gds_unit": 1e-6,
                    "layer_mapping": [
                        {
                            "layer": 1,
                            "voltage": 0.5,
                            "z_position": -15e-9,
                        }
                    ],
                },
                {
                    "name": "rect_gate",
                    "shape": "rectangle",
                    "x_range": [40e-6, 45e-6],
                    "y_range": [5e-6, 15e-6],
                    "z_position": -15e-9,
                    "voltage": -0.3,
                },
            ],
            "boundary_conditions": {
                "z_top": {"type": "neumann", "value": 0.0},
                "z_bottom": {"type": "dirichlet", "value": 0.0},
                "x_sides": {"type": "neumann", "value": 0.0},
                "y_sides": {"type": "neumann", "value": 0.0},
            },
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        structure = StructureManager(str(yaml_file))

        # Both electrodes should be present
        assert structure.electrode_mask.any()

        # Check that we have both voltages
        unique_voltages = np.unique(
            structure.electrode_voltages[structure.electrode_mask]
        )
        assert 0.5 in unique_voltages or np.isclose(unique_voltages, 0.5).any()
        assert -0.3 in unique_voltages or np.isclose(unique_voltages, -0.3).any()
