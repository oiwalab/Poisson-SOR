"""Test cases for StructureManager

Tests for structure management, YAML configuration loading, and validation
"""

import numpy as np
import pytest
import sys
import tempfile
import yaml
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from structure_manager import StructureManager


def test_structure_manager_valid_config():
    """Test loading valid configuration with StructureManager

    New coordinate system:
    - SiO2: z=0nm -> z=-50nm
    - Si: z=-50nm -> z=-100nm
    """
    # Valid case (new coordinate system: z_range = [z_max, z_min])
    valid_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "SiO2", "z_range": [0, -50e-9], "epsilon_r": 3.9},  # Surface side
            {
                "material": "Si",
                "z_range": [-50e-9, -100e-9],
                "epsilon_r": 11.7,
            },  # Bottom side
        ],
        "boundary_conditions": {
            "z_top": {"type": "neumann", "value": 0.0},
            "z_bottom": {"type": "neumann", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(valid_config, f)
        temp_file = f.name

    try:
        manager = StructureManager()
        manager.load_from_yaml(temp_file)

        # Check permittivity distribution (array shape: (nz, nx, ny))
        assert manager.epsilon_array is not None
        assert manager.epsilon_array.shape == (11, 11, 11)  # 100nm / 10nm + 1

        # Check permittivity at layer boundary
        # z=-50nm -> k=5 (k = -z/h = -(-50e-9)/10e-9 = 5)
        k_interface = int(-(-50e-9) / 10e-9)
        eps_sio2 = manager.epsilon_array[k_interface - 1, 5, 5]  # k=4 (SiO2 side)
        eps_si = manager.epsilon_array[k_interface, 5, 5]  # k=5 (Si side)
        assert np.abs(eps_sio2 - 3.9) < 1e-6
        assert np.abs(eps_si - 11.7) < 1e-6

    finally:
        Path(temp_file).unlink()


def test_load_from_actual_yaml_file():
    """Test loading from actual YAML configuration file

    Tests that the type conversion fix works with real YAML files
    containing scientific notation as strings
    """
    # Path to actual example config file
    config_path = Path(__file__).parent.parent / "configs" / "example.yaml"

    # Test direct instantiation with config_path
    manager = StructureManager(str(config_path))

    # Verify that values were correctly converted to float
    assert isinstance(manager.size_x, float), "size_x should be float"
    assert isinstance(manager.size_y, float), "size_y should be float"
    assert isinstance(manager.size_z, float), "size_z should be float"
    assert isinstance(manager.h, float), "grid_spacing should be float"

    # Verify grid dimensions
    assert manager.nx > 0, "nx should be positive"
    assert manager.ny > 0, "ny should be positive"
    assert manager.nz > 0, "nz should be positive"

    # Verify arrays were initialized
    assert manager.epsilon_array is not None
    assert manager.epsilon_array.shape == (manager.nz, manager.nx, manager.ny)

    # Verify electrodes were loaded
    assert len(manager.electrodes) > 0, "Should have electrodes defined"
    assert manager.electrode_mask is not None
    assert manager.electrode_voltages is not None

    # Verify layers were loaded
    assert len(manager.layers) > 0, "Should have layers defined"

    print(f"\nSuccessfully loaded config from {config_path}")
    print(f"Grid size: ({manager.nx}, {manager.ny}, {manager.nz})")
    print(f"Domain size: ({manager.size_x}, {manager.size_y}, {manager.size_z})")
    print(f"Grid spacing: {manager.h}")


def test_structure_manager_layer_gap():
    """Test for layer gap detection in StructureManager

    Detect gaps in new coordinate system
    """
    gap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {
                "material": "SiO2",
                "z_range": [0, -40e-9],
                "epsilon_r": 3.9,
            },  # z=0 -> z=-40nm
            {
                "material": "Si",
                "z_range": [-50e-9, -100e-9],
                "epsilon_r": 11.7,
            },  # z=-50nm -> z=-100nm (gap exists)
        ],
        "boundary_conditions": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(gap_config, f)
        temp_file = f.name

    try:
        manager = StructureManager()
        with pytest.raises(ValueError, match="Gap detected"):
            manager.load_from_yaml(temp_file)
    finally:
        Path(temp_file).unlink()


def test_structure_manager_layer_overlap():
    """Test for layer overlap detection in StructureManager

    Detect overlaps in new coordinate system
    """
    overlap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {
                "material": "SiO2",
                "z_range": [0, -60e-9],
                "epsilon_r": 3.9,
            },  # z=0 -> z=-60nm
            {
                "material": "Si",
                "z_range": [-50e-9, -100e-9],
                "epsilon_r": 11.7,
            },  # z=-50nm -> z=-100nm (overlap exists)
        ],
        "boundary_conditions": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(overlap_config, f)
        temp_file = f.name

    try:
        manager = StructureManager()
        with pytest.raises(ValueError, match="Overlap detected"):
            manager.load_from_yaml(temp_file)
    finally:
        Path(temp_file).unlink()


def test_electrode_overlap_detection():
    """Test for electrode overlap detection

    New coordinate system: Electrodes are placed at surface (z=0)
    """
    overlap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "Si", "z_range": [0, -100e-9], "epsilon_r": 11.7},
        ],
        "electrodes": [
            {
                "name": "gate1",
                "shape": "rectangle",
                "x_range": [10e-9, 40e-9],
                "y_range": [10e-9, 40e-9],
                "z_position": -10e-9,  # Electrode bottom (negative value)
                "voltage": -0.5,
            },
            {
                "name": "gate2",
                "shape": "rectangle",
                "x_range": [30e-9, 60e-9],  # Overlaps with gate1
                "y_range": [30e-9, 60e-9],
                "z_position": -10e-9,  # Electrode bottom (negative value)
                "voltage": -1.0,
            },
        ],
        "boundary_conditions": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(overlap_config, f)
        temp_file = f.name

    try:
        manager = StructureManager()
        with pytest.raises(ValueError, match="Electrode overlap detected"):
            manager.load_from_yaml(temp_file)
    finally:
        Path(temp_file).unlink()


def test_electrode_no_overlap():
    """Test with no electrode overlap

    New coordinate system: Electrodes are placed at surface (z=0)
    """
    no_overlap_config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "Si", "z_range": [0, -100e-9], "epsilon_r": 11.7},
        ],
        "electrodes": [
            {
                "name": "gate1",
                "shape": "rectangle",
                "x_range": [10e-9, 30e-9],
                "y_range": [10e-9, 30e-9],
                "z_position": -10e-9,  # Electrode bottom (negative value)
                "voltage": -0.5,
            },
            {
                "name": "gate2",
                "shape": "rectangle",
                "x_range": [40e-9, 60e-9],  # No overlap with gate1
                "y_range": [40e-9, 60e-9],
                "z_position": -10e-9,  # Electrode bottom (negative value)
                "voltage": -1.0,
            },
        ],
        "boundary_conditions": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(no_overlap_config, f)
        temp_file = f.name

    try:
        manager = StructureManager()
        manager.load_from_yaml(temp_file)  # Verify no error occurs

        # Check electrode mask (array shape: (nz, nx, ny))
        assert manager.electrode_mask is not None
        assert manager.electrode_mask.any(), (
            "Electrode mask should have some True values"
        )

        # Check electrode voltages
        assert manager.electrode_voltages is not None
        assert manager.electrode_voltages.min() == -1.0
        assert manager.electrode_voltages.max() == 0.0  # Non-electrode regions are 0V

    finally:
        Path(temp_file).unlink()


def test_material_data_efficiency():
    """Test that material data is stored efficiently

    Verifies that unique_materials and material_indices are properly generated
    """
    config = {
        "domain": {
            "size": [100e-9, 100e-9, 100e-9],
            "grid_spacing": 10e-9,
        },
        "layers": [
            {"material": "SiO2", "z_range": [0, -50e-9], "epsilon_r": 3.9},
            {"material": "Si", "z_range": [-50e-9, -100e-9], "epsilon_r": 11.7},
        ],
        "boundary_conditions": {
            "z_top": {"type": "neumann", "value": 0.0},
            "z_bottom": {"type": "neumann", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_file = f.name

    try:
        manager = StructureManager(temp_file)

        # Should have only 2 unique materials despite 11 z-layers
        assert len(manager.unique_materials) == 2, (
            f"Should have 2 unique materials, got {len(manager.unique_materials)}"
        )

        # Should have material indices for all z-layers
        assert manager.material_indices.shape == (11,), (
            f"Should have 11 material indices, got shape {manager.material_indices.shape}"
        )

        # Check that material indices reference correct materials
        # Layers 0-4 should be SiO2 (index 0 or 1)
        # Layers 5-10 should be Si (index 0 or 1)
        mat_at_z0 = manager.get_material_at_z(0)
        mat_at_z10 = manager.get_material_at_z(10)

        assert mat_at_z0.name == "SiO2", f"Material at z=0 should be SiO2, got {mat_at_z0.name}"
        assert mat_at_z10.name == "Si", f"Material at z=10 should be Si, got {mat_at_z10.name}"

    finally:
        Path(temp_file).unlink()
