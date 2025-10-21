"""Test cases for PoissonSolver and StructureManager

Basic functionality verification and validation
"""

import numpy as np
import pytest
import sys
import tempfile
import yaml
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poisson_solver import PoissonSolver
from structure_manager import StructureManager


def test_uniform_dielectric_neumann():
    """Test with uniform dielectric and Neumann boundary conditions

    With rho=0 and Neumann boundary conditions, potential should be constant everywhere
    """
    # Small grid (new coordinate system: shape = (nz, nx, ny))
    nz, nx, ny = 10, 10, 10
    h = 1e-9  # 1nm (isotropic grid)

    # Uniform permittivity (Si) - array shape: (nz, nx, ny)
    epsilon = np.ones((nz, nx, ny)) * 11.7

    # Neumann boundary conditions (all with d_phi/d_n = 0)
    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    # Initialize solver
    solver = PoissonSolver(
        epsilon=epsilon,
        grid_spacing=h,
        boundary_conditions=boundary_conditions,
        omega=1.5,
        tolerance=1e-6,
        max_iterations=1000,
    )

    # Solve (zero charge density)
    phi, info = solver.solve()

    # With rho=0 and Neumann BC, potential should be constant
    assert phi.std() < 1e-6, (
        "Potential should be constant with zero charge and Neumann BC"
    )
    assert info["converged"], "Solver should converge"
    assert info["iterations"] >= 1


def test_parallel_plate_capacitor():
    """Test for parallel plate capacitor (simplified version)

    Approximated as 1D problem varying only in z direction

    New coordinate system:
    - z_top (k=0, z=0nm): 1V
    - z_bottom (k=nz-1, z=-20nm): 0V
    """
    # Isotropic grid (small grid for 1D problem)
    # Array shape: (nz, nx, ny)
    nz, nx, ny = 11, 3, 3
    h = 2e-9  # 2nm (isotropic grid)

    # Uniform permittivity - array shape: (nz, nx, ny)
    epsilon = np.ones((nz, nx, ny)) * 3.9  # SiO2

    # Boundary conditions: Dirichlet at top and bottom (fixed voltage)
    boundary_conditions = {
        "z_top": {"type": "dirichlet", "value": 1.0},  # Surface (k=0): 1V
        "z_bottom": {"type": "dirichlet", "value": 0.0},  # Bottom (k=nz-1): 0V
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    # Initialize solver
    solver = PoissonSolver(
        epsilon=epsilon,
        grid_spacing=h,
        boundary_conditions=boundary_conditions,
        omega=1.5,
        tolerance=1e-8,
        max_iterations=5000,
    )

    # Set initial condition with Dirichlet boundary conditions manually
    # Array shape: (nz, nx, ny)
    phi_initial = np.zeros((nz, nx, ny))
    phi_initial[0, :, :] = 1.0  # z_top (k=0): 1V
    phi_initial[-1, :, :] = 0.0  # z_bottom (k=nz-1): 0V

    # Solve
    phi, info = solver.solve(phi_initial=phi_initial)

    # Check for NaN
    assert not np.isnan(phi).any(), "Solution should not contain NaN"

    # Check if boundary conditions are correctly set
    assert np.abs(phi[0, :, :].mean() - 1.0) < 1e-6, "Top boundary (k=0) should be 1V"
    assert np.abs(phi[-1, :, :].mean() - 0.0) < 1e-6, (
        "Bottom boundary (k=nz-1) should be 0V"
    )

    # Compare with analytical solution: for parallel plate, phi(k) = V_top * (1 - k/K) (linear, 1->0)
    # k=0 -> phi=1, k=nz-1 -> phi=0
    k_coords = np.arange(nz)
    K = nz - 1
    phi_analytical = 1.0 - k_coords / K  # Linear from 1 to 0

    # Compare at center point (array shape: (nz, nx, ny))
    phi_numerical = phi[:, 1, 1]

    print("\nParallel plate capacitor test:")
    print(f"Converged: {info['converged']}, Iterations: {info['iterations']}")
    print(f"Max absolute error: {np.abs(phi_numerical - phi_analytical).max():.6e}")

    # Check error against analytical solution at interior points (excluding boundaries)
    error = np.abs(phi_numerical[1:-1] - phi_analytical[1:-1])
    max_error = error.max()
    assert max_error < 0.01, (
        f"Max error {max_error:.6e} should be < 0.01 (1% of voltage range)"
    )


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


def test_electrode_volume():
    """Test treating electrodes as 3D volumes

    New coordinate system:
    - Array shape: (nz, nx, ny)
    - Electrodes extend downward from surface (k=0, z=0) as 3D volumes
    """
    # Small grid (array shape: (nz, nx, ny))
    nz, nx, ny = 11, 11, 11
    h = 10e-9  # 10nm (isotropic grid)

    # Uniform permittivity
    epsilon = np.ones((nz, nx, ny)) * 11.7

    # Electrode mask: 3D volume at center (top 2 layers: k=0,1)
    # Array shape: (nz, nx, ny)
    electrode_mask = np.zeros((nz, nx, ny), dtype=bool)
    k_electrode_top = 0  # Surface
    k_electrode_bottom = 1  # z-index of electrode bottom (2 layers: k=0,1)
    electrode_mask[k_electrode_top : k_electrode_bottom + 1, 4:7, 4:7] = (
        True  # Top 2 layers, center 3x3
    )

    # Electrode voltage: -0.5V
    electrode_voltages = np.zeros((nz, nx, ny))
    electrode_voltages[k_electrode_top : k_electrode_bottom + 1, 4:7, 4:7] = -0.5

    # Boundary conditions (all Neumann)
    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    # Initialize solver
    solver = PoissonSolver(
        epsilon=epsilon,
        grid_spacing=h,
        boundary_conditions=boundary_conditions,
        omega=1.8,
        tolerance=1e-6,
        max_iterations=10000,
        electrode_mask=electrode_mask,
        electrode_voltages=electrode_voltages,
    )

    # Set initial condition (start from values close to electrode voltage)
    phi_initial = np.zeros((nz, nx, ny))
    phi_initial[electrode_mask] = -0.5

    # Solve
    phi, info = solver.solve(phi_initial=phi_initial)

    # Check voltage in entire electrode region
    electrode_phi = phi[electrode_mask]
    assert np.allclose(electrode_phi, -0.5, atol=1e-6), (
        f"Electrode potential should be -0.5V, but got mean={electrode_phi.mean():.6f}"
    )

    # Potential outside electrode should be higher than electrode voltage (closer to 0V)
    # Array shape: (nz, nx, ny) -> [k, i, j]
    non_electrode_phi = phi[5, 0, 0]  # Edge point of middle layer
    assert non_electrode_phi > electrode_phi.mean(), (
        "Non-electrode region should have higher potential than electrode"
    )

    # Check for NaN
    assert not np.isnan(phi).any(), "Solution should not contain NaN"

    assert info["converged"], "Solver should converge"
