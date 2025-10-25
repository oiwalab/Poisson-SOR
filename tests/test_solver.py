"""Test cases for PoissonSolver

Basic functionality verification and validation
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poisson_solver import PoissonSolver


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
    params = {
        "epsilon": epsilon,
        "grid_spacing": h,
        "boundary_conditions": boundary_conditions,
        "electrode_mask": None,
        "electrode_voltages": None,
    }
    solver = PoissonSolver(
        params,
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
    params = {
        "epsilon": epsilon,
        "grid_spacing": h,
        "boundary_conditions": boundary_conditions,
        "electrode_mask": None,
        "electrode_voltages": None,
    }
    solver = PoissonSolver(
        params,
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
    params = {
        "epsilon": epsilon,
        "grid_spacing": h,
        "boundary_conditions": boundary_conditions,
        "electrode_mask": electrode_mask,
        "electrode_voltages": electrode_voltages,
    }
    solver = PoissonSolver(
        params,
        omega=1.8,
        tolerance=1e-6,
        max_iterations=10000,
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
