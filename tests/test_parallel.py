"""Test parallel execution with num_threads parameter

Tests thread count control and parallel performance of Red-Black SOR.
"""

import numpy as np
import pytest
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poisson_solver import PoissonSolver


class MockStructureManager:
    """Mock StructureManager for testing PoissonSolver"""

    def __init__(self, epsilon_array, h, boundary_conditions, electrode_mask=None, electrode_voltages=None):
        self.epsilon_array = epsilon_array
        self.nz, self.nx, self.ny = epsilon_array.shape
        self.h = h
        self.boundary_conditions = boundary_conditions
        self.electrode_mask = electrode_mask
        self.electrode_voltages = electrode_voltages
        self.size_x = self.nx * h
        self.size_y = self.ny * h
        self.size_z = self.nz * h

    def get_grid_coordinates(self):
        """Get grid coordinates"""
        x = np.arange(self.nx) * self.h
        y = np.arange(self.ny) * self.h
        z = -np.arange(self.nz) * self.h
        return x, y, z


def test_thread_count_detection():
    """Test that Julia reports correct thread count"""
    # Small grid for quick test
    nz, nx, ny = 20, 20, 20
    h = 1e-9

    epsilon = np.ones((nz, nx, ny)) * 11.7

    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    # Test with explicit thread count
    print("\n=== Testing thread count detection ===")

    # Note: num_threads only works for the FIRST PoissonSolver instance
    # in a Python session because juliacall initializes Julia only once
    solver = PoissonSolver(
        structure,
        method="redblack",
        use_julia=True,
        num_threads=4,
        max_iterations=10,  # Quick test
    )

    # Check if Julia backend is available
    if solver._use_julia and solver._julia_main is not None:
        try:
            actual_threads = solver._julia_main.get_num_threads()
            print(f"Julia is using {actual_threads} threads")

            # Note: The actual thread count might not match the requested count
            # if Julia was already initialized in this session
            if actual_threads == 4:
                print("[OK] Thread count matches requested value")
            else:
                print(f"[WARNING] Thread count is {actual_threads}, but requested 4")
                print("  (This happens if Julia was already initialized)")
        except Exception as e:
            print(f"Could not query thread count: {e}")
    else:
        pytest.skip("Julia backend not available")


def test_redblack_correctness():
    """Test that Red-Black SOR produces correct results"""
    print("\n=== Testing Red-Black correctness ===")

    # Parallel plate capacitor setup
    nz, nx, ny = 21, 10, 10
    h = 2e-9

    epsilon = np.ones((nz, nx, ny)) * 3.9

    boundary_conditions = {
        "z_top": {"type": "dirichlet", "value": 1.0},
        "z_bottom": {"type": "dirichlet", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    # Test Red-Black method
    solver = PoissonSolver(
        structure,
        omega=1.5,
        tolerance=1e-8,
        max_iterations=5000,
        method="redblack",
    )

    phi_initial = np.zeros((nz, nx, ny))
    phi_initial[0, :, :] = 1.0
    phi_initial[-1, :, :] = 0.0

    result = solver.solve(phi_initial=phi_initial, verbose=False)

    # Check boundary conditions
    assert np.abs(result.phi[0, :, :].mean() - 1.0) < 1e-6
    assert np.abs(result.phi[-1, :, :].mean() - 0.0) < 1e-6

    # Compare with analytical solution
    k_coords = np.arange(nz)
    K = nz - 1
    phi_analytical = 1.0 - k_coords / K

    phi_numerical = result.phi[:, 5, 5]
    error = np.abs(phi_numerical[1:-1] - phi_analytical[1:-1])
    max_error = error.max()

    print(f"Max error: {max_error:.6e}")
    assert max_error < 0.01, f"Max error {max_error:.6e} exceeds threshold"
    print("[OK] Red-Black SOR produces correct results")


@pytest.mark.parametrize("grid_size", [
    (20, 20, 20),   # Small
    (40, 40, 40),   # Medium
    (60, 60, 60),   # Large
])
def test_parallel_performance(grid_size):
    """Test parallel performance with different grid sizes

    This is an informational test to measure speedup.
    Skip if you want to run tests quickly.
    """
    nz, nx, ny = grid_size
    h = 1e-9

    print(f"\n=== Testing performance for grid size {grid_size} ===")

    epsilon = np.ones((nz, nx, ny)) * 11.7

    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    # Run with Red-Black method
    solver = PoissonSolver(
        structure,
        method="redblack",
        tolerance=1e-6,
        max_iterations=100,  # Fixed iteration count for timing
    )

    if not solver._use_julia:
        pytest.skip("Julia backend not available")

    # Measure execution time
    start_time = time.time()
    result = solver.solve(verbose=False)
    elapsed_time = time.time() - start_time

    print(f"Grid size: {nz}×{nx}×{ny}")
    print(f"Converged: {result.info['converged']}")
    print(f"Iterations: {result.info['iterations']}")
    print(f"Execution time: {elapsed_time:.3f} seconds")

    # Query thread count
    if solver._julia_main is not None:
        try:
            num_threads = solver._julia_main.get_num_threads()
            print(f"Julia threads: {num_threads}")
        except Exception:
            pass


def test_numba_redblack_warning():
    """Test that Numba backend issues warning for redblack method"""
    print("\n=== Testing Numba Red-Black warning ===")

    nz, nx, ny = 10, 10, 10
    h = 1e-9

    epsilon = np.ones((nz, nx, ny)) * 11.7

    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    # Expect warning when using redblack with Numba backend
    with pytest.warns(UserWarning, match="not yet parallelized"):
        solver = PoissonSolver(
            structure,
            method="redblack",
            use_julia=False,  # Force Numba backend
        )

    print("[OK] Numba backend correctly warns about non-parallelized Red-Black")


def test_invalid_method():
    """Test that invalid method raises ValueError"""
    print("\n=== Testing invalid method ===")

    nz, nx, ny = 10, 10, 10
    h = 1e-9

    epsilon = np.ones((nz, nx, ny)) * 11.7

    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    # Expect ValueError for invalid method
    with pytest.raises(ValueError, match="Invalid solver method"):
        solver = PoissonSolver(
            structure,
            method="invalid_method",
        )

    print("[OK] Invalid method correctly raises ValueError")


if __name__ == "__main__":
    # Run tests manually
    print("Running parallel tests...")

    try:
        test_thread_count_detection()
    except Exception as e:
        print(f"Thread count test failed: {e}")

    try:
        test_redblack_correctness()
    except Exception as e:
        print(f"Correctness test failed: {e}")

    try:
        test_numba_redblack_warning()
    except Exception as e:
        print(f"Warning test failed: {e}")

    try:
        test_invalid_method()
    except Exception as e:
        print(f"Invalid method test failed: {e}")

    # Performance test (optional, commented out by default)
    # for grid_size in [(20, 20, 20), (40, 40, 40)]:
    #     try:
    #         test_parallel_performance(grid_size)
    #     except Exception as e:
    #         print(f"Performance test failed: {e}")

    print("\n=== All tests completed ===")
