"""Benchmark Julia vs Numba implementation

Compares performance of Julia and Numba backends for PoissonSolver
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poisson_solver import PoissonSolver


class MockStructureManager:
    """Mock StructureManager for benchmarking"""

    def __init__(self, epsilon_array, h, boundary_conditions):
        self.epsilon_array = epsilon_array
        self.nz, self.nx, self.ny = epsilon_array.shape
        self.h = h
        self.boundary_conditions = boundary_conditions
        self.electrode_mask = None
        self.electrode_voltages = None
        self.size_x = self.nx * h
        self.size_y = self.ny * h
        self.size_z = self.nz * h


def benchmark_solver(use_julia: bool, grid_size: tuple = (50, 50, 50)):
    """Benchmark solver with specified backend

    Parameters
    ----------
    use_julia : bool
        Whether to use Julia backend
    grid_size : tuple
        Grid dimensions (nz, nx, ny)

    Returns
    -------
    elapsed_time : float
        Time taken in seconds
    iterations : int
        Number of iterations to converge
    """
    nz, nx, ny = grid_size
    h = 2e-9  # 2nm grid spacing

    # Uniform permittivity
    epsilon = np.ones((nz, nx, ny)) * 11.7

    # Boundary conditions
    boundary_conditions = {
        "z_top": {"type": "dirichlet", "value": 1.0},
        "z_bottom": {"type": "dirichlet", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    # Create structure
    structure = MockStructureManager(
        epsilon_array=epsilon, h=h, boundary_conditions=boundary_conditions
    )

    # Create solver
    solver = PoissonSolver(
        structure,
        omega=1.8,
        tolerance=1e-8,
        max_iterations=5000,
        use_julia=use_julia,
    )

    # Initial condition
    phi_initial = np.zeros((nz, nx, ny))
    phi_initial[0, :, :] = 1.0
    phi_initial[-1, :, :] = 0.0

    # Benchmark
    start_time = time.time()
    result = solver.solve(phi_initial=phi_initial, verbose=False)
    elapsed_time = time.time() - start_time

    return elapsed_time, result.info["iterations"]


def main():
    """Run benchmark comparison"""
    print("=" * 60)
    print("Poisson Solver Benchmark: Julia vs Numba")
    print("=" * 60)

    # Test different grid sizes
    grid_sizes = [
        (30, 30, 30),
        (50, 50, 50),
        (70, 70, 70),
    ]

    for grid_size in grid_sizes:
        nz, nx, ny = grid_size
        total_points = nz * nx * ny

        print(f"\nGrid size: {nz} x {nx} x {ny} ({total_points:,} points)")
        print("-" * 60)

        # Benchmark Julia
        try:
            time_julia, iters_julia = benchmark_solver(use_julia=True, grid_size=grid_size)
            print(f"Julia backend:  {time_julia:.3f}s ({iters_julia} iterations)")
        except Exception as e:
            print(f"Julia backend:  Failed - {e}")
            time_julia = None

        # Benchmark Numba
        try:
            time_numba, iters_numba = benchmark_solver(
                use_julia=False, grid_size=grid_size
            )
            print(f"Numba backend:  {time_numba:.3f}s ({iters_numba} iterations)")
        except Exception as e:
            print(f"Numba backend:  Failed - {e}")
            time_numba = None

        # Show speedup
        if time_julia and time_numba:
            speedup = time_numba / time_julia
            print(f"\nSpeedup: {speedup:.2f}x (Julia vs Numba)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
