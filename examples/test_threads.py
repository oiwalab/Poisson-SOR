"""Simple script to test thread count setting

Run this script in a fresh Python process to test thread count setting.
"""

import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poisson_solver import PoissonSolver


class MockStructureManager:
    """Mock StructureManager for testing"""

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

    def get_grid_coordinates(self):
        x = np.arange(self.nx) * self.h
        y = np.arange(self.ny) * self.h
        z = -np.arange(self.nz) * self.h
        return x, y, z


def main():
    print("=== Testing Julia Thread Count Setting ===\n")

    # Small grid for quick test
    nz, nx, ny = 30, 30, 30
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

    # Test with different thread counts
    print("Creating solver with num_threads=4...")
    solver = PoissonSolver(
        structure,
        method="redblack",
        num_threads=4,
        max_iterations=100,
        tolerance=1e-6,
    )

    if solver._use_julia and solver._julia_main is not None:
        # Query actual thread count
        try:
            actual_threads = solver._julia_main.get_num_threads()
            print(f"\nActual Julia threads: {actual_threads}")

            if actual_threads == 4:
                print("[SUCCESS] Thread count matches requested value!")
            else:
                print(f"[INFO] Thread count is {actual_threads}, requested was 4")
                print("This might happen due to system constraints or Julia configuration")
        except Exception as e:
            print(f"Error querying thread count: {e}")

        # Run a quick solve to test
        print("\nRunning solver...")
        import time
        start = time.time()
        result = solver.solve(verbose=False)
        elapsed = time.time() - start

        print(f"Converged: {result.info['converged']}")
        print(f"Iterations: {result.info['iterations']}")
        print(f"Time: {elapsed:.3f} seconds")
    else:
        print("Julia backend not available")


if __name__ == "__main__":
    main()
