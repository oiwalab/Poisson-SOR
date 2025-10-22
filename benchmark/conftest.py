"""Common fixtures for benchmark tests

Focuses on realistic use case: two-layer heterostructure with multiple finger gate electrodes
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poisson_solver import PoissonSolver


@pytest.fixture(params=[20, 50, 100])
def grid_size(request):
    """Grid size parameter: small (20), medium (50), large (100)"""
    return request.param


@pytest.fixture
def solver(grid_size):
    """Two-layer heterostructure solver with multiple finger gate electrodes

    Realistic configuration based on example.yaml:
    - Top layer: SiO2 (epsilon_r = 3.9), ~20% of domain
    - Bottom layer: Si (epsilon_r = 11.7), ~80% of domain
    - Three finger gate electrodes with different voltages (-0.5V, -1.0V, -0.5V)
    - Electrode depth: ~10% of domain
    """
    nz, nx, ny = grid_size, grid_size, grid_size
    h = 1e-9

    epsilon = np.ones((nz, nx, ny)) * 11.7
    epsilon[nz // 5 :, :, :] = 3.9

    electrode_mask = np.zeros((nz, nx, ny), dtype=bool)
    electrode_voltages = np.zeros((nz, nx, ny))

    electrode_depth = max(1, nz // 10)

    finger_1_x = slice(nx // 10, 3 * nx // 10)
    finger_1_y = slice(ny // 4, 3 * ny // 4)
    electrode_mask[0:electrode_depth, finger_1_x, finger_1_y] = True
    electrode_voltages[0:electrode_depth, finger_1_x, finger_1_y] = -0.5

    finger_2_x = slice(4 * nx // 10, 6 * nx // 10)
    finger_2_y = slice(ny // 4, 3 * ny // 4)
    electrode_mask[0:electrode_depth, finger_2_x, finger_2_y] = True
    electrode_voltages[0:electrode_depth, finger_2_x, finger_2_y] = -1.0

    finger_3_x = slice(7 * nx // 10, 9 * nx // 10)
    finger_3_y = slice(ny // 4, 3 * ny // 4)
    electrode_mask[0:electrode_depth, finger_3_x, finger_3_y] = True
    electrode_voltages[0:electrode_depth, finger_3_x, finger_3_y] = -0.5

    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

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

    return solver
