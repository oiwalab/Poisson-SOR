"""Poisson equation solver package

Provides 3D Poisson equation solver using SOR method with result container.
"""

from .poisson_solver import PoissonSolver
from .poisson_result import PoissonResult

__all__ = ["PoissonSolver", "PoissonResult"]
