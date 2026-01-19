"""GPU-accelerated Poisson solver backend using CuPy (CUDA)"""

from .rbsor_cupy import (
    redblack_sor_iteration_gpu,
    apply_boundary_conditions_gpu,
)

__all__ = [
    "redblack_sor_iteration_gpu",
    "apply_boundary_conditions_gpu",
]
