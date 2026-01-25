"""3D Poisson equation solver using SOR method

Solves Poisson equation -∇⋅(ε∇φ)=ρ for systems with non-uniform permittivity
"""

import os
import warnings
from pathlib import Path
import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from .poisson_result import PoissonResult
from .core import _sor_iteration_jit, _redblack_sor_iteration_jit

if TYPE_CHECKING:
    from structure import StructureManager


class PoissonSolver:
    """3D Poisson solver using SOR method

    New coordinate system: z = 0 (surface, k=0) -> z = -size_z (bottom, k=nz-1)

    Parameters
    ----------
    structure : StructureManager
        Structure manager containing all structure information
    omega : float, optional
        SOR relaxation parameter (1 < omega < 2), default=1.8
    tolerance : float, optional
        Convergence threshold, default=1e-6
    max_iterations : int, optional
        Maximum number of iterations, default=10000
    method : str, optional
        SOR method to use: "sor" (standard) or "redblack" (Red-Black SOR), default="sor"
    use_julia : bool, optional
        Use Julia backend for faster computation, default=False
    num_threads : int, optional
        Number of threads for Julia parallel execution. Must be set before
        first PoissonSolver instantiation. If None, uses Julia default.
    use_gpu : bool, optional
        Use GPU backend (CuPy/CUDA) for faster computation, default=False.
        Requires CuPy and NVIDIA GPU with CUDA support.
    """

    def __init__(
        self,
        structure: "StructureManager",
        omega: float = 1.8,
        tolerance: float = 1e-6,
        max_iterations: int = 10000,
        method: str = "sor",
        use_julia: bool = False,
        num_threads: Optional[int] = None,
        use_gpu: bool = False,
    ):
        self.structure = structure

        # Frequently accessed data (stored locally for performance)
        self.epsilon = structure.epsilon_array
        self.nz, self.nx, self.ny = self.epsilon.shape  # Array shape: (nz, nx, ny)
        self.h = structure.h  # Isotropic grid spacing
        self.boundary_conditions = structure.boundary_conditions
        self.electrode_mask = structure.electrode_mask
        self.electrode_voltages = structure.electrode_voltages

        # Solver parameters
        self.omega = omega
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Vacuum permittivity (F/m)
        self.epsilon_0 = 8.854187817e-12

        # Convergence history (stores phi difference between iterations)
        self.convergence_history = []

        # Initialize backend
        self.method = method.lower()
        self.num_threads = num_threads
        self._use_julia = use_julia
        self._use_gpu = use_gpu
        self._cp = None  # CuPy module reference
        self.validate_method()

        # Initialize GPU backend if requested
        if not self._use_julia and self._use_gpu:
            self._initialize_gpu()

        self._julia_main = None
        if self._use_julia:
            self._initialize_julia()

    def validate_method(self):
        """Validate solver method"""
        valid_methods = ["sor", "redblack", "multigrid"]

        if self.method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid solver method: {self.method}. Choose from {valid_methods}."
            )
        if self.method in {"sor", "multigrid"} and self._use_gpu:
            warnings.warn(
                "Standard SOR method does not support GPU backend. "
                "Falling back to CPU implementation.",
                UserWarning,
            )

    def _initialize_gpu(self):
        """Initialize GPU backend using CuPy"""
        try:
            import cupy as cp

            # Check if CUDA is available
            if cp.cuda.runtime.getDeviceCount() == 0:
                warnings.warn(
                    "No CUDA devices found. Falling back to CPU implementation.",
                    UserWarning,
                )
                self._use_gpu = False
                return

            self._cp = cp

            # Report GPU info
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            device_name = (
                props["name"].decode()
                if isinstance(props["name"], bytes)
                else props["name"]
            )
            print(f"GPU backend initialized: {device_name}")

        except ImportError:
            warnings.warn(
                "CuPy not available. Install with: uv add cupy-cuda12x. "
                "Falling back to CPU implementation.",
                UserWarning,
            )
            self._use_gpu = False
        except Exception as e:
            warnings.warn(
                f"Failed to initialize GPU backend: {e}. "
                "Falling back to CPU implementation.",
                UserWarning,
            )
            self._use_gpu = False

    def _initialize_julia(self):
        """Initialize Julia environment and load solver module"""

        # Set Julia thread count BEFORE importing juliacall
        if self.num_threads is not None:
            import sys

            # juliacall has not been imported yet in a fresh Python session
            if "juliacall" not in sys.modules:
                os.environ["JULIA_NUM_THREADS"] = str(self.num_threads)
                print(f"Setting Julia threads to {self.num_threads}")
            else:
                warnings.warn(
                    "Julia already initialized. num_threads setting ignored. "
                    "Set num_threads before creating the first PoissonSolver instance.",
                    UserWarning,
                )

        try:
            from juliacall import Main as jl

            # # Get path to Julia source file
            # if self.method.lower() == "sor":
            #     julia_file = Path(__file__).parent / "poisson_julia" / "sor_solver.jl"
            # elif self.method.lower() == "redblack":
            #     julia_file = (
            #         Path(__file__).parent / "poisson_julia" / "redblack_solver.jl"
            #     )
            # else:
            #     raise ValueError(
            #         f"Invalid solver method: {self.method}. Choose 'sor' or 'redblack'."
            #     )

            julia_file = Path(__file__).parent / "julia_backend" / "base_solver.jl"

            if not julia_file.exists():
                print(f"Warning: Julia solver file not found at {julia_file}")
                print("Falling back to Numba implementation")
                self._use_julia = False
                return

            # Load Julia source file
            jl.include(str(julia_file))

            self._julia_main = jl

            # Report actual thread count for redblack method
            try:
                actual_threads = jl.get_num_threads()
                print(
                    f"Julia backend initialized successfully with {actual_threads} threads"
                )
            except Exception:
                print("Julia backend initialized successfully")

        except ImportError:
            print("Warning: juliacall not available. Install with: uv add juliacall")
            print("Falling back to Numba implementation")
            self._use_julia = False
        except Exception as e:
            print(f"Warning: Failed to initialize Julia backend: {e}")
            print("Falling back to Numba implementation")
            self._use_julia = False

    def solve(
        self,
        rho: Optional[np.ndarray] = None,
        phi_initial: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> PoissonResult:
        """Solve the Poisson equation using selected backend

        Parameters
        ----------
        rho : np.ndarray, optional
            Charge density distribution (C/m^3), shape=(nz, nx, ny)
            Treated as zero if None
        phi_initial : np.ndarray, optional
            Initial potential distribution (V)
        verbose : bool, optional
            Print convergence progress (default: True)

        Returns
        -------
        result : PoissonResult
            PoissonResult object containing phi, coordinates, materials, and convergence info
        """
        # Select backend: GPU > Julia > Python/Numba
        if not self._use_julia and self._use_gpu and self._cp is not None:
            return self._solve_gpu(rho, phi_initial, verbose)
        elif self._use_julia and self._julia_main is not None:
            return self._solve_julia(rho, phi_initial, verbose)
        else:
            return self._solve_python(rho, phi_initial, verbose)

    def _solve_python(
        self,
        rho: Optional[np.ndarray] = None,
        phi_initial: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> PoissonResult:
        """Solve the Poisson equation

        Parameters
        ----------
        rho : np.ndarray, optional
            Charge density distribution (C/m^3), shape=(nz, nx, ny)
            Treated as zero if None
        phi_initial : np.ndarray, optional
            Initial potential distribution (V)
        verbose : bool, optional
            Print convergence progress (default: True)

        Returns
        -------
        result : PoissonResult
            PoissonResult object containing phi, coordinates, materials, and convergence info
        """
        # Initialize charge density
        if rho is None:
            rho = np.zeros((self.nz, self.nx, self.ny))

        # Initialize potential
        if phi_initial is None:
            phi = np.zeros((self.nz, self.nx, self.ny))
        else:
            phi = phi_initial.copy()

        self.convergence_history = []

        # Set electrode potential (fixed values)
        if self.electrode_mask is not None and self.electrode_voltages is not None:
            phi[self.electrode_mask] = self.electrode_voltages[self.electrode_mask]

        # Apply initial boundary conditions
        phi = self.apply_boundary_conditions(phi)

        # Store previous phi for convergence check
        phi_prev = phi.copy()

        # SOR iteration
        for iteration in range(self.max_iterations):
            phi = self._sor_iteration(phi, rho)

            # Reapply boundary conditions
            phi = self.apply_boundary_conditions(phi)

            # Compute maximum change in phi
            phi_diff = np.max(np.abs(phi - phi_prev))
            self.convergence_history.append(phi_diff)

            if verbose:
                if (iteration + 1) % 1000 == 0:
                    print("=" * 40)
                if (iteration + 1) % 100 == 0:
                    print(f"Iteration {iteration + 1}: Max Δφ = {phi_diff:.6e}")

            # Check convergence
            if phi_diff < self.tolerance:
                info = {
                    "converged": True,
                    "iterations": iteration + 1,
                    "final_phi_change": phi_diff,
                }
                return self._create_solver_result(phi, info)
            if np.isnan(phi_diff) or np.isinf(phi_diff):
                raise ValueError("Phi change became NaN or Inf, diverging solution.")

            # Update phi_prev for next iteration
            phi_prev = phi.copy()

        # Reached maximum iterations
        info = {
            "converged": False,
            "iterations": self.max_iterations,
            "final_phi_change": phi_diff,
        }
        return self._create_solver_result(phi, info)

    def _solve_julia(
        self,
        rho: Optional[np.ndarray] = None,
        phi_initial: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> PoissonResult:
        """Solve the Poisson equation using Julia backend

        Parameters
        ----------
        rho : np.ndarray, optional
            Charge density distribution (C/m^3), shape=(nz, nx, ny)
            Treated as zero if None
        phi_initial : np.ndarray, optional
            Initial potential distribution (V)
        verbose : bool, optional
            Print convergence progress (default: True)

        Returns
        -------
        result : PoissonResult
            PoissonResult object containing phi, coordinates, materials, and convergence info
        """
        # Initialize charge density
        if rho is None:
            rho = np.zeros((self.nz, self.nx, self.ny))

        # Initialize potential
        if phi_initial is None:
            phi_initial = np.zeros((self.nz, self.nx, self.ny))

        # Prepare electrode data
        if self.electrode_mask is None:
            electrode_mask = np.zeros((self.nz, self.nx, self.ny), dtype=np.bool_)
            electrode_voltages = np.zeros((self.nz, self.nx, self.ny))
        else:
            electrode_mask = self.electrode_mask
            electrode_voltages = self.electrode_voltages

        # Convert numpy arrays to Julia arrays
        jl = self._julia_main
        phi_initial_jl = jl.Array(phi_initial)
        rho_jl = jl.Array(rho)
        epsilon_jl = jl.Array(self.epsilon)
        electrode_mask_jl = jl.Array(electrode_mask)
        electrode_voltages_jl = jl.Array(electrode_voltages)

        # Select Julia method type
        if self.method == "redblack" and not self._use_gpu:
            _method = jl.RedBlack()
        elif self.method == "redblack" and self._use_gpu:
            _method = jl.CUDARedBlack()
        elif self.method == "multigrid":
            _method = jl.MultiGrid()
        elif self.method == "sor":
            _method = jl.SOR()

        # Call Julia solver
        phi_result, info_dict = jl.solve_poisson(
            phi_initial_jl,
            rho_jl,
            epsilon_jl,
            electrode_mask_jl,
            electrode_voltages_jl,
            self.boundary_conditions,
            self.h,
            self.omega,
            self.epsilon_0,
            self.tolerance,
            self.max_iterations,
            _method,
            verbose,
        )

        # Convert Julia arrays back to numpy
        phi_final = np.array(phi_result)

        # Store convergence history
        self.convergence_history = list(info_dict["convergence_history"])

        # Create info dict for PoissonResult
        info = {
            "converged": bool(info_dict["converged"]),
            "iterations": int(info_dict["iterations"]),
            "final_phi_change": float(info_dict["final_phi_change"]),
        }

        return self._create_solver_result(phi_final, info)

    def _solve_gpu(
        self,
        rho: Optional[np.ndarray] = None,
        phi_initial: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> PoissonResult:
        """Solve the Poisson equation using GPU backend (CuPy/CUDA)

        Parameters
        ----------
        rho : np.ndarray, optional
            Charge density distribution (C/m^3), shape=(nz, nx, ny)
            Treated as zero if None
        phi_initial : np.ndarray, optional
            Initial potential distribution (V)
        verbose : bool, optional
            Print convergence progress (default: True)

        Returns
        -------
        result : PoissonResult
            PoissonResult object containing phi, coordinates, materials, and convergence info
        """
        from .core.gpu import (
            redblack_sor_iteration_gpu,
            apply_boundary_conditions_gpu,
        )

        cp = self._cp
        float_gpu = cp.float64

        # Initialize charge density
        if rho is None:
            rho = np.zeros((self.nz, self.nx, self.ny))

        # Initialize potential
        if phi_initial is None:
            phi_initial = np.zeros((self.nz, self.nx, self.ny))

        # Prepare electrode data
        if self.electrode_mask is None:
            electrode_mask = np.zeros((self.nz, self.nx, self.ny), dtype=np.bool_)
            electrode_voltages = np.zeros((self.nz, self.nx, self.ny))
        else:
            electrode_mask = self.electrode_mask
            electrode_voltages = self.electrode_voltages

        # Transfer arrays to GPU
        phi_gpu = cp.asarray(phi_initial, dtype=float_gpu)
        rho_gpu = cp.asarray(rho, dtype=float_gpu)
        epsilon_gpu = cp.asarray(self.epsilon, dtype=float_gpu)
        electrode_mask_gpu = cp.asarray(electrode_mask)
        electrode_voltages_gpu = cp.asarray(electrode_voltages, dtype=float_gpu)

        # Set electrode potential (fixed values)
        phi_gpu[electrode_mask_gpu] = electrode_voltages_gpu[electrode_mask_gpu]

        # Apply initial boundary conditions
        apply_boundary_conditions_gpu(phi_gpu, self.boundary_conditions)

        # Store previous phi for convergence check
        phi_prev_gpu = phi_gpu.copy()

        self.convergence_history = []

        # SOR iteration
        converged = False
        final_iteration = 0
        phi_diff = 0.0

        for iteration in range(self.max_iterations):
            # Red-Black SOR iteration on GPU
            redblack_sor_iteration_gpu(
                phi_gpu,
                rho_gpu,
                epsilon_gpu,
                electrode_mask_gpu,
                self.h,
                self.omega,
                self.epsilon_0,
            )

            # Reapply boundary conditions
            apply_boundary_conditions_gpu(phi_gpu, self.boundary_conditions)

            # Enforce electrode potential
            phi_gpu[electrode_mask_gpu] = electrode_voltages_gpu[electrode_mask_gpu]

            # Compute maximum change in phi (GPU reduction)
            phi_diff = float(cp.max(cp.abs(phi_gpu - phi_prev_gpu)))
            self.convergence_history.append(phi_diff)

            if verbose:
                if (iteration + 1) % 1000 == 0:
                    print("=" * 40)
                if (iteration + 1) % 100 == 0:
                    print(f"Iteration {iteration + 1}: Max Δφ = {phi_diff:.6e}")

            # Check convergence
            if phi_diff < self.tolerance:
                converged = True
                final_iteration = iteration + 1
                break

            if np.isnan(phi_diff) or np.isinf(phi_diff):
                raise ValueError("Phi change became NaN or Inf, diverging solution.")

            # Update phi_prev for next iteration
            cp.copyto(phi_prev_gpu, phi_gpu)

        if not converged:
            final_iteration = self.max_iterations

        # Transfer result back to CPU
        phi_final = cp.asnumpy(phi_gpu)

        info = {
            "converged": converged,
            "iterations": final_iteration,
            "final_phi_change": phi_diff,
        }

        return self._create_solver_result(phi_final, info)

    def _create_solver_result(self, phi: np.ndarray, info: Dict) -> PoissonResult:
        """Create PoissonResult object from solution

        Parameters
        ----------
        phi : np.ndarray
            Potential distribution (V), shape=(nz, nx, ny)
        info : Dict
            Convergence information

        Returns
        -------
        result : PoissonResult
            PoissonResult object with phi, structure reference, and info
        """
        # Create PoissonResult with structure reference
        result = PoissonResult(
            phi=phi,
            structure=self.structure,
            info=info,
        )

        return result

    def _sor_iteration(self, phi: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Single iteration update using SOR method

        Uses finite difference formula for non-uniform permittivity
        Harmonic mean is used for permittivity at z-direction interfaces only
        For heterostructure: assumes permittivity is uniform in x,y directions

        New coordinate system: array shape (nz, nx, ny), loop order k (z) -> i (x) -> j (y)
        """
        # Prepare electrode mask for JIT function
        if self.electrode_mask is None:
            electrode_mask = np.zeros((self.nz, self.nx, self.ny), dtype=np.bool_)
        else:
            electrode_mask = self.electrode_mask

        # Call appropriate JIT-compiled function based on method
        if self.method == "redblack":
            return _redblack_sor_iteration_jit(
                phi,
                rho,
                self.epsilon,
                electrode_mask,
                self.h,
                self.omega,
                self.epsilon_0,
                self.nz,
                self.nx,
                self.ny,
            )
        else:  # method == "sor"
            return _sor_iteration_jit(
                phi,
                rho,
                self.epsilon,
                electrode_mask,
                self.h,
                self.omega,
                self.epsilon_0,
                self.nz,
                self.nx,
                self.ny,
            )

    def apply_boundary_conditions(self, phi: np.ndarray) -> np.ndarray:
        """Apply boundary conditions

        Supports basic Neumann/Dirichlet and periodic boundary conditions

        New coordinate system:
        - z_top (k=0): surface (z=0nm)
        - z_bottom (k=nz-1): bottom (z=-size_z)
        - Array shape: (nz, nx, ny)
        """
        phi_new = phi.copy()
        bc = self.boundary_conditions

        # Boundary conditions in z direction
        # z_top: k=0 (surface, z=0nm)
        if bc.get("z_top", {}).get("type") == "neumann":
            value = bc["z_top"].get("value", 0.0)
            # Approximate d_phi/d_z = value using central difference
            phi_new[0, :, :] = phi_new[1, :, :] - value * self.h
        elif bc.get("z_top", {}).get("type") == "dirichlet":
            value = bc["z_top"].get("value", 0.0)
            phi_new[0, :, :] = value

        # z_bottom: k=nz-1 (bottom, z=-size_z)
        if bc.get("z_bottom", {}).get("type") == "neumann":
            value = bc["z_bottom"].get("value", 0.0)
            phi_new[-1, :, :] = phi_new[-2, :, :] + value * self.h
        elif bc.get("z_bottom", {}).get("type") == "dirichlet":
            value = bc["z_bottom"].get("value", 0.0)
            phi_new[-1, :, :] = value

        # Boundary conditions in x direction (i=0, i=nx-1)
        if bc.get("x_sides", {}).get("type") == "neumann":
            value = bc["x_sides"].get("value", 0.0)
            phi_new[:, 0, :] = phi_new[:, 1, :] - value * self.h
            phi_new[:, -1, :] = phi_new[:, -2, :] + value * self.h
        elif bc.get("x_sides", {}).get("type") == "dirichlet":
            value = bc["x_sides"].get("value", 0.0)
            phi_new[:, 0, :] = value
            phi_new[:, -1, :] = value
        elif bc.get("x_sides", {}).get("type") == "periodic":
            phi_new[:, 0, :] = phi_new[:, -2, :]
            phi_new[:, -1, :] = phi_new[:, 1, :]

        # Boundary conditions in y direction (j=0, j=ny-1)
        if bc.get("y_sides", {}).get("type") == "neumann":
            value = bc["y_sides"].get("value", 0.0)
            phi_new[:, :, 0] = phi_new[:, :, 1] - value * self.h
            phi_new[:, :, -1] = phi_new[:, :, -2] + value * self.h
        elif bc.get("y_sides", {}).get("type") == "dirichlet":
            value = bc["y_sides"].get("value", 0.0)
            phi_new[:, :, 0] = value
            phi_new[:, :, -1] = value
        elif bc.get("y_sides", {}).get("type") == "periodic":
            phi_new[:, :, 0] = phi_new[:, :, -2]
            phi_new[:, :, -1] = phi_new[:, :, 1]

        return phi_new

    def compute_residual(self, phi: np.ndarray, rho: np.ndarray) -> float:
        """Compute residual

        Uses L2 norm
        For heterostructure: assumes permittivity is uniform in x,y directions

        New coordinate system: array shape (nz, nx, ny), loop order k (z) -> i (x) -> j (y)
        """
        residual_array = np.zeros_like(phi)
        h2 = self.h**2

        for k in range(1, self.nz - 1):
            eps_k = self.epsilon[k, 0, 0]

            eps_zp = self.epsilon[k, 0, 0]
            eps_zm = self.epsilon[k - 1, 0, 0]

            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    laplacian = (
                        eps_k
                        * (
                            phi[k, i + 1, j]
                            + phi[k, i - 1, j]
                            + phi[k, i, j + 1]
                            + phi[k, i, j - 1]
                            - 4 * phi[k, i, j]
                        )
                        + eps_zp * (phi[k + 1, i, j] - phi[k, i, j])
                        - eps_zm * (phi[k, i, j] - phi[k - 1, i, j])
                    ) / h2

                    residual_array[k, i, j] = laplacian + rho[k, i, j] / self.epsilon_0

        return np.sqrt(np.mean(residual_array**2)) * h2


# @njit
# def _sor_iteration_jit(
#     phi: np.ndarray,
#     rho: np.ndarray,
#     epsilon: np.ndarray,
#     electrode_mask: np.ndarray,
#     h: float,
#     omega: float,
#     epsilon_0: float,
#     nz: int,
#     nx: int,
#     ny: int,
# ) -> np.ndarray:
#     """JIT-compiled SOR iteration core computation

#     Parameters
#     ----------
#     phi : np.ndarray
#         Potential distribution (nz, nx, ny)
#     rho : np.ndarray
#         Charge density distribution (nz, nx, ny)
#     epsilon : np.ndarray
#         Permittivity distribution (nz, nx, ny)
#     electrode_mask : np.ndarray
#         Electrode mask (nz, nx, ny), True where electrodes exist
#     h : float
#         Grid spacing
#     omega : float
#         SOR relaxation parameter
#     epsilon_0 : float
#         Vacuum permittivity
#     nz, nx, ny : int
#         Grid dimensions

#     Returns
#     -------
#     phi : np.ndarray
#         Updated potential distribution
#     """
#     h2 = h * h

#     for k in range(1, nz - 1):
#         eps_k = epsilon[k, 0, 0]

#         eps_zp = epsilon[k, 0, 0]
#         eps_zm = epsilon[k - 1, 0, 0]

#         az = eps_zp / h2
#         bz = eps_zm / h2
#         axy = eps_k / h2

#         A = 4 * axy + az + bz

#         for i in range(1, nx - 1):
#             for j in range(1, ny - 1):
#                 if electrode_mask[k, i, j]:
#                     continue

#                 B = (
#                     axy
#                     * (
#                         phi[k, i + 1, j]
#                         + phi[k, i - 1, j]
#                         + phi[k, i, j + 1]
#                         + phi[k, i, j - 1]
#                     )
#                     + az * phi[k + 1, i, j]
#                     + bz * phi[k - 1, i, j]
#                     + rho[k, i, j] / epsilon_0
#                 )

#                 phi[k, i, j] = (1 - omega) * phi[k, i, j] + omega * (B / A)

#     return phi
