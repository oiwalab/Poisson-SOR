"""3D Poisson equation solver using SOR method

Solves Poisson equation -∇⋅(ε∇φ)=ρ for systems with non-uniform permittivity
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from numba import njit
from poisson_result import PoissonResult

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
    """

    def __init__(
        self,
        structure: "StructureManager",
        omega: float = 1.8,
        tolerance: float = 1e-6,
        max_iterations: int = 10000,
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

        self._precompute_z_interfaces()

    def solve(
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

    def _precompute_z_interfaces(self):
        """Precompute harmonic mean of permittivity at z-direction interfaces

        For heterostructure applications where permittivity is uniform in x,y
        but varies in z direction. Detects interfaces where epsilon[k] != epsilon[k+1]
        and precomputes harmonic mean values to avoid repeated calculations during iteration.

        Stores results in self.eps_z_interfaces as:
        {k: eps_interface} where eps_interface is harmonic mean between layer k and k+1

        Also creates self.eps_z_array (nz-1,) for JIT function:
        eps_z_array[k] is harmonic mean between layer k and k+1
        """
        self.eps_z_interfaces = {}
        self.eps_z_array = np.zeros(self.nz - 1)

        for k in range(self.nz - 1):
            eps_k = self.epsilon[k, 0, 0]
            eps_kp = self.epsilon[k + 1, 0, 0]

            if eps_k != eps_kp:
                eps_harmonic = 2 * eps_k * eps_kp / (eps_k + eps_kp)
                self.eps_z_interfaces[k] = eps_harmonic
                self.eps_z_array[k] = eps_harmonic
            else:
                self.eps_z_array[k] = eps_k

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

        # Call JIT-compiled function
        return _sor_iteration_jit(
            phi,
            rho,
            self.epsilon,
            self.eps_z_array,
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

    def apply_surface_boundary(self, phi: np.ndarray) -> np.ndarray:
        """Apply mixed boundary conditions at surface

        At electrode positions: Dirichlet boundary condition (fixed voltage)
        At non-electrode positions: Neumann boundary condition (d_phi/d_z = 0)

        Parameters
        ----------
        phi : np.ndarray
            Potential distribution

        Returns
        -------
        phi_new : np.ndarray
            Potential distribution after applying boundary conditions
        """
        phi_new = phi.copy()
        k_surface = -1  # z-direction index of surface

        if self.electrode_mask is None or self.electrode_voltages is None:
            # Use default Neumann boundary condition if no electrode info
            default_value = self.boundary_conditions.get("z_top", {}).get(
                "default_neumann_value", 0.0
            )
            if default_value == 0.0:
                phi_new[:, :, k_surface] = phi_new[:, :, k_surface - 1]
            else:
                phi_new[:, :, k_surface] = (
                    phi_new[:, :, k_surface - 1] + default_value * self.h
                )
            return phi_new

        # Apply boundary conditions at each grid point
        for i in range(self.nx):
            for j in range(self.ny):
                if self.electrode_mask[i, j, k_surface]:
                    # If electrode exists: Dirichlet boundary condition
                    phi_new[i, j, k_surface] = self.electrode_voltages[i, j, k_surface]
                else:
                    # If no electrode: Neumann boundary condition (d_phi/d_z = 0)
                    default_value = self.boundary_conditions.get("z_top", {}).get(
                        "default_neumann_value", 0.0
                    )
                    if default_value == 0.0:
                        phi_new[i, j, k_surface] = phi_new[i, j, k_surface - 1]
                    else:
                        phi_new[i, j, k_surface] = (
                            phi_new[i, j, k_surface - 1] + default_value * self.h
                        )

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

            eps_zp = self.eps_z_interfaces.get(k, eps_k)
            eps_zm = self.eps_z_interfaces.get(k - 1, eps_k)

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

                    residual_array[k, i, j] = -laplacian + rho[k, i, j] / self.epsilon_0

        return np.sqrt(np.mean(residual_array**2)) * h2


@njit
def _sor_iteration_jit(
    phi: np.ndarray,
    rho: np.ndarray,
    epsilon: np.ndarray,
    eps_z_array: np.ndarray,
    electrode_mask: np.ndarray,
    h: float,
    omega: float,
    epsilon_0: float,
    nz: int,
    nx: int,
    ny: int,
) -> np.ndarray:
    """JIT-compiled SOR iteration core computation

    Parameters
    ----------
    phi : np.ndarray
        Potential distribution (nz, nx, ny)
    rho : np.ndarray
        Charge density distribution (nz, nx, ny)
    epsilon : np.ndarray
        Permittivity distribution (nz, nx, ny)
    eps_z_array : np.ndarray
        Harmonic mean at z-interfaces, shape (nz-1,)
        eps_z_array[k] is harmonic mean between layer k and k+1
    electrode_mask : np.ndarray
        Electrode mask (nz, nx, ny), True where electrodes exist
    h : float
        Grid spacing
    omega : float
        SOR relaxation parameter
    epsilon_0 : float
        Vacuum permittivity
    nz, nx, ny : int
        Grid dimensions

    Returns
    -------
    phi : np.ndarray
        Updated potential distribution
    """
    h2 = h * h

    for k in range(1, nz - 1):
        eps_k = epsilon[k, 0, 0]

        eps_zp = eps_z_array[k]
        eps_zm = eps_z_array[k - 1]

        az = eps_zp / h2
        bz = eps_zm / h2
        axy = eps_k / h2

        A = 4 * axy + az + bz

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if electrode_mask[k, i, j]:
                    continue

                B = (
                    axy
                    * (
                        phi[k, i + 1, j]
                        + phi[k, i - 1, j]
                        + phi[k, i, j + 1]
                        + phi[k, i, j - 1]
                    )
                    + az * phi[k + 1, i, j]
                    + bz * phi[k - 1, i, j]
                    + rho[k, i, j] / epsilon_0
                )

                phi[k, i, j] = (1 - omega) * phi[k, i, j] + omega * (B / A)

    return phi
