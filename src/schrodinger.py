"""
2D Schrödinger Equation Solver

Solves the 2D effective-mass Schrödinger equation:
    -ℏ²/(2m*) ∇²ψ + V(x,y)ψ = Eψ

using finite difference method and sparse matrix eigenvalue solver.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.constants import hbar, elementary_charge
from typing import Optional


class SchrodingerSolver2D:
    """
    2D Schrödinger equation solver using finite difference method.

    Solves the effective-mass Schrödinger equation in 2D using a 5-point
    stencil finite difference discretization and scipy.sparse.linalg.eigsh
    for efficient computation of low-energy eigenstates.

    Uses Dirichlet boundary conditions: ψ = 0 at all boundaries.

    Parameters
    ----------
    potential : np.ndarray
        2D potential energy array, shape (nx, ny), units: eV
    dx : float
        Grid spacing in x direction, units: meters
    dy : float
        Grid spacing in y direction, units: meters
    effective_mass : float
        Effective mass of particle, units: kg

    Attributes
    ----------
    nx, ny : int
        Grid dimensions
    potential : np.ndarray
        Potential energy in Joules (converted from Volts)
    hamiltonian : scipy.sparse.csr_matrix
        Sparse Hamiltonian matrix (built on demand)

    Notes
    -----
    - Boundary condition: Dirichlet (ψ = 0 at boundaries)
    - Energy units: Input V (Volts), Output E (eV)
    - Wavefunction normalization: ∫|ψ|²dxdy = 1

    Examples
    --------
    >>> from scipy.constants import electron_mass
    >>> V = np.zeros((50, 50))  # Flat potential
    >>> solver = SchrodingerSolver2D(V, 1e-9, 1e-9, 0.19*electron_mass)
    >>> energies, psi = solver.solve(n_states=5)
    >>> print(f"Ground state: {energies[0]:.4f} eV")
    """

    def __init__(
        self,
        potential: np.ndarray,
        dx: float,
        dy: float,
        effective_mass: float,
    ):
        # Validate inputs
        if potential.ndim != 2:
            raise ValueError(f"Potential must be 2D array, got shape {potential.shape}")
        if dx <= 0 or dy <= 0:
            raise ValueError(f"Grid spacing must be positive, got dx={dx}, dy={dy}")
        if effective_mass <= 0:
            raise ValueError(f"Effective mass must be positive, got {effective_mass}")

        self.nx, self.ny = potential.shape
        self.dx = dx
        self.dy = dy
        self.effective_mass = effective_mass

        self.potential = potential  # [eV]

        # Hamiltonian will be built on demand
        self._hamiltonian = None

    @property
    def hamiltonian(self) -> sparse.csr_matrix:
        """
        Get or build the Hamiltonian matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse Hamiltonian matrix
        """
        if self._hamiltonian is None:
            self._hamiltonian = self._build_hamiltonian()
        return self._hamiltonian

    def solve(
        self,
        n_states: int = 5,
        which: str = "SA",
        sigma: Optional[float] = None,
        tol: float = 0,
        maxiter: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve for lowest energy eigenstates.

        Parameters
        ----------
        n_states : int, optional
            Number of eigenstates to compute (default: 5)
        which : str, optional
            Which eigenvalues to find: 'SA' = smallest algebraic (default)
        sigma : float, optional
            Shift for shift-invert mode (finds eigenvalues near sigma)
        tol : float, optional
            Convergence tolerance (default: 0 = machine precision)
        maxiter : int, optional
            Maximum iterations for eigensolver

        Returns
        -------
        energies : np.ndarray
            Energy eigenvalues in eV, shape (n_states,)
        wavefunctions : np.ndarray
            Normalized wavefunctions, shape (n_states, nx, ny)

        Notes
        -----
        Uses scipy.sparse.linalg.eigsh which is optimized for Hermitian
        matrices. The 'SA' option efficiently finds the lowest energy states.
        """
        if n_states < 1:
            raise ValueError(f"n_states must be >= 1, got {n_states}")

        H = self.hamiltonian

        if n_states >= H.shape[0] - 1:
            raise ValueError(
                f"n_states ({n_states}) must be < matrix size - 1 ({H.shape[0] - 1})"
            )

        # Solve eigenvalue problem
        eigenvalues, eigenvectors = eigsh(
            H,
            k=n_states,
            which=which,
            sigma=sigma,
            tol=tol,
            maxiter=maxiter,
        )

        # Convert energies from Joules to eV
        energies = eigenvalues / elementary_charge

        # Reshape eigenvectors to 2D wavefunctions and normalize
        wavefunctions = np.zeros((n_states, self.nx, self.ny))

        for i in range(n_states):
            psi_flat = eigenvectors[:, i]
            psi_2d = self._unflatten_wavefunction(psi_flat)
            wavefunctions[i] = self.normalize_wavefunction(psi_2d)

        return energies, wavefunctions

    def _build_hamiltonian(self) -> sparse.csr_matrix:
        """
        Build Hamiltonian with Dirichlet boundary conditions.

        Boundary points are excluded (ψ = 0 at boundaries).
        Interior points: (nx-2) × (ny-2)

        Uses vectorized operations for efficiency.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse Hamiltonian matrix
        """
        # Potential energy in Joules (convert from eV)
        V_joules = self.potential * elementary_charge

        # Interior grid dimensions (excluding boundaries)
        nx_int = self.nx - 2
        ny_int = self.ny - 2
        n_points = nx_int * ny_int

        if n_points < 1:
            raise ValueError(
                f"Grid too small for Dirichlet BC: need nx,ny >= 3, got {self.nx},{self.ny}"
            )

        # Kinetic energy coefficients
        coeff_xx = hbar**2 / (2 * self.effective_mass * self.dx**2)
        coeff_yy = hbar**2 / (2 * self.effective_mass * self.dy**2)
        coeff_diag = 2 * coeff_xx + 2 * coeff_yy

        # Extract interior potential (exclude boundaries)
        V_interior = V_joules[1:-1, 1:-1].flatten()

        # Diagonal: kinetic energy (diagonal part) + potential
        diagonal = coeff_diag + V_interior

        # Off-diagonal elements (kinetic energy coupling)
        # x-direction: coupling to left/right neighbors (offset ±ny_int)
        offdiag_x = -coeff_xx * np.ones(n_points - ny_int)

        # y-direction: coupling to up/down neighbors (offset ±1)
        # Need to exclude connections across x-boundaries
        offdiag_y = -coeff_yy * np.ones(n_points - 1)
        # Remove connections at y-boundaries (every ny_int-th element)
        offdiag_y[ny_int - 1 :: ny_int] = 0

        # Build sparse matrix using diagonals
        H = sparse.diags(
            [diagonal, offdiag_x, offdiag_x, offdiag_y, offdiag_y],
            [0, -ny_int, ny_int, -1, 1],
            shape=(n_points, n_points),
            format="csr",
        )

        return H

    def _unflatten_wavefunction(self, psi_flat: np.ndarray) -> np.ndarray:
        """
        Convert flat wavefunction array to 2D grid.

        Parameters
        ----------
        psi_flat : np.ndarray
            Flat wavefunction from eigensolver (interior points only)

        Returns
        -------
        np.ndarray
            2D wavefunction, shape (nx, ny), with zero boundaries
        """
        psi_2d = np.zeros((self.nx, self.ny))

        # Interior points only (boundaries remain zero)
        nx_int = self.nx - 2
        ny_int = self.ny - 2
        psi_interior = psi_flat.reshape(nx_int, ny_int)
        psi_2d[1:-1, 1:-1] = psi_interior

        return psi_2d

    def normalize_wavefunction(self, psi: np.ndarray) -> np.ndarray:
        """
        Normalize wavefunction so that ∫|ψ|²dxdy = 1.

        Parameters
        ----------
        psi : np.ndarray
            Wavefunction, shape (nx, ny)

        Returns
        -------
        np.ndarray
            Normalized wavefunction
        """
        norm_squared = np.sum(np.abs(psi) ** 2) * self.dx * self.dy
        norm = np.sqrt(norm_squared)

        if norm < 1e-12:
            raise ValueError("Wavefunction norm is too small (near zero)")

        return psi / norm

    def compute_probability_density(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute probability density |ψ|².

        Parameters
        ----------
        psi : np.ndarray
            Wavefunction, shape (nx, ny)

        Returns
        -------
        np.ndarray
            Probability density, shape (nx, ny)
        """
        return np.abs(psi) ** 2

    def compute_expectation_value(self, psi: np.ndarray, operator: np.ndarray) -> float:
        """
        Compute expectation value <ψ|Ô|ψ> for a diagonal operator.

        Parameters
        ----------
        psi : np.ndarray
            Wavefunction, shape (nx, ny), assumed normalized
        operator : np.ndarray
            Operator in position representation, shape (nx, ny)

        Returns
        -------
        float
            Expectation value

        Notes
        -----
        For diagonal operators only (position-dependent operators).
        For example, <x>, <y>, <V(x,y)>, etc.
        """
        integrand = np.abs(psi) ** 2 * operator
        return np.sum(integrand) * self.dx * self.dy

    def save(self, filepath: str):
        """
        Save solver state to npz file.

        Parameters
        ----------
        filepath : str
            Path to save file (.npz extension recommended)

        Notes
        -----
        Saves all parameters needed to reconstruct the solver.
        Hamiltonian is not saved (rebuilt on load).
        """
        np.savez(
            filepath,
            potential=self.potential,  # Save in eV
            dx=self.dx,
            dy=self.dy,
            effective_mass=self.effective_mass,
        )

    @staticmethod
    def load(filepath: str) -> "SchrodingerSolver2D":
        """
        Load solver from npz file.

        Parameters
        ----------
        filepath : str
            Path to saved file

        Returns
        -------
        SchrodingerSolver2D
            Reconstructed solver instance
        """
        data = np.load(filepath)

        return SchrodingerSolver2D(
            potential=data["potential"],
            dx=float(data["dx"]),
            dy=float(data["dy"]),
            effective_mass=float(data["effective_mass"]),
        )
