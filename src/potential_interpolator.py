"""Linear interpolator for 2D potential distributions at specific z-layer

Uses superposition principle for Poisson equation to compute potential
distributions for arbitrary electrode voltages from pre-computed basis solutions.

Mathematical Background
-----------------------
For linear Poisson equation -∇·(ε∇φ) = ρ with N electrodes, the solution
can be decomposed as:

    φ(V₁, V₂, ..., Vₙ) = φ_particular + Σᵢ Vᵢ·φᵢ

where:
- φ_particular: Solution with all electrodes at 0V, charge density ρ (computed once)
- φᵢ: Solution with electrode i at 1V, others at 0V, ρ=0 (computed for each electrode)

This decomposition is exact for linear systems and allows fast computation
of potential for arbitrary voltage combinations without re-solving.

Example
-------
>>> from structure import StructureManager
>>> from poisson_solver import PoissonSolver
>>> from potential_interpolator import PotentialInterpolator
>>>
>>> # Load structure
>>> manager = StructureManager("configs/example.yaml")
>>>
>>> # Create interpolator at Si/SiO2 interface (z=-20nm for example.yaml)
>>> interp = PotentialInterpolator(
...     manager,
...     z_position=-20e-9,
...     omega=1.8,
...     tolerance=1e-6
... )
>>>
>>> # Compute potential for specific voltages
>>> voltages = {"finger_gate_1": 0.5, "finger_gate_2": 1.0, "finger_gate_3": 0.5}
>>> pot_2d = interp(voltages)  # Returns (nx, ny) array
"""

import numpy as np
from typing import Dict, Optional
import warnings


class PotentialInterpolator:
    """Linear interpolator for 2D potential at specific z-coordinate

    Computes 2D potential distributions by linear combination of pre-computed
    basis solutions. Each basis function corresponds to one electrode at 1V
    with all others at 0V, computed with ρ=0.

    Parameters
    ----------
    structure_manager : StructureManager
        Structure manager with electrode definitions
    z_position : float
        Target z-coordinate (m), e.g., -20e-9 for z=-20nm
        Must be within computational domain
    charge_density : np.ndarray, optional
        Charge density distribution (C/m³), shape=(nz, nx, ny)
        Used only for particular solution computation
    carrier_type : str, optional
        "electron" or "hole" to define potential sign convention (default: "electron")
    omega : float, optional
        SOR relaxation parameter for PoissonSolver (default: 1.8)
    tolerance : float, optional
        Convergence tolerance for PoissonSolver (default: 1e-6)
    max_iterations : int, optional
        Maximum iterations for PoissonSolver (default: 10000)
    verbose : int, optional
        Print progress during basis computation (default: 1)

    Attributes
    ----------
    z_position : float
        Target z-coordinate (m)
    z_index : int
        Grid index corresponding to z_position
    electrode_names : List[str]
        Ordered list of electrode names
    basis_phi : np.ndarray
        Basis functions φ (n_electrodes, nx, ny) at z=z_position, computed with ρ=0
    particular_band_edge : np.ndarray
        Particular solution as band edge (Ec or Ev in eV) (nx, ny) at z=z_position
        Computed from φ_particular with all V=0, ρ≠0
    electron_affinity : float
        Electron affinity χ (eV) at z=z_position
    band_gap : float
        Band gap Eg (eV) at z=z_position
    x : np.ndarray
        x-coordinates (m), shape=(nx,)
    y : np.ndarray
        y-coordinates (m), shape=(ny,)

    Raises
    ------
    ValueError
        If z_position is out of computational domain
        If structure has no electrodes defined

    Notes
    -----
    - Basis functions are computed with ρ=0 for exact linear decomposition
    - Particular solution includes charge density effects
    - Computation time scales linearly with number of electrodes
    - For systems with many electrodes, consider saving/loading interpolator
    - Use __call__ method for convenient interpolation: `pot_2d = interp(voltages)`

    Examples
    --------
    >>> # Create interpolator at Si/SiO2 interface
    >>> interp = PotentialInterpolator(
    ...     manager,
    ...     z_position=-20e-9,
    ...     omega=1.8,
    ...     tolerance=1e-6
    ... )
    >>>
    >>> # Option 1: Use __call__
    >>> pot_2d = interp({"gate1": 0.5, "gate2": 1.0})
    >>>
    >>> # Option 2: Use interpolate method
    >>> pot_2d = interp.interpolate({"gate1": 0.5, "gate2": 1.0})
    """

    def __init__(
        self,
        structure_manager,  # StructureManager type
        z_position: float,
        charge_density: Optional[np.ndarray] = None,
        carrier_type: str = "electron",
        omega: float = 1.8,
        tolerance: float = 1e-6,
        max_iterations: int = 10000,
        verbose: int = 1,
    ):
        # Import here to avoid circular dependency
        from structure import StructureManager
        from poisson_solver import PoissonSolver

        # Get coordinate arrays
        x_coords, y_coords, z_coords = structure_manager.get_grid_coordinates()

        # Find z_index corresponding to z_position
        z_index = self._find_nearest_z_index(z_coords, z_position)
        actual_z = z_coords[z_index]

        if verbose:
            print(f"Target z-position: {z_position * 1e9:.2f} nm")
            print(f"Nearest grid point: z_index={z_index}, z={actual_z * 1e9:.2f} nm")
            if abs(actual_z - z_position) > structure_manager.h / 2:
                warnings.warn(
                    f"Requested z={z_position * 1e9:.2f} nm is far from grid point "
                    f"z={actual_z * 1e9:.2f} nm (offset={abs(actual_z - z_position) * 1e9:.2f} nm)"
                )

        # Store z information
        self.z_position = actual_z  # Use actual grid point
        self.z_index = z_index
        self.carrier_type = carrier_type
        # Check that electrodes exist
        if not structure_manager.electrodes:
            raise ValueError("Structure has no electrodes defined")

        self.electrode_names = [e["name"] for e in structure_manager.electrodes]
        self.n_electrodes = len(self.electrode_names)

        # Store grid dimensions
        self.nx = structure_manager.nx
        self.ny = structure_manager.ny
        self.nz = structure_manager.nz

        # Store coordinates
        self.x = x_coords
        self.y = y_coords

        # Store grid spacing and domain size
        self.dx = structure_manager.h  # Grid spacing (isotropic)
        self.dy = structure_manager.h
        self.Lx = structure_manager.size_x
        self.Ly = structure_manager.size_y

        # Store original electrode voltages for restoration
        self._original_voltages = [e["voltage"] for e in structure_manager.electrodes]

        # Store solver parameters
        self._solver_params = {
            "omega": omega,
            "tolerance": tolerance,
            "max_iterations": max_iterations,
        }

        # Store material parameters at z_index for band edge conversion
        material_at_z = structure_manager.get_material_at_z(z_index)
        self.electron_affinity = material_at_z.electron_affinity  # eV
        self.band_gap = material_at_z.band_gap  # eV

        # Initialize arrays:
        # - basis_phi: electrostatic potential φ for linear superposition
        # - particular_band_edge: pre-computed band edge (Ec or Ev) for efficiency
        self.basis_phi = np.zeros((self.n_electrodes, self.nx, self.ny))
        self.particular_band_edge = np.zeros((self.nx, self.ny))

        if verbose:
            print(f"Computing basis functions for {self.n_electrodes} electrodes...")

        # Step 1: Compute particular solution (all electrodes at 0V, ρ≠0)
        self._compute_particular_solution(structure_manager, charge_density, verbose)

        # Step 2: Compute basis functions (one electrode at 1V, others at 0V, ρ=0)
        self._compute_basis_functions(structure_manager, verbose)

        # Restore original voltages
        for i, electrode in enumerate(structure_manager.electrodes):
            electrode["voltage"] = self._original_voltages[i]
        structure_manager.get_electrode_voltages()

        if verbose:
            print("Interpolator initialization complete.")

    @staticmethod
    def _find_nearest_z_index(z_coords: np.ndarray, z_target: float) -> int:
        """Find grid index nearest to target z-coordinate

        Parameters
        ----------
        z_coords : np.ndarray
            Grid z-coordinates (m), shape=(nz,)
        z_target : float
            Target z-coordinate (m)

        Returns
        -------
        z_index : int
            Index of nearest grid point

        Raises
        ------
        ValueError
            If z_target is outside computational domain
        """
        # Check if target is within domain
        z_min, z_max = z_coords.min(), z_coords.max()
        if not (z_min <= z_target <= z_max):
            raise ValueError(
                f"z_position={z_target * 1e9:.2f} nm is outside domain "
                f"[{z_min * 1e9:.2f}, {z_max * 1e9:.2f}] nm"
            )

        # Find nearest index
        z_index = int(np.argmin(np.abs(z_coords - z_target)))
        return z_index

    def _compute_particular_solution(
        self,
        structure_manager,
        charge_density: Optional[np.ndarray],
        verbose: int,
    ) -> None:
        """Compute particular solution with all electrodes at 0V

        Parameters
        ----------
        structure_manager : StructureManager
            Structure manager (will be modified)
        charge_density : np.ndarray, optional
            Charge density (C/m³), shape=(nz, nx, ny)
        verbose : int
            Print progress
        """
        from poisson_solver import PoissonSolver

        if verbose:
            print("  [1/2] Computing particular solution (all V=0, ρ≠0)...")

        # Set all electrodes to 0V
        for electrode in structure_manager.electrodes:
            electrode["voltage"] = 0.0
        structure_manager.get_electrode_voltages()

        # Create solver with current electrode configuration
        solver = PoissonSolver(structure_manager, **self._solver_params)

        # Solve with charge density
        solver_verbose = verbose > 1
        result = solver.solve(rho=charge_density, verbose=solver_verbose)

        if not result.info.get("converged", False):
            warnings.warn(
                f"Particular solution did not converge: "
                f"iterations={result.info.get('iterations', 0)}"
            )

        # Extract 2D slice and convert to band edge immediately (computed once)
        # This avoids repeated conversion during interpolation
        phi_2d = result.phi[self.z_index, :, :]
        if self.carrier_type == "electron":
            # Ec = -φ - χ
            self.particular_band_edge = -phi_2d - self.electron_affinity
        else:  # hole
            # Ev = -φ - χ - Eg
            self.particular_band_edge = -phi_2d - self.electron_affinity - self.band_gap

        if verbose:
            print(
                f"      Converged: {result.info.get('converged', False)}, "
                f"Iterations: {result.info.get('iterations', 0)}"
            )

    def _compute_basis_functions(
        self,
        structure_manager,
        verbose: int,
    ) -> None:
        """Compute basis functions for each electrode

        Each basis function: electrode i at 1V, others at 0V, ρ=0

        Parameters
        ----------
        structure_manager : StructureManager
            Structure manager (will be modified)
        verbose : int
            Print progress
        """
        from poisson_solver import PoissonSolver

        if verbose:
            print(f"  [2/2] Computing {self.n_electrodes} basis functions (ρ=0)...")

        for i, electrode_name in enumerate(self.electrode_names):
            if verbose:
                print(
                    f"      [{i + 1}/{self.n_electrodes}] Electrode: {electrode_name}"
                )

            # Set voltages: electrode i = 1V, others = 0V
            for j, electrode in enumerate(structure_manager.electrodes):
                electrode["voltage"] = 1.0 if j == i else 0.0
            structure_manager.get_electrode_voltages()

            # Create solver with current electrode configuration
            solver = PoissonSolver(structure_manager, **self._solver_params)

            # Solve with ρ=0 for exact linear decomposition
            solver_verbose = verbose > 1
            result = solver.solve(rho=None, verbose=solver_verbose)

            if not result.info.get("converged", False):
                warnings.warn(
                    f"Basis function {i} ({electrode_name}) did not converge: "
                    f"iterations={result.info.get('iterations', 0)}"
                )

            # Extract 2D slice of electrostatic potential φ at z_index
            # Store φ directly without conversion to Ec/Ev (conversion happens in interpolate())
            self.basis_phi[i, :, :] = result.phi[self.z_index, :, :]

    def interpolate(self, voltages: Dict[str, float]) -> np.ndarray:
        """Compute 2D band edge at z_position for given electrode voltages

        Uses optimized linear superposition:
            Band_edge = Band_edge_particular - Σᵢ Vᵢ·φᵢ

        where Band_edge_particular is pre-computed (Ec or Ev) and φᵢ are basis potentials.
        The negative sign comes from: Ec = -φ - χ, so ΔEc = -Δφ

        Parameters
        ----------
        voltages : Dict[str, float]
            Electrode voltages (V), e.g., {"finger_gate_1": 0.5, "finger_gate_2": 1.0}
            All electrodes must be specified

        Returns
        -------
        band_edge_2d : np.ndarray
            2D band edge distribution (eV) at z=z_position, shape=(nx, ny)
            Returns Ec for electrons, Ev for holes

        Raises
        ------
        ValueError
            If voltage keys do not match electrode names

        Examples
        --------
        >>> voltages = {"gate1": 0.5, "gate2": 1.0, "gate3": 0.5}
        >>> Ec_2d = interpolator.interpolate(voltages)  # Returns Ec if carrier_type="electron"
        >>> Ec_2d.shape
        (100, 100)
        """
        # Validate voltage dictionary
        voltage_keys = set(voltages.keys())
        electrode_keys = set(self.electrode_names)

        if voltage_keys != electrode_keys:
            missing = electrode_keys - voltage_keys
            extra = voltage_keys - electrode_keys
            error_msg = "Voltage keys do not match electrode names.\n"
            if missing:
                error_msg += f"  Missing: {missing}\n"
            if extra:
                error_msg += f"  Extra: {extra}\n"
            error_msg += f"  Expected: {electrode_keys}"
            raise ValueError(error_msg)

        # Start with pre-computed band edge (particular solution)
        band_edge_2d = self.particular_band_edge.copy()

        # Add contributions from each electrode
        # Note: Ec = -φ - χ, so when φ increases by Vᵢ·φᵢ, Ec decreases by Vᵢ·φᵢ
        for i, electrode_name in enumerate(self.electrode_names):
            V_i = voltages[electrode_name]
            band_edge_2d -= V_i * self.basis_phi[i, :, :]

        return band_edge_2d

    def __call__(self, voltages: Dict[str, float]) -> np.ndarray:
        """Convenience method for interpolation

        Equivalent to calling interpolate(voltages).

        Parameters
        ----------
        voltages : Dict[str, float]
            Electrode voltages (V)

        Returns
        -------
        pot_2d : np.ndarray
            2D potential distribution (V) at z=z_position, shape=(nx, ny)

        Examples
        --------
        >>> pot_2d = interp({"gate1": 0.5, "gate2": 1.0})
        """
        return self.interpolate(voltages)

    def save(self, filepath: str) -> None:
        """Save interpolator state to NPZ file

        Saves all necessary data for reconstruction without re-solving.

        Parameters
        ----------
        filepath : str
            Path to save file (.npz format)

        Notes
        -----
        Use `PotentialInterpolator.load(filepath)` to reconstruct
        Material parameters (electron_affinity, band_gap) are not saved since
        particular_band_edge is already computed.
        """
        np.savez(
            filepath,
            z_position=self.z_position,
            z_index=self.z_index,
            carrier_type=self.carrier_type,
            electrode_names=np.array(self.electrode_names, dtype=object),
            basis_phi=self.basis_phi,
            particular_band_edge=self.particular_band_edge,
            x=self.x,
            y=self.y,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
        )
        print(f"Interpolator saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "PotentialInterpolator":
        """Load interpolator from NPZ file

        Parameters
        ----------
        filepath : str
            Path to saved file (.npz format)

        Returns
        -------
        interpolator : PotentialInterpolator
            Reconstructed interpolator instance

        Notes
        -----
        Creates a "skeleton" instance without StructureManager or PoissonSolver,
        only for interpolation purposes. Cannot re-compute basis functions.
        Material parameters are not loaded since they're not needed for interpolation.
        """
        data = np.load(filepath, allow_pickle=True)

        # Create empty instance (bypass __init__)
        instance = cls.__new__(cls)

        # Restore attributes
        instance.z_position = float(data["z_position"])
        instance.z_index = int(data["z_index"])
        instance.carrier_type = str(data["carrier_type"])
        instance.electrode_names = list(data["electrode_names"])
        instance.n_electrodes = len(instance.electrode_names)
        instance.basis_phi = data["basis_phi"]
        instance.particular_band_edge = data["particular_band_edge"]
        instance.x = data["x"]
        instance.y = data["y"]
        instance.nx = int(data["nx"])
        instance.ny = int(data["ny"])
        instance.nz = int(data["nz"])

        print(f"Interpolator loaded from: {filepath}")
        print(f"  Electrodes: {instance.electrode_names}")
        print(
            f"  z-position: {instance.z_position * 1e9:.2f} nm (index={instance.z_index})"
        )
        print(f"  carrier_type: {instance.carrier_type}")

        return instance

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"PotentialInterpolator(z_position={self.z_position * 1e9:.2f} nm, "
            f"z_index={self.z_index}, "
            f"n_electrodes={self.n_electrodes}, "
            f"grid=({self.nx}, {self.ny}))"
            f"carrier_type={self.carrier_type}"
        )
