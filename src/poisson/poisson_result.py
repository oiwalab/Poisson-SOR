"""Poisson solver result container with band structure information

Stores potential and structure reference, computes band edges on demand
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from structure import StructureManager


@dataclass
class PoissonResult:
    """Container for Poisson solver results with band structure

    Parameters
    ----------
    phi : np.ndarray
        Electrostatic potential distribution (V), shape=(nz, nx, ny)
    structure : StructureManager
        Structure manager containing grid coordinates and material information
    info : Dict
        Convergence information (converged, iterations, final_phi_change)

    Notes
    -----
    Band edges are computed on demand and referenced to vacuum level (E=0):
    - Ec(r) = -q·φ(r) - χ(r)  [eV]
    - Ev(r) = Ec(r) - Eg(r)  [eV]
    where q is elementary charge, χ is electron affinity, Eg is band gap

    Unit conversion: q·φ [J] = q·φ [C·V] → [eV] by dividing by q
    Therefore: q·φ [eV] = φ [V] (numerically equal)
    """

    phi: np.ndarray
    structure: "StructureManager"
    info: Dict

    def __post_init__(self):
        """Validate array shapes"""
        nz, nx, ny = self.phi.shape

        # Check structure dimensions match phi
        if self.structure.nz != nz or self.structure.nx != nx or self.structure.ny != ny:
            raise ValueError(
                f"Structure dimensions ({self.structure.nz}, {self.structure.nx}, {self.structure.ny}) "
                f"do not match phi shape ({nz}, {nx}, {ny})"
            )

    def compute_Ec(self) -> np.ndarray:
        """Compute conduction band edge

        Returns
        -------
        Ec : np.ndarray
            Conduction band edge (eV), shape=(nz, nx, ny)

        Notes
        -----
        Ec(r) = -q·φ(r) - χ(z) [eV]
        where φ is in V, χ is in eV
        Unit conversion: q·φ [J] / q [C] = φ [V] → φ [eV] (numerically equal)
        """
        nz, nx, ny = self.phi.shape
        Ec = np.zeros((nz, nx, ny))

        for k in range(nz):
            mat = self.structure.get_material_at_z(k)
            chi = mat.electron_affinity  # eV
            # φ [V] is numerically equal to q·φ [eV]
            Ec[k, :, :] = -self.phi[k, :, :] - chi

        return Ec

    def compute_Ev(self) -> np.ndarray:
        """Compute valence band edge

        Returns
        -------
        Ev : np.ndarray
            Valence band edge (eV), shape=(nz, nx, ny)

        Notes
        -----
        Ev(r) = Ec(r) - Eg(z) [eV]
        where Eg(z) is band gap at each z-layer
        """
        Ec = self.compute_Ec()
        nz, nx, ny = self.phi.shape
        Ev = np.zeros((nz, nx, ny))

        for k in range(nz):
            mat = self.structure.get_material_at_z(k)
            Eg = mat.band_gap  # eV
            Ev[k, :, :] = Ec[k, :, :] - Eg

        return Ev

    @property
    def Ec(self) -> np.ndarray:
        """Conduction band edge (eV), shape=(nz, nx, ny)"""
        return self.compute_Ec()

    @property
    def Ev(self) -> np.ndarray:
        """Valence band edge (eV), shape=(nz, nx, ny)"""
        return self.compute_Ev()

    @property
    def x(self) -> np.ndarray:
        """x-coordinates (m), shape=(nx,)"""
        x, _, _ = self.structure.get_grid_coordinates()
        return x

    @property
    def y(self) -> np.ndarray:
        """y-coordinates (m), shape=(ny,)"""
        _, y, _ = self.structure.get_grid_coordinates()
        return y

    @property
    def z(self) -> np.ndarray:
        """z-coordinates (m), shape=(nz,)"""
        _, _, z = self.structure.get_grid_coordinates()
        return z

    def get_band_diagram_1d(
        self, x_idx: Optional[int] = None, y_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract 1D band diagram along z-direction at specified (x,y) position

        Parameters
        ----------
        x_idx : int, optional
            Index in x-direction (default: center)
        y_idx : int, optional
            Index in y-direction (default: center)

        Returns
        -------
        z : np.ndarray
            z-coordinates (m), shape=(nz,)
        Ec : np.ndarray
            Conduction band edge along z (eV), shape=(nz,)
        Ev : np.ndarray
            Valence band edge along z (eV), shape=(nz,)
        phi : np.ndarray
            Electrostatic potential along z (V), shape=(nz,)
        """
        nz, nx, ny = self.phi.shape

        # Use center if not specified
        if x_idx is None:
            x_idx = nx // 2
        if y_idx is None:
            y_idx = ny // 2

        # Validate indices
        if not (0 <= x_idx < nx):
            raise ValueError(f"x_idx={x_idx} out of range [0, {nx - 1}]")
        if not (0 <= y_idx < ny):
            raise ValueError(f"y_idx={y_idx} out of range [0, {ny - 1}]")

        # Compute band edges
        Ec_full = self.compute_Ec()
        Ev_full = self.compute_Ev()

        # Get z-coordinates from structure
        _, _, z = self.structure.get_grid_coordinates()

        # Extract 1D slices
        Ec_slice = Ec_full[:, x_idx, y_idx]
        Ev_slice = Ev_full[:, x_idx, y_idx]
        phi_slice = self.phi[:, x_idx, y_idx]

        return z, Ec_slice, Ev_slice, phi_slice

    def save(self, filepath: str) -> None:
        """Save solver result to file

        Parameters
        ----------
        filepath : str
            Path to save file (.npz format)

        Notes
        -----
        Saves phi and convergence info. Structure information is not saved.
        To reconstruct full result, the original StructureManager is needed.
        """
        # Save to file
        np.savez(
            filepath,
            phi=self.phi,
            converged=self.info.get("converged", False),
            iterations=self.info.get("iterations", 0),
            final_phi_change=self.info.get("final_phi_change", 0.0),
        )

        print(f"Solver result saved to: {filepath}")
        print("Note: Structure information not saved. Original StructureManager needed for reconstruction.")

    @classmethod
    def load(cls, filepath: str, structure: "StructureManager") -> "PoissonResult":
        """Load solver result from file

        Parameters
        ----------
        filepath : str
            Path to saved file (.npz format)
        structure : StructureManager
            Structure manager (must match the one used when saving)

        Returns
        -------
        result : PoissonResult
            Loaded solver result

        Notes
        -----
        Requires the original StructureManager to reconstruct the full result
        """
        data = np.load(filepath, allow_pickle=True)

        # Load arrays
        phi = data["phi"]

        # Reconstruct info dict
        info = {
            "converged": bool(data["converged"]),
            "iterations": int(data["iterations"]),
            "final_phi_change": float(data["final_phi_change"]),
        }

        return cls(
            phi=phi,
            structure=structure,
            info=info,
        )

    @property
    def nz(self) -> int:
        """Number of grid points in z-direction"""
        return self.phi.shape[0]

    @property
    def nx(self) -> int:
        """Number of grid points in x-direction"""
        return self.phi.shape[1]

    @property
    def ny(self) -> int:
        """Number of grid points in y-direction"""
        return self.phi.shape[2]

    def __repr__(self) -> str:
        """String representation of solver result"""
        return (
            f"PoissonResult(shape=({self.nz}, {self.nx}, {self.ny}), "
            f"converged={self.info.get('converged', False)}, "
            f"iterations={self.info.get('iterations', 0)})"
        )
