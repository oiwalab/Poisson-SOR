"""Solver result container with band structure information

Stores potential and materials, computes band edges on demand
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from pathlib import Path


@dataclass
class SolverResult:
    """Container for Poisson solver results with band structure

    Parameters
    ----------
    phi : np.ndarray
        Electrostatic potential distribution (V), shape=(nz, nx, ny)
    x : np.ndarray
        x-coordinates (m), shape=(nx,)
    y : np.ndarray
        y-coordinates (m), shape=(ny,)
    z : np.ndarray
        z-coordinates (m), shape=(nz,)
    materials : List[Material]
        Material objects for each z-layer, length=nz
        Assumes materials are uniform in x-y plane at each z
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
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    materials: Optional[
        List
    ]  # List of Material objects, one per z-layer (None if not provided)
    info: Dict

    def __post_init__(self):
        """Validate array shapes and materials list"""
        nz, nx, ny = self.phi.shape

        # Check coordinate array shapes
        if self.x.shape != (nx,):
            raise ValueError(
                f"x coordinate shape {self.x.shape} does not match nx={nx}"
            )
        if self.y.shape != (ny,):
            raise ValueError(
                f"y coordinate shape {self.y.shape} does not match ny={ny}"
            )
        if self.z.shape != (nz,):
            raise ValueError(
                f"z coordinate shape {self.z.shape} does not match nz={nz}"
            )

        # Check materials list length (if provided)
        if self.materials is not None and len(self.materials) != nz:
            raise ValueError(
                f"materials list length {len(self.materials)} does not match nz={nz}"
            )

    def compute_Ec(self) -> np.ndarray:
        """Compute conduction band edge

        Returns
        -------
        Ec : np.ndarray
            Conduction band edge (eV), shape=(nz, nx, ny)

        Raises
        ------
        ValueError
            If materials list is not available

        Notes
        -----
        Ec(r) = -q·φ(r) - χ(z) [eV]
        where φ is in V, χ is in eV
        Unit conversion: q·φ [J] / q [C] = φ [V] → φ [eV] (numerically equal)
        """
        if self.materials is None:
            raise ValueError(
                "Materials list is not available. Cannot compute band edges."
            )

        nz, nx, ny = self.phi.shape
        Ec = np.zeros((nz, nx, ny))

        for k in range(nz):
            chi = self.materials[k].electron_affinity  # eV
            # φ [V] is numerically equal to q·φ [eV]
            Ec[k, :, :] = -self.phi[k, :, :] - chi

        return Ec

    def compute_Ev(self) -> np.ndarray:
        """Compute valence band edge

        Returns
        -------
        Ev : np.ndarray
            Valence band edge (eV), shape=(nz, nx, ny)

        Raises
        ------
        ValueError
            If materials list is not available

        Notes
        -----
        Ev(r) = Ec(r) - Eg(z) [eV]
        where Eg(z) is band gap at each z-layer
        """
        if self.materials is None:
            raise ValueError(
                "Materials list is not available. Cannot compute band edges."
            )

        Ec = self.compute_Ec()
        nz, nx, ny = self.phi.shape
        Ev = np.zeros((nz, nx, ny))

        for k in range(nz):
            Eg = self.materials[k].band_gap  # eV
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

        # Extract 1D slices
        z_slice = self.z
        Ec_slice = Ec_full[:, x_idx, y_idx]
        Ev_slice = Ev_full[:, x_idx, y_idx]
        phi_slice = self.phi[:, x_idx, y_idx]

        return z_slice, Ec_slice, Ev_slice, phi_slice

    def save(self, filepath: str) -> None:
        """Save solver result to file

        Parameters
        ----------
        filepath : str
            Path to save file (.npz format)

        Notes
        -----
        Material objects are saved as material names and parameters
        to enable reconstruction
        """
        # Extract material parameters for each z-layer
        material_names = [mat.name for mat in self.materials]
        epsilon_r_list = [mat.epsilon_r for mat in self.materials]
        chi_list = [mat.electron_affinity for mat in self.materials]
        Eg_list = [mat.band_gap for mat in self.materials]

        # Save to file
        np.savez(
            filepath,
            phi=self.phi,
            x=self.x,
            y=self.y,
            z=self.z,
            material_names=material_names,
            epsilon_r=epsilon_r_list,
            electron_affinity=chi_list,
            band_gap=Eg_list,
            converged=self.info.get("converged", False),
            iterations=self.info.get("iterations", 0),
            final_phi_change=self.info.get("final_phi_change", 0.0),
        )

        print(f"Solver result saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SolverResult":
        """Load solver result from file

        Parameters
        ----------
        filepath : str
            Path to saved file (.npz format)

        Returns
        -------
        result : SolverResult
            Loaded solver result

        Notes
        -----
        Reconstructs Material objects from saved parameters
        """
        from materials import Material

        data = np.load(filepath, allow_pickle=True)

        # Load arrays
        phi = data["phi"]
        x = data["x"]
        y = data["y"]
        z = data["z"]
        material_names = data["material_names"]
        epsilon_r = data["epsilon_r"]
        electron_affinity = data["electron_affinity"]
        band_gap = data["band_gap"]

        # Reconstruct materials list (one per z-layer)
        nz = phi.shape[0]
        materials = []
        for k in range(nz):
            mat = Material(
                name=str(material_names[k]),
                epsilon_r=float(epsilon_r[k]),
                electron_affinity=float(electron_affinity[k]),
                band_gap=float(band_gap[k]),
            )
            materials.append(mat)

        # Reconstruct info dict
        info = {
            "converged": bool(data["converged"]),
            "iterations": int(data["iterations"]),
            "final_phi_change": float(data["final_phi_change"]),
        }

        return cls(
            phi=phi,
            x=x,
            y=y,
            z=z,
            materials=materials,
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
            f"SolverResult(shape=({self.nz}, {self.nx}, {self.ny}), "
            f"converged={self.info.get('converged', False)}, "
            f"iterations={self.info.get('iterations', 0)})"
        )
