"""Material properties database and management

Provides material parameters for heterostructure band structure calculations
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Material:
    """Material properties container

    Parameters
    ----------
    name : str
        Material name
    epsilon_r : float
        Relative permittivity (dimensionless)
    electron_affinity : float
        Electron affinity χ (eV) - energy from vacuum level to conduction band edge
    band_gap : float
        Band gap Eg (eV)
    effective_mass_e : float, optional
        Electron effective mass (in units of m0, electron rest mass)
    effective_mass_h : float, optional
        Hole effective mass (in units of m0, electron rest mass)

    Notes
    -----
    Band structure reference:
    - Vacuum level: E = 0 (reference)
    - Conduction band edge: Ec = -χ (eV)
    - Valence band edge: Ev = Ec - Eg (eV)
    """

    name: str
    epsilon_r: float
    electron_affinity: float
    band_gap: float
    effective_mass_e: Optional[float] = None
    effective_mass_h: Optional[float] = None

    def __post_init__(self):
        """Validate material parameters"""
        if self.epsilon_r <= 0:
            raise ValueError(f"epsilon_r must be positive, got {self.epsilon_r}")
        if self.band_gap < 0:
            raise ValueError(f"band_gap must be non-negative, got {self.band_gap}")


# Material database (at cryogenic temperature ~4K)
MATERIAL_DATABASE: Dict[str, Dict] = {
    "Si": {
        "epsilon_r": 11.7,
        "electron_affinity": 4.05,  # eV at 4K
        "band_gap": 1.12,  # eV at 4K (indirect gap)
        "effective_mass_e": 0.26,  # Longitudinal effective mass (m0 units)
        "effective_mass_h": 0.36,  # Heavy hole effective mass (m0 units)
    },
    "SiO2": {
        "epsilon_r": 3.9,
        "electron_affinity": 0.9,  # eV
        "band_gap": 9.0,  # eV (insulator)
        "effective_mass_e": None,  # Not relevant for insulator
        "effective_mass_h": None,
    },
    # Placeholder for future materials
    # "SiGe": {...},
    # "GaAs": {...},
}


def get_material(name: str, overrides: Optional[Dict] = None) -> Material:
    """Get material from database with optional parameter overrides

    Parameters
    ----------
    name : str
        Material name (must exist in MATERIAL_DATABASE)
    overrides : Dict, optional
        Dictionary of parameters to override from database
        Supported keys: epsilon_r, electron_affinity, band_gap,
                       effective_mass_e, effective_mass_h

    Returns
    -------
    material : Material
        Material object with specified parameters

    Raises
    ------
    ValueError
        If material name not found in database

    Examples
    --------
    >>> si = get_material("Si")
    >>> si.band_gap
    1.12

    >>> si_custom = get_material("Si", {"band_gap": 1.15})
    >>> si_custom.band_gap
    1.15
    """
    if name not in MATERIAL_DATABASE:
        available = ", ".join(MATERIAL_DATABASE.keys())
        raise ValueError(
            f"Material '{name}' not found in database. "
            f"Available materials: {available}"
        )

    # Start with database defaults
    params = MATERIAL_DATABASE[name].copy()

    # Apply overrides if provided
    if overrides is not None:
        for key, value in overrides.items():
            if key in params:
                params[key] = value
            else:
                # Warn but allow unknown keys (for future extensibility)
                pass

    # Create Material object
    return Material(name=name, **params)


def list_materials() -> list:
    """List all available materials in database

    Returns
    -------
    materials : list
        List of material names
    """
    return list(MATERIAL_DATABASE.keys())
