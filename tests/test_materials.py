"""Test cases for material database and Material class"""

import pytest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from materials import Material, get_material, list_materials, MATERIAL_DATABASE


def test_material_creation():
    """Test Material object creation"""
    si = Material(
        name="Si",
        epsilon_r=11.7,
        electron_affinity=4.05,
        band_gap=1.12,
        effective_mass_e=0.26,
        effective_mass_h=0.36,
    )

    assert si.name == "Si"
    assert si.epsilon_r == 11.7
    assert si.electron_affinity == 4.05
    assert si.band_gap == 1.12
    assert si.effective_mass_e == 0.26
    assert si.effective_mass_h == 0.36


def test_material_validation():
    """Test Material parameter validation"""
    # Negative epsilon_r should raise error
    with pytest.raises(ValueError, match="epsilon_r must be positive"):
        Material(
            name="Invalid",
            epsilon_r=-1.0,
            electron_affinity=4.0,
            band_gap=1.0,
        )

    # Negative band_gap should raise error
    with pytest.raises(ValueError, match="band_gap must be non-negative"):
        Material(
            name="Invalid",
            epsilon_r=10.0,
            electron_affinity=4.0,
            band_gap=-1.0,
        )


def test_material_database():
    """Test that material database contains expected materials"""
    assert "Si" in MATERIAL_DATABASE
    assert "SiO2" in MATERIAL_DATABASE

    # Check Si parameters
    si_params = MATERIAL_DATABASE["Si"]
    assert si_params["epsilon_r"] == 11.7
    assert si_params["electron_affinity"] == 4.05
    assert si_params["band_gap"] == 1.12
    assert si_params["effective_mass_e"] == 0.26
    assert si_params["effective_mass_h"] == 0.36

    # Check SiO2 parameters
    sio2_params = MATERIAL_DATABASE["SiO2"]
    assert sio2_params["epsilon_r"] == 3.9
    assert sio2_params["electron_affinity"] == 0.9
    assert sio2_params["band_gap"] == 9.0


def test_get_material_from_database():
    """Test get_material() retrieves correct material from database"""
    si = get_material("Si")

    assert si.name == "Si"
    assert si.epsilon_r == 11.7
    assert si.electron_affinity == 4.05
    assert si.band_gap == 1.12
    assert si.effective_mass_e == 0.26
    assert si.effective_mass_h == 0.36

    sio2 = get_material("SiO2")
    assert sio2.name == "SiO2"
    assert sio2.epsilon_r == 3.9
    assert sio2.electron_affinity == 0.9
    assert sio2.band_gap == 9.0
    assert sio2.effective_mass_e is None
    assert sio2.effective_mass_h is None


def test_get_material_not_found():
    """Test get_material() raises error for unknown material"""
    with pytest.raises(ValueError, match="Material 'Unknown' not found"):
        get_material("Unknown")


def test_material_override():
    """Test parameter override in get_material()"""
    # Override band_gap
    si_custom = get_material("Si", {"band_gap": 1.15})
    assert si_custom.band_gap == 1.15
    assert si_custom.epsilon_r == 11.7  # Other params unchanged

    # Override multiple parameters
    si_multi = get_material(
        "Si", {"band_gap": 1.15, "electron_affinity": 4.10}
    )
    assert si_multi.band_gap == 1.15
    assert si_multi.electron_affinity == 4.10
    assert si_multi.epsilon_r == 11.7  # Unchanged


def test_list_materials():
    """Test list_materials() returns all available materials"""
    materials = list_materials()

    assert isinstance(materials, list)
    assert "Si" in materials
    assert "SiO2" in materials
    assert len(materials) >= 2


def test_material_properties_consistency():
    """Test that material properties are physically reasonable"""
    for name in list_materials():
        mat = get_material(name)

        # All materials should have positive epsilon_r
        assert mat.epsilon_r > 0, f"{name}: epsilon_r should be positive"

        # All materials should have non-negative band_gap
        assert mat.band_gap >= 0, f"{name}: band_gap should be non-negative"

        # Electron affinity should be positive (by convention)
        assert mat.electron_affinity > 0, f"{name}: electron_affinity should be positive"

        # If effective masses are defined, they should be positive
        if mat.effective_mass_e is not None:
            assert mat.effective_mass_e > 0, f"{name}: effective_mass_e should be positive"
        if mat.effective_mass_h is not None:
            assert mat.effective_mass_h > 0, f"{name}: effective_mass_h should be positive"
