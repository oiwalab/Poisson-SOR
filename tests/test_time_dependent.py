"""Unit tests for TimeDependentPotential and VoltageFunction classes

Tests time-dependent voltage interpolation and potential calculation
"""

import pytest
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from structure_manager import StructureManager
from potential_interpolator import PotentialInterpolator
from time_dependent_potential import VoltageFunction, TimeDependentPotential


@pytest.fixture
def structure_manager():
    """Create StructureManager from small test config"""
    config_path = Path(__file__).parent / "test_config_small.yaml"
    return StructureManager(str(config_path))


@pytest.fixture
def potential_interpolator(structure_manager):
    """Create PotentialInterpolator for testing"""
    return PotentialInterpolator(
        structure_manager,
        z_position=-10e-9,
        omega=1.8,
        tolerance=1e-5,
        max_iterations=5000,
        verbose=False
    )


class TestVoltageFunction:
    """Test VoltageFunction class"""

    def test_constant_voltage(self):
        """Test constant voltage"""
        vf = VoltageFunction(0.5)
        assert vf(0) == 0.5
        assert vf(1e-9) == 0.5
        assert vf(1e6) == 0.5
        assert vf.t_range is None

    def test_analytical_function(self):
        """Test analytical function"""
        vf = VoltageFunction(lambda t: 0.5 * t)
        assert vf(0) == 0.0
        assert vf(2.0) == pytest.approx(1.0)
        assert vf.t_range is None

    def test_discrete_data_linear(self):
        """Test discrete data with linear interpolation"""
        t = np.array([0, 1e-9, 2e-9])
        V = np.array([0.0, 1.0, 0.0])
        vf = VoltageFunction((t, V), kind='linear')

        assert vf(0) == 0.0
        assert vf(1e-9) == 1.0
        assert vf(2e-9) == 0.0
        assert vf(0.5e-9) == pytest.approx(0.5)  # Linear interpolation
        assert vf.t_range == (0, 2e-9)

    def test_discrete_data_extrapolation(self):
        """Test extrapolation behavior (constant)"""
        t = np.array([1e-9, 2e-9])
        V = np.array([0.5, 1.0])
        vf = VoltageFunction((t, V))

        # Before range: use first value
        assert vf(0) == 0.5

        # After range: use last value
        assert vf(3e-9) == 1.0

    def test_invalid_data(self):
        """Test error handling for invalid data"""
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            VoltageFunction((np.array([0, 1]), np.array([0, 1, 2])))

        # Too few points
        with pytest.raises(ValueError, match="at least 2"):
            VoltageFunction((np.array([0]), np.array([0])))

        # Non-monotonic time
        with pytest.raises(ValueError, match="strictly increasing"):
            VoltageFunction((np.array([0, 2, 1]), np.array([0, 1, 0])))

    def test_repr(self):
        """Test string representation"""
        vf1 = VoltageFunction(0.5)
        assert "constant" in repr(vf1)

        vf2 = VoltageFunction(lambda t: t)
        assert "analytical" in repr(vf2)

        vf3 = VoltageFunction((np.array([0, 1e-9]), np.array([0, 1])))
        assert "discrete" in repr(vf3)


class TestTimeDependentPotentialInit:
    """Test TimeDependentPotential initialization"""

    def test_init_constant_voltages(self, potential_interpolator):
        """Test initialization with constant voltages"""
        voltages = {"gate_left": 0.5, "gate_right": 1.0}

        td_pot = TimeDependentPotential(potential_interpolator, voltages)

        assert len(td_pot.electrode_names) == 2
        assert td_pot.t_range is None  # All constant

    def test_init_mixed_voltages(self, potential_interpolator):
        """Test initialization with mixed voltage types"""
        t = np.linspace(0, 2e-9, 10)
        V = np.sin(2*np.pi*1e9*t)

        voltages = {
            "gate_left": (t, V),  # Discrete
            "gate_right": lambda t: 0.5  # Analytical
        }

        td_pot = TimeDependentPotential(potential_interpolator, voltages)

        assert td_pot.t_range == (0, 2e-9)

    def test_init_electrode_validation(self, potential_interpolator):
        """Test electrode name validation"""
        # Missing electrode
        with pytest.raises(ValueError, match="Missing"):
            TimeDependentPotential(potential_interpolator, {"gate_left": 0.5})

        # Extra electrode
        with pytest.raises(ValueError, match="Extra"):
            TimeDependentPotential(
                potential_interpolator,
                {"gate_left": 0.5, "gate_right": 1.0, "extra": 0.0}
            )

    def test_init_no_validation(self, potential_interpolator):
        """Test initialization without validation"""
        voltages = {"gate_left": 0.5}  # Missing gate_right

        # Should work with validate_electrodes=False
        td_pot = TimeDependentPotential(
            potential_interpolator,
            voltages,
            validate_electrodes=False
        )

        assert len(td_pot.voltage_functions) == 1


class TestTimeDependentPotentialComputation:
    """Test potential computation"""

    def test_constant_matches_interpolator(self, potential_interpolator):
        """Test that constant voltages match PotentialInterpolator"""
        voltages = {"gate_left": 0.5, "gate_right": 1.0}

        td_pot = TimeDependentPotential(potential_interpolator, voltages)

        # Get potential at any time
        phi_td = td_pot(t=1e-9)

        # Compare with direct interpolator
        phi_direct = potential_interpolator(voltages)

        np.testing.assert_allclose(phi_td, phi_direct, atol=1e-10)

    def test_time_dependent_voltage(self, potential_interpolator):
        """Test time-dependent voltage calculation"""
        voltages = {
            "gate_left": lambda t: 0.5 + 0.3 * np.sin(2*np.pi*1e9*t),
            "gate_right": 1.0
        }

        td_pot = TimeDependentPotential(potential_interpolator, voltages)

        # Test at t=0
        phi_0 = td_pot(t=0)
        voltages_0 = {"gate_left": 0.5, "gate_right": 1.0}
        phi_direct_0 = potential_interpolator(voltages_0)
        np.testing.assert_allclose(phi_0, phi_direct_0, atol=1e-10)

        # Test at t=0.25/f (quarter period, sin=1)
        phi_quarter = td_pot(t=0.25e-9)
        voltages_quarter = {"gate_left": 0.8, "gate_right": 1.0}
        phi_direct_quarter = potential_interpolator(voltages_quarter)
        np.testing.assert_allclose(phi_quarter, phi_direct_quarter, atol=1e-10)

    def test_get_voltage_at_time(self, potential_interpolator):
        """Test get_voltage_at_time method"""
        voltages = {
            "gate_left": lambda t: 2*t,  # Linear
            "gate_right": 0.5  # Constant
        }

        td_pot = TimeDependentPotential(potential_interpolator, voltages)

        V_0 = td_pot.get_voltage_at_time(0)
        assert V_0["gate_left"] == 0.0
        assert V_0["gate_right"] == 0.5

        V_1 = td_pot.get_voltage_at_time(1.0)
        assert V_1["gate_left"] == 2.0
        assert V_1["gate_right"] == 0.5

    def test_get_time_series(self, potential_interpolator):
        """Test get_time_series method"""
        voltages = {
            "gate_left": lambda t: 0.5 * np.sin(2*np.pi*1e9*t),
            "gate_right": 1.0
        }

        td_pot = TimeDependentPotential(potential_interpolator, voltages)

        t_array = np.linspace(0, 1e-9, 10)
        phi_series = td_pot.get_time_series(t_array, progress=False)

        assert phi_series.shape == (10, potential_interpolator.nx, potential_interpolator.ny)

        # Check that each time point matches individual computation
        for i, t in enumerate(t_array):
            phi_individual = td_pot(t)
            np.testing.assert_allclose(phi_series[i], phi_individual, atol=1e-10)


class TestTimeDependentPotentialRepr:
    """Test string representation"""

    def test_repr(self, potential_interpolator):
        """Test __repr__ method"""
        voltages = {
            "gate_left": lambda t: t,  # Analytical
            "gate_right": 0.5  # Constant
        }

        td_pot = TimeDependentPotential(potential_interpolator, voltages)

        repr_str = repr(td_pot)

        assert "TimeDependentPotential" in repr_str
        assert "electrodes: 2" in repr_str
        assert "analytical: 1" in repr_str
        assert "constant: 1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
