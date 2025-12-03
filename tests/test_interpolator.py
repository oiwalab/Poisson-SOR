"""Unit tests for PotentialInterpolator class

Tests linear interpolation of 2D potential distributions
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from structure import StructureManager
from poisson import PoissonSolver
from potential_interpolator import PotentialInterpolator


@pytest.fixture
def small_config():
    """Path to small test configuration (fast, ~21x21x21 grid)"""
    config_path = Path(__file__).parent / "test_config_small.yaml"
    return str(config_path)


@pytest.fixture
def structure_manager(small_config):
    """Create StructureManager from small test config"""
    manager = StructureManager(small_config)
    return manager


# Common solver parameters for tests
SOLVER_PARAMS = {"omega": 1.8, "tolerance": 1e-5, "max_iterations": 5000}


class TestPotentialInterpolatorInit:
    """Test initialization of PotentialInterpolator"""

    def test_init_with_z_position(self, structure_manager):
        """Test initialization with z-coordinate"""
        manager = structure_manager

        # Create interpolator at Si/SiO2 interface (z=-10nm for small config)
        interp = PotentialInterpolator(
            manager,
            z_position=-10e-9,
            **SOLVER_PARAMS,
            verbose=False
        )

        assert interp.z_position == pytest.approx(-10e-9, abs=1e-9)
        assert 0 <= interp.z_index < manager.nz
        assert interp.n_electrodes == 2  # test_config_small.yaml has 2 electrodes
        assert interp.electrode_names == ["gate_left", "gate_right"]
        assert interp.basis_phi.shape == (2, manager.nx, manager.ny)
        assert interp.particular_band_edge.shape == (manager.nx, manager.ny)

    def test_init_at_surface(self, structure_manager):
        """Test initialization at surface (z=0)"""
        manager = structure_manager

        interp = PotentialInterpolator(manager, z_position=0.0, verbose=False, **SOLVER_PARAMS)

        assert interp.z_position == 0.0
        assert interp.z_index == 0

    def test_init_at_bottom(self, structure_manager):
        """Test initialization at domain bottom"""
        manager = structure_manager

        # Bottom is at z = -size_z
        z_bottom = -manager.size_z
        interp = PotentialInterpolator(
            manager, z_position=z_bottom, verbose=False
        )

        assert interp.z_position == pytest.approx(z_bottom, abs=1e-9)
        assert interp.z_index == manager.nz - 1

    def test_init_out_of_bounds(self, structure_manager):
        """Test that out-of-bounds z raises ValueError"""
        manager = structure_manager

        # Too negative
        with pytest.raises(ValueError, match="outside domain"):
            PotentialInterpolator(manager, z_position=-200e-9, verbose=False, **SOLVER_PARAMS)

        # Too positive
        with pytest.raises(ValueError, match="outside domain"):
            PotentialInterpolator(manager, z_position=10e-9, verbose=False, **SOLVER_PARAMS)

    def test_init_with_charge_density(self, structure_manager):
        """Test initialization with non-zero charge density"""
        manager = structure_manager

        # Create uniform charge density
        rho = np.ones((manager.nz, manager.nx, manager.ny)) * 1e15  # C/m^3

        interp = PotentialInterpolator(
            manager, z_position=-10e-9, charge_density=rho, verbose=False
        )

        # Particular band edge should be non-zero
        assert np.max(np.abs(interp.particular_band_edge)) > 1e-10


class TestPotentialInterpolatorInterpolation:
    """Test interpolation functionality"""

    def test_interpolate_all_zero(self, structure_manager):
        """Test interpolation with all voltages zero"""
        manager = structure_manager

        interp = PotentialInterpolator(
            manager, z_position=-10e-9, verbose=False
        )

        voltages = {"gate_left": 0.0, "gate_right": 0.0}
        band_edge = interp.interpolate(voltages)

        # Should equal particular solution (already as band edge)
        assert band_edge.shape == (manager.nx, manager.ny)
        np.testing.assert_allclose(band_edge, interp.particular_band_edge, atol=1e-10)

    def test_interpolate_single_electrode(self, structure_manager):
        """Test interpolation with single electrode at 1V"""
        manager = structure_manager

        interp = PotentialInterpolator(
            manager, z_position=-10e-9, verbose=False
        )

        # Only first electrode at 1V
        voltages = {"gate_left": 1.0, "gate_right": 0.0}
        band_edge = interp.interpolate(voltages)

        # Should equal particular_band_edge - 1.0 * basis_phi[0]
        # (Note: negative sign because Ec = -φ - χ, so ΔEc = -Δφ)
        expected_band_edge = interp.particular_band_edge - 1.0 * interp.basis_phi[0]
        np.testing.assert_allclose(band_edge, expected_band_edge, atol=1e-10)

    def test_linearity_scaling(self, structure_manager):
        """Test linearity: Ec(2V) scaling for single electrode"""
        manager = structure_manager

        interp = PotentialInterpolator(
            manager, z_position=-10e-9, verbose=False
        )

        # Compute for 1V
        voltages_1V = {"gate_left": 1.0, "gate_right": 0.0}
        Ec_1V = interp(voltages_1V)

        # Compute for 2V
        voltages_2V = {"gate_left": 2.0, "gate_right": 0.0}
        Ec_2V = interp(voltages_2V)

        # Check linearity: Ec(2V) = Ec_particular - 2.0 * φ_basis[0]
        expected_Ec = interp.particular_band_edge - 2.0 * interp.basis_phi[0]
        np.testing.assert_allclose(Ec_2V, expected_Ec, atol=1e-10)

        # Also verify linearity: Ec(2V) - Ec_particular = 2 * (Ec(1V) - Ec_particular)
        np.testing.assert_allclose(
            Ec_2V - interp.particular_band_edge,
            2.0 * (Ec_1V - interp.particular_band_edge),
            atol=1e-10
        )

    def test_superposition(self, structure_manager):
        """Test superposition principle for band edges"""
        manager = structure_manager

        interp = PotentialInterpolator(
            manager, z_position=-10e-9, verbose=False
        )

        # Compute individual solutions
        V1 = {"gate_left": 0.5, "gate_right": 0.0}
        V2 = {"gate_left": 0.0, "gate_right": 1.0}
        Ec_1 = interp(V1)
        Ec_2 = interp(V2)

        # Compute combined solution
        V_combined = {"gate_left": 0.5, "gate_right": 1.0}
        Ec_combined = interp(V_combined)

        # Check superposition (need to add particular solution back to avoid double subtraction)
        # Ec(V1+V2) = Ec(V1) + Ec(V2) - Ec_particular
        expected = Ec_1 + Ec_2 - interp.particular_band_edge
        np.testing.assert_allclose(Ec_combined, expected, atol=1e-10)

    def test_call_method(self, structure_manager):
        """Test __call__ method is equivalent to interpolate"""
        manager = structure_manager

        interp = PotentialInterpolator(
            manager, z_position=-10e-9, verbose=False
        )

        voltages = {"gate_left": 0.5, "gate_right": 1.0}

        phi_interpolate = interp.interpolate(voltages)
        phi_call = interp(voltages)

        np.testing.assert_array_equal(phi_interpolate, phi_call)

    def test_missing_electrode_raises_error(self, structure_manager):
        """Test that missing electrode in voltages raises ValueError"""
        manager = structure_manager

        interp = PotentialInterpolator(
            manager, z_position=-10e-9, verbose=False
        )

        # Missing gate_right
        voltages = {"gate_left": 0.5}

        with pytest.raises(ValueError, match="Missing"):
            interp(voltages)

    def test_extra_electrode_raises_error(self, structure_manager):
        """Test that extra electrode in voltages raises ValueError"""
        manager = structure_manager

        interp = PotentialInterpolator(
            manager, z_position=-10e-9, verbose=False
        )

        # Extra electrode
        voltages = {
            "gate_left": 0.5,
            "gate_right": 1.0,
            "extra_gate": 1.0,
        }

        with pytest.raises(ValueError, match="Extra"):
            interp(voltages)


class TestPotentialInterpolatorAccuracy:
    """Test accuracy by comparing with direct solver"""

    def test_accuracy_vs_direct_solve(self, structure_manager):
        """Compare interpolation with direct Poisson solver

        PotentialInterpolator with carrier_type='electron' returns Ec, not phi
        """
        manager = structure_manager

        # Create interpolator (carrier_type='electron' is default, returns Ec)
        interp = PotentialInterpolator(
            manager, z_position=-10e-9, carrier_type="electron",
            **SOLVER_PARAMS, verbose=False
        )

        # Test voltages
        test_voltages = {"gate_left": 0.5, "gate_right": 1.0}

        # Get interpolated result (returns Ec for electrons)
        Ec_interp = interp(test_voltages)

        # Solve directly with same voltages
        for i, (name, voltage) in enumerate(test_voltages.items()):
            manager.electrodes[i]["voltage"] = voltage
        manager.get_electrode_voltages()

        solver = PoissonSolver(manager, **SOLVER_PARAMS)
        result_direct = solver.solve(rho=None, verbose=False)
        Ec_direct = result_direct.Ec[interp.z_index, :, :]  # Get Ec, not phi

        # Should be very close (within solver tolerance)
        # Looser tolerance due to iterative solver convergence differences
        np.testing.assert_allclose(Ec_interp, Ec_direct, atol=1e-5, rtol=1e-5)

    def test_accuracy_different_voltages(self, structure_manager):
        """Test accuracy for different voltage combinations

        PotentialInterpolator with carrier_type='electron' returns Ec, not phi
        """
        manager = structure_manager

        # Create interpolator (carrier_type='electron' is default, returns Ec)
        interp = PotentialInterpolator(
            manager, z_position=-10e-9, carrier_type="electron",
            **SOLVER_PARAMS, verbose=False
        )

        # Test multiple voltage combinations
        voltage_sets = [
            {"gate_left": 0.0, "gate_right": 0.0},
            {"gate_left": 1.0, "gate_right": 1.0},
            {"gate_left": -0.5, "gate_right": 0.5},
            {"gate_left": 0.3, "gate_right": 0.7},
        ]

        for voltages in voltage_sets:
            # Interpolated result (returns Ec for electrons)
            Ec_interp = interp(voltages)

            # Direct solve
            for i, (name, voltage) in enumerate(voltages.items()):
                manager.electrodes[i]["voltage"] = voltage
            manager.get_electrode_voltages()

            solver = PoissonSolver(manager, **SOLVER_PARAMS)
            result_direct = solver.solve(rho=None, verbose=False)
            Ec_direct = result_direct.Ec[interp.z_index, :, :]  # Get Ec, not phi

            # Compare (looser tolerance due to solver convergence)
            np.testing.assert_allclose(
                Ec_interp, Ec_direct, atol=1e-5, rtol=1e-5,
                err_msg=f"Failed for voltages: {voltages}"
            )


class TestPotentialInterpolatorSaveLoad:
    """Test save/load functionality"""

    def test_save_and_load(self, structure_manager):
        """Test saving and loading interpolator"""
        manager = structure_manager

        # Create interpolator
        interp_original = PotentialInterpolator(
            manager, z_position=-10e-9, verbose=False
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            interp_original.save(filepath)

            # Load
            interp_loaded = PotentialInterpolator.load(filepath)

            # Check attributes match
            assert interp_loaded.z_position == interp_original.z_position
            assert interp_loaded.z_index == interp_original.z_index
            assert interp_loaded.electrode_names == interp_original.electrode_names
            assert interp_loaded.n_electrodes == interp_original.n_electrodes
            np.testing.assert_array_equal(interp_loaded.x, interp_original.x)
            np.testing.assert_array_equal(interp_loaded.y, interp_original.y)
            np.testing.assert_array_equal(
                interp_loaded.basis_phi, interp_original.basis_phi
            )
            np.testing.assert_array_equal(
                interp_loaded.particular_band_edge, interp_original.particular_band_edge
            )

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_loaded_interpolator_works(self, structure_manager):
        """Test that loaded interpolator produces same results"""
        manager = structure_manager

        interp_original = PotentialInterpolator(
            manager, z_position=-10e-9, verbose=False
        )

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            interp_original.save(filepath)
            interp_loaded = PotentialInterpolator.load(filepath)

            # Test interpolation
            voltages = {"gate_left": 0.5, "gate_right": 1.0}

            phi_original = interp_original(voltages)
            phi_loaded = interp_loaded(voltages)

            np.testing.assert_array_equal(phi_original, phi_loaded)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestPotentialInterpolatorRepr:
    """Test string representation"""

    def test_repr(self, structure_manager):
        """Test __repr__ method"""
        manager = structure_manager

        interp = PotentialInterpolator(
            manager, z_position=-10e-9, verbose=False
        )

        repr_str = repr(interp)

        assert "PotentialInterpolator" in repr_str
        assert "z_position" in repr_str
        assert "z_index" in repr_str
        assert "n_electrodes=2" in repr_str  # test_config_small has 2 electrodes
        assert f"grid=({manager.nx}, {manager.ny})" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
