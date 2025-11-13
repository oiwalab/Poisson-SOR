"""
Tests for SchrodingerSolver2D

Validates the 2D Schrödinger equation solver using analytical solutions
for various test cases.
"""

import numpy as np
import pytest
from scipy.constants import hbar, electron_mass
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from schrodinger import SchrodingerSolver2D


class TestSchrodingerSolver2DInit:
    """Test initialization and parameter validation"""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters"""
        V = np.zeros((30, 30))
        solver = SchrodingerSolver2D(V, 1e-9, 1e-9, 0.19 * electron_mass)

        assert solver.nx == 30
        assert solver.ny == 30
        assert solver.dx == 1e-9
        assert solver.dy == 1e-9
        assert solver.effective_mass == 0.19 * electron_mass

    def test_init_invalid_potential_shape(self):
        """Test that 1D or 3D potential raises error"""
        V_1d = np.zeros(30)
        with pytest.raises(ValueError, match="Potential must be 2D"):
            SchrodingerSolver2D(V_1d, 1e-9, 1e-9, electron_mass)

    def test_init_invalid_grid_spacing(self):
        """Test that negative or zero grid spacing raises error"""
        V = np.zeros((30, 30))
        with pytest.raises(ValueError, match="Grid spacing must be positive"):
            SchrodingerSolver2D(V, -1e-9, 1e-9, electron_mass)

        with pytest.raises(ValueError, match="Grid spacing must be positive"):
            SchrodingerSolver2D(V, 1e-9, 0, electron_mass)

    def test_init_invalid_mass(self):
        """Test that negative or zero mass raises error"""
        V = np.zeros((30, 30))
        with pytest.raises(ValueError, match="Effective mass must be positive"):
            SchrodingerSolver2D(V, 1e-9, 1e-9, -electron_mass)

    def test_grid_too_small_for_dirichlet(self):
        """Test that grid smaller than 3x3 raises error"""
        V = np.zeros((2, 2))
        solver = SchrodingerSolver2D(V, 1e-9, 1e-9, electron_mass)
        with pytest.raises(ValueError, match="Grid too small"):
            solver._build_hamiltonian()


class TestHarmonicOscillator2D:
    """Test 2D isotropic harmonic oscillator"""

    def test_harmonic_oscillator_2d(self):
        """
        Test 2D isotropic harmonic oscillator.

        V(x,y) = 0.5 * m * ω² * (x² + y²)

        Analytical energy levels:
        E_{nx,ny} = ℏω(nx + ny + 1)

        Ground state: E_{0,0} = ℏω
        First excited (degenerate): E_{1,0} = E_{0,1} = 2ℏω
        """
        # Parameters
        n = 30  # Grid size
        L = 40e-9  # 40 nm box
        dx = dy = L / (n - 1)

        m_eff = 0.19 * electron_mass
        omega = 1e13  # rad/s

        # Create 2D parabolic potential
        x = np.linspace(-L / 2, L / 2, n)
        y = np.linspace(-L / 2, L / 2, n)
        X, Y = np.meshgrid(x, y, indexing="ij")

        k = m_eff * omega**2
        V_joules = 0.5 * k * (X**2 + Y**2)
        V_volts = V_joules / (1.602176634e-19)

        # Solve for first 3 states
        solver = SchrodingerSolver2D(V_volts, dx, dy, m_eff)
        energies, wavefunctions = solver.solve(n_states=3)

        # Analytical energies
        # n=0: (0,0) -> E = ℏω
        # n=1: (1,0) or (0,1) -> E = 2ℏω (degenerate)
        # n=2: (2,0), (0,2), or (1,1) -> E = 3ℏω (degenerate)
        E_analytical = hbar * omega * np.array([1, 2, 2]) / (1.602176634e-19)  # eV

        # Compare ground state (well-defined)
        rel_error_ground = np.abs(energies[0] - E_analytical[0]) / E_analytical[0]
        assert (
            rel_error_ground < 0.05
        ), f"Ground state error {rel_error_ground:.3%} exceeds 5%"

        # Check first excited state (degenerate, may have numerical splitting)
        # Average should match analytical
        E_first_excited_avg = np.mean(energies[1:3])
        rel_error_excited = (
            np.abs(E_first_excited_avg - E_analytical[1]) / E_analytical[1]
        )
        assert (
            rel_error_excited < 0.30
        ), f"First excited state error {rel_error_excited:.3%} exceeds 30%"


class TestParticleInBox:
    """Test infinite square well (particle in a box)"""

    def test_particle_in_box_2d(self):
        """
        Test 2D infinite square well.

        V = 0 inside box, V = ∞ at boundaries (Dirichlet BC)

        Analytical energy levels:
        E_{nx,ny} = (ℏ²π²/2m) * (nx²/Lx² + ny²/Ly²)

        where nx, ny = 1, 2, 3, ... (quantum numbers)
        """
        # Parameters
        n = 30
        Lx = Ly = 30e-9  # 30nm square box
        dx = Lx / (n - 1)
        dy = Ly / (n - 1)

        m_eff = 0.19 * electron_mass

        # Flat potential (infinite walls are Dirichlet BC)
        V = np.zeros((n, n))

        # Solve
        solver = SchrodingerSolver2D(V, dx, dy, m_eff)
        energies, wavefunctions = solver.solve(n_states=5)

        # Analytical energies (quantum numbers: nx, ny = 1, 2, 3, ...)
        # Note: Dirichlet BC means ψ=0 at boundaries, so effective box size
        # is approximately (n-2)*dx ≈ Lx
        L_eff_x = (n - 2) * dx
        L_eff_y = (n - 2) * dy

        def energy_particle_in_box(nx_qn, ny_qn, Lx, Ly, m):
            """Analytical energy for 2D particle in a box"""
            return (hbar**2 * np.pi**2 / (2 * m)) * (
                nx_qn**2 / Lx**2 + ny_qn**2 / Ly**2
            )

        # Ground state: (nx=1, ny=1)
        E_11 = (
            energy_particle_in_box(1, 1, L_eff_x, L_eff_y, m_eff) / 1.602176634e-19
        )
        # First excited (degenerate): (nx=1, ny=2) and (nx=2, ny=1)
        E_12 = (
            energy_particle_in_box(1, 2, L_eff_x, L_eff_y, m_eff) / 1.602176634e-19
        )
        E_21 = (
            energy_particle_in_box(2, 1, L_eff_x, L_eff_y, m_eff) / 1.602176634e-19
        )

        # Check ground state
        rel_error_ground = np.abs(energies[0] - E_11) / E_11
        assert (
            rel_error_ground < 0.10
        ), f"Ground state error {rel_error_ground:.3%} exceeds 10%"

        # Check first excited states (degenerate for square box)
        E_excited_avg = np.mean(energies[1:3])
        E_analytical_excited = (E_12 + E_21) / 2
        rel_error_excited = (
            np.abs(E_excited_avg - E_analytical_excited) / E_analytical_excited
        )
        assert (
            rel_error_excited < 0.25
        ), f"Excited state error {rel_error_excited:.3%} exceeds 25%"


class TestWavefunctionProperties:
    """Test wavefunction normalization and orthogonality"""

    def test_wavefunction_normalization(self):
        """Test that all wavefunctions are normalized: ∫|ψ|²dxdy = 1"""
        # Simple harmonic oscillator
        n = 30
        L = 40e-9
        dx = dy = L / (n - 1)
        m_eff = 0.19 * electron_mass
        omega = 1e13

        x = np.linspace(-L / 2, L / 2, n)
        y = np.linspace(-L / 2, L / 2, n)
        X, Y = np.meshgrid(x, y, indexing="ij")

        k = m_eff * omega**2
        V_volts = 0.5 * k * (X**2 + Y**2) / (1.602176634e-19)

        solver = SchrodingerSolver2D(V_volts, dx, dy, m_eff)
        energies, wavefunctions = solver.solve(n_states=5)

        # Check normalization for each state
        for i, psi in enumerate(wavefunctions):
            norm = np.sum(np.abs(psi) ** 2) * dx * dy
            assert np.abs(norm - 1.0) < 1e-6, f"State {i} norm = {norm}, expected 1.0"

    def test_wavefunction_orthogonality(self):
        """Test that wavefunctions are orthogonal: <ψ_i|ψ_j> = δ_ij"""
        n = 30
        L = 40e-9
        dx = dy = L / (n - 1)
        m_eff = 0.19 * electron_mass
        omega = 1e13

        x = np.linspace(-L / 2, L / 2, n)
        y = np.linspace(-L / 2, L / 2, n)
        X, Y = np.meshgrid(x, y, indexing="ij")

        k = m_eff * omega**2
        V_volts = 0.5 * k * (X**2 + Y**2) / (1.602176634e-19)

        solver = SchrodingerSolver2D(V_volts, dx, dy, m_eff)
        energies, wavefunctions = solver.solve(n_states=5)

        # Check orthogonality for all pairs
        for i in range(len(wavefunctions)):
            for j in range(len(wavefunctions)):
                overlap = (
                    np.sum(np.conj(wavefunctions[i]) * wavefunctions[j]) * dx * dy
                )

                if i == j:
                    # Diagonal: should be 1 (normalized)
                    assert (
                        np.abs(overlap - 1.0) < 1e-6
                    ), f"<ψ_{i}|ψ_{i}> = {overlap}, expected 1.0"
                else:
                    # Off-diagonal: should be 0 (orthogonal)
                    assert (
                        np.abs(overlap) < 1e-8
                    ), f"<ψ_{i}|ψ_{j}> = {overlap}, expected 0.0"

    def test_probability_density(self):
        """Test probability density computation"""
        V = np.zeros((30, 30))
        solver = SchrodingerSolver2D(V, 1e-9, 1e-9, 0.19 * electron_mass)
        energies, psi = solver.solve(n_states=1)

        prob_density = solver.compute_probability_density(psi[0])

        # Check that probability density is non-negative
        assert np.all(prob_density >= 0), "Probability density has negative values"

        # Check normalization
        total_prob = np.sum(prob_density) * solver.dx * solver.dy
        assert np.abs(total_prob - 1.0) < 1e-6, f"Total probability = {total_prob}"


class TestSaveLoad:
    """Test save and load functionality"""

    def test_save_load(self, tmp_path):
        """Test that solver can be saved and loaded"""
        # Create solver with deterministic potential (harmonic oscillator)
        n = 25
        L = 30e-9
        dx = dy = L / (n - 1)
        m_eff = 0.19 * electron_mass
        omega = 1e13

        # Create parabolic potential (stable, non-degenerate)
        x = np.linspace(-L / 2, L / 2, n)
        y = np.linspace(-L / 2, L / 2, n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        k = m_eff * omega**2
        V_volts = 0.5 * k * (X**2 + Y**2) / (1.602176634e-19)

        solver = SchrodingerSolver2D(V_volts, dx, dy, m_eff)

        # Solve for reference (ground state only to avoid degeneracy issues)
        energies_orig, psi_orig = solver.solve(n_states=1)

        # Save
        save_path = tmp_path / "test_solver.npz"
        solver.save(str(save_path))

        # Load
        solver_loaded = SchrodingerSolver2D.load(str(save_path))

        # Check parameters
        assert solver_loaded.nx == solver.nx
        assert solver_loaded.ny == solver.ny
        assert solver_loaded.dx == solver.dx
        assert solver_loaded.dy == solver.dy
        assert solver_loaded.effective_mass == solver.effective_mass
        np.testing.assert_allclose(
            solver_loaded.potential, solver.potential, rtol=1e-10
        )

        # Solve again and compare
        energies_loaded, psi_loaded = solver_loaded.solve(n_states=1)

        np.testing.assert_allclose(energies_loaded, energies_orig, rtol=1e-10)
        # Wavefunctions may differ by global phase, compare probabilities
        prob_orig = np.abs(psi_orig[0]) ** 2
        prob_loaded = np.abs(psi_loaded[0]) ** 2
        # Relative tolerance for probability density (allows small numerical differences)
        np.testing.assert_allclose(prob_orig, prob_loaded, rtol=1e-5)


class TestExpectationValue:
    """Test expectation value computation"""

    def test_expectation_value_position(self):
        """Test expectation value of position for symmetric potential"""
        # Centered harmonic oscillator - expect <x> = <y> = 0
        n = 30
        L = 40e-9
        dx = dy = L / (n - 1)
        m_eff = 0.19 * electron_mass
        omega = 1e13

        x = np.linspace(-L / 2, L / 2, n)
        y = np.linspace(-L / 2, L / 2, n)
        X, Y = np.meshgrid(x, y, indexing="ij")

        k = m_eff * omega**2
        V_volts = 0.5 * k * (X**2 + Y**2) / (1.602176634e-19)

        solver = SchrodingerSolver2D(V_volts, dx, dy, m_eff)
        energies, psi = solver.solve(n_states=1)

        # Ground state should be centered at origin
        x_expect = solver.compute_expectation_value(psi[0], X)
        y_expect = solver.compute_expectation_value(psi[0], Y)

        # Should be near zero (within numerical precision)
        assert np.abs(x_expect) < 1e-10, f"<x> = {x_expect}, expected ~0"
        assert np.abs(y_expect) < 1e-10, f"<y> = {y_expect}, expected ~0"
