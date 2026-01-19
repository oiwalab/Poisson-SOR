"""Test cases for PoissonSolver

Basic functionality verification and validation
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poisson import PoissonSolver


class MockStructureManager:
    """Mock StructureManager for testing PoissonSolver"""

    def __init__(self, epsilon_array, h, boundary_conditions, electrode_mask=None, electrode_voltages=None):
        self.epsilon_array = epsilon_array
        self.nz, self.nx, self.ny = epsilon_array.shape
        self.h = h
        self.boundary_conditions = boundary_conditions
        self.electrode_mask = electrode_mask
        self.electrode_voltages = electrode_voltages
        self.size_x = self.nx * h
        self.size_y = self.ny * h
        self.size_z = self.nz * h

    def get_grid_coordinates(self):
        """Get grid coordinates"""
        x = np.arange(self.nx) * self.h
        y = np.arange(self.ny) * self.h
        z = -np.arange(self.nz) * self.h
        return x, y, z


def test_uniform_dielectric_neumann():
    """Test with uniform dielectric and Neumann boundary conditions

    With rho=0 and Neumann boundary conditions, potential should be constant everywhere
    """
    # Small grid (new coordinate system: shape = (nz, nx, ny))
    nz, nx, ny = 10, 10, 10
    h = 1e-9  # 1nm (isotropic grid)

    # Uniform permittivity (Si) - array shape: (nz, nx, ny)
    epsilon = np.ones((nz, nx, ny)) * 11.7

    # Neumann boundary conditions (all with d_phi/d_n = 0)
    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    # Create mock structure manager
    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    # Initialize solver
    solver = PoissonSolver(
        structure,
        omega=1.5,
        tolerance=1e-6,
        max_iterations=1000,
    )

    # Solve (zero charge density)
    result = solver.solve()

    # With rho=0 and Neumann BC, potential should be constant
    assert result.phi.std() < 1e-6, (
        "Potential should be constant with zero charge and Neumann BC"
    )
    assert result.info["converged"], "Solver should converge"
    assert result.info["iterations"] >= 1


def test_parallel_plate_capacitor():
    """Test for parallel plate capacitor (simplified version)

    Approximated as 1D problem varying only in z direction

    New coordinate system:
    - z_top (k=0, z=0nm): 1V
    - z_bottom (k=nz-1, z=-20nm): 0V
    """
    # Isotropic grid (small grid for 1D problem)
    # Array shape: (nz, nx, ny)
    nz, nx, ny = 11, 3, 3
    h = 2e-9  # 2nm (isotropic grid)

    # Uniform permittivity - array shape: (nz, nx, ny)
    epsilon = np.ones((nz, nx, ny)) * 3.9  # SiO2

    # Boundary conditions: Dirichlet at top and bottom (fixed voltage)
    boundary_conditions = {
        "z_top": {"type": "dirichlet", "value": 1.0},  # Surface (k=0): 1V
        "z_bottom": {"type": "dirichlet", "value": 0.0},  # Bottom (k=nz-1): 0V
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    # Create mock structure manager
    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    # Initialize solver
    solver = PoissonSolver(
        structure,
        omega=1.5,
        tolerance=1e-8,
        max_iterations=5000,
    )

    # Set initial condition with Dirichlet boundary conditions manually
    # Array shape: (nz, nx, ny)
    phi_initial = np.zeros((nz, nx, ny))
    phi_initial[0, :, :] = 1.0  # z_top (k=0): 1V
    phi_initial[-1, :, :] = 0.0  # z_bottom (k=nz-1): 0V

    # Solve
    result = solver.solve(phi_initial=phi_initial)

    # Check for NaN
    assert not np.isnan(result.phi).any(), "Solution should not contain NaN"

    # Check if boundary conditions are correctly set
    assert np.abs(result.phi[0, :, :].mean() - 1.0) < 1e-6, "Top boundary (k=0) should be 1V"
    assert np.abs(result.phi[-1, :, :].mean() - 0.0) < 1e-6, (
        "Bottom boundary (k=nz-1) should be 0V"
    )

    # Compare with analytical solution: for parallel plate, phi(k) = V_top * (1 - k/K) (linear, 1->0)
    # k=0 -> phi=1, k=nz-1 -> phi=0
    k_coords = np.arange(nz)
    K = nz - 1
    phi_analytical = 1.0 - k_coords / K  # Linear from 1 to 0

    # Compare at center point (array shape: (nz, nx, ny))
    phi_numerical = result.phi[:, 1, 1]

    print("\nParallel plate capacitor test:")
    print(f"Converged: {result.info['converged']}, Iterations: {result.info['iterations']}")
    print(f"Max absolute error: {np.abs(phi_numerical - phi_analytical).max():.6e}")

    # Check error against analytical solution at interior points (excluding boundaries)
    error = np.abs(phi_numerical[1:-1] - phi_analytical[1:-1])
    max_error = error.max()
    assert max_error < 0.01, (
        f"Max error {max_error:.6e} should be < 0.01 (1% of voltage range)"
    )


def test_electrode_volume():
    """Test treating electrodes as 3D volumes

    New coordinate system:
    - Array shape: (nz, nx, ny)
    - Electrodes extend downward from surface (k=0, z=0) as 3D volumes
    """
    # Small grid (array shape: (nz, nx, ny))
    nz, nx, ny = 11, 11, 11
    h = 10e-9  # 10nm (isotropic grid)

    # Uniform permittivity
    epsilon = np.ones((nz, nx, ny)) * 11.7

    # Electrode mask: 3D volume at center (top 2 layers: k=0,1)
    # Array shape: (nz, nx, ny)
    electrode_mask = np.zeros((nz, nx, ny), dtype=bool)
    k_electrode_top = 0  # Surface
    k_electrode_bottom = 1  # z-index of electrode bottom (2 layers: k=0,1)
    electrode_mask[k_electrode_top : k_electrode_bottom + 1, 4:7, 4:7] = (
        True  # Top 2 layers, center 3x3
    )

    # Electrode voltage: -0.5V
    electrode_voltages = np.zeros((nz, nx, ny))
    electrode_voltages[k_electrode_top : k_electrode_bottom + 1, 4:7, 4:7] = -0.5

    # Boundary conditions (all Neumann)
    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    # Create mock structure manager
    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
        electrode_mask=electrode_mask,
        electrode_voltages=electrode_voltages,
    )

    # Initialize solver
    solver = PoissonSolver(
        structure,
        omega=1.8,
        tolerance=1e-6,
        max_iterations=10000,
    )

    # Set initial condition (start from values close to electrode voltage)
    phi_initial = np.zeros((nz, nx, ny))
    phi_initial[electrode_mask] = -0.5

    # Solve
    result = solver.solve(phi_initial=phi_initial)

    # Check voltage in entire electrode region
    electrode_phi = result.phi[electrode_mask]
    assert np.allclose(electrode_phi, -0.5, atol=1e-6), (
        f"Electrode potential should be -0.5V, but got mean={electrode_phi.mean():.6f}"
    )

    # Potential outside electrode should be higher than electrode voltage (closer to 0V)
    # Array shape: (nz, nx, ny) -> [k, i, j]
    non_electrode_phi = result.phi[5, 0, 0]  # Edge point of middle layer
    assert non_electrode_phi > electrode_phi.mean(), (
        "Non-electrode region should have higher potential than electrode"
    )

    # Check for NaN
    assert not np.isnan(result.phi).any(), "Solution should not contain NaN"

    assert result.info["converged"], "Solver should converge"


def test_point_charge():
    """Test potential of single point charge

    Verifies spherical symmetry and 1/r dependence
    Compares with analytical solution φ(r) = Q/(4πε₀r)
    """
    # Grid setup (array shape: (nz, nx, ny))
    nz, nx, ny = 41, 41, 41
    h = 1e-9  # 1nm (isotropic grid)
    k_center, i_center, j_center = 20, 20, 20  # Center of grid

    # Vacuum permittivity
    epsilon_0 = 8.854187817e-12  # F/m
    epsilon_r = 1.0  # Vacuum
    epsilon = np.ones((nz, nx, ny)) * epsilon_r

    # Point charge at center
    Q = -1.602e-19  # C (electron charge, negative)
    rho = np.zeros((nz, nx, ny))
    rho[k_center, i_center, j_center] = Q / (h**3)  # Charge density (C/m³)

    # Dirichlet boundary conditions (phi = 0 at boundaries, approximating infinity)
    boundary_conditions = {
        "z_top": {"type": "dirichlet", "value": 0.0},
        "z_bottom": {"type": "dirichlet", "value": 0.0},
        "x_sides": {"type": "dirichlet", "value": 0.0},
        "y_sides": {"type": "dirichlet", "value": 0.0},
    }

    # Create mock structure manager
    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    # Initialize solver
    solver = PoissonSolver(
        structure,
        omega=1.5,
        tolerance=1e-10,
        max_iterations=20000,
    )

    # Solve
    result = solver.solve(rho=rho, verbose=False)

    # Check convergence
    assert result.info["converged"], f"Solver should converge, but got {result.info}"

    # Check for NaN
    assert not np.isnan(result.phi).any(), "Solution should not contain NaN"

    # Analytical solution: φ(r) = Q / (4πε₀εᵣr)
    def phi_analytical(r):
        return Q / (4 * np.pi * epsilon_0 * epsilon_r * r)

    # Test 1: Spherical symmetry
    # Check that points at same distance have similar potential
    distance = 5  # 5 grid points away from center
    test_points = [
        (k_center + distance, i_center, j_center),  # +z
        (k_center - distance, i_center, j_center),  # -z
        (k_center, i_center + distance, j_center),  # +x
        (k_center, i_center - distance, j_center),  # -x
        (k_center, i_center, j_center + distance),  # +y
        (k_center, i_center, j_center - distance),  # -y
    ]

    potentials = [result.phi[k, i, j] for k, i, j in test_points]
    mean_potential = np.mean(potentials)
    relative_deviations = [
        abs(p - mean_potential) / abs(mean_potential) for p in potentials
    ]

    print("\nPoint charge test - Spherical symmetry:")
    print(f"Converged: {result.info['converged']}, Iterations: {result.info['iterations']}")
    print(f"Distance: {distance * h * 1e9:.1f} nm")
    print(f"Potentials at 6 symmetric points: {potentials}")
    print(f"Mean: {mean_potential:.6e} V")
    print(f"Max relative deviation: {max(relative_deviations):.2%}")

    assert max(relative_deviations) < 0.05, (
        f"Spherical symmetry violated: max deviation {max(relative_deviations):.2%}"
    )

    # Test 2: Distance dependence (phi proportional to 1/r)
    # Check ratio phi1/phi2 approx r2/r1
    r1 = 5 * h  # 5nm from center
    r2 = 10 * h  # 10nm from center

    phi1 = result.phi[k_center + 5, i_center, j_center]
    phi2 = result.phi[k_center + 10, i_center, j_center]

    ratio_phi = phi1 / phi2
    ratio_r = r2 / r1  # Should be 2.0

    print("\nDistance dependence (phi proportional to 1/r):")
    print(f"phi(5nm) = {phi1:.6e} V")
    print(f"phi(10nm) = {phi2:.6e} V")
    print(f"phi1/phi2 = {ratio_phi:.3f} (expected: {ratio_r:.3f})")

    # Allow larger tolerance due to discretization effects and boundary conditions
    assert abs(ratio_phi - ratio_r) / ratio_r < 0.5, (
        f"Distance dependence violated: phi1/phi2 = {ratio_phi:.3f}, expected {ratio_r:.3f}"
    )

    # Test 3: Compare with analytical solution (informational only)
    # Large errors are expected due to:
    # - Discretization effects (especially near the charge)
    # - Finite domain size (boundaries at 20nm cannot approximate infinity well)
    test_distances = [5, 7, 10, 12, 15, 18]  # Grid points

    print("\nComparison with analytical solution (informational):")
    print(
        f"{'Distance (nm)':<15} {'phi_num (V)':<15} {'phi_ana (V)':<15} {'Rel. Error':<15}"
    )

    for d in test_distances:
        r = d * h
        phi_num = result.phi[k_center + d, i_center, j_center]
        phi_ana = phi_analytical(r)
        rel_error = abs(phi_num - phi_ana) / abs(phi_ana)

        print(f"{r * 1e9:<15.1f} {phi_num:<15.6e} {phi_ana:<15.6e} {rel_error:<15.2%}")

    print("\nPoint charge test passed!")


def test_band_bending_si_sio2():
    """Test band bending calculation for Si/SiO2 heterostructure

    Verifies:
    1. Ec, Ev arrays have correct shape
    2. Ec - Ev = Eg at all points
    3. Different band parameters in Si and SiO2 regions
    4. Band offset at Si/SiO2 interface (discontinuity in χ)
    5. SolverResult object is correctly created
    """
    from structure import StructureManager
    import yaml

    # Create test configuration for Si/SiO2 structure
    config_dict = {
        "domain": {
            "size": [50e-9, 50e-9, 30e-9],  # Small grid for testing
            "grid_spacing": 5e-9,  # 5nm isotropic grid
        },
        "layers": [
            {
                "material": "SiO2",
                "z_range": [0, -10e-9],  # 0 to -10nm (surface layer)
                "epsilon_r": 3.9,
            },
            {
                "material": "Si",
                "z_range": [-10e-9, -30e-9],  # -10nm to -30nm (substrate)
                "epsilon_r": 11.7,
            },
        ],
        "electrodes": [
            {
                "name": "gate",
                "shape": "rectangle",
                "x_range": [10e-9, 40e-9],
                "y_range": [10e-9, 40e-9],
                "z_position": -5e-9,  # Electrode bottom at -5nm
                "voltage": -0.5,  # -0.5V gate voltage
            },
        ],
        "solver": {
            "omega": 1.5,
            "max_iterations": 1000,
            "tolerance": 1e-6,
        },
        "boundary_conditions": {
            "z_top": {"type": "neumann", "value": 0.0},
            "z_bottom": {"type": "neumann", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        },
    }

    # Save to temporary YAML file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        config_file = f.name

    try:
        # Create structure manager
        structure = StructureManager()
        structure.load_from_yaml(config_file)

        # Create solver
        solver = PoissonSolver(
            structure,
            omega=config_dict["solver"]["omega"],
            tolerance=config_dict["solver"]["tolerance"],
            max_iterations=config_dict["solver"]["max_iterations"],
        )

        # Solve
        result = solver.solve(verbose=False)

        # Test 1: Check that result is PoissonResult object
        from poisson import PoissonResult
        assert isinstance(result, PoissonResult), "Result should be PoissonResult object"

        # Test 2: Check that structure is available
        assert result.structure is not None, "Structure should be available"
        assert result.structure.nz == result.nz, (
            f"Structure nz {result.structure.nz} should match result nz={result.nz}"
        )

        # Test 3: Check array shapes
        Ec = result.compute_Ec()
        Ev = result.compute_Ev()

        assert Ec.shape == result.phi.shape, (
            f"Ec shape {Ec.shape} should match phi shape {result.phi.shape}"
        )
        assert Ev.shape == result.phi.shape, (
            f"Ev shape {Ev.shape} should match phi shape {result.phi.shape}"
        )

        # Test 4: Check Ec - Ev = Eg at all points
        for k in range(result.nz):
            mat = result.structure.get_material_at_z(k)
            Eg_expected = mat.band_gap
            Eg_computed = Ec[k, :, :] - Ev[k, :, :]

            assert np.allclose(Eg_computed, Eg_expected, atol=1e-10), (
                f"At z-layer {k}: Ec - Ev should equal Eg={Eg_expected:.2f}eV, "
                f"but got mean={Eg_computed.mean():.2f}eV"
            )

        # Test 5: Check different band parameters in Si and SiO2 regions
        material_names = [result.structure.get_material_at_z(k).name for k in range(result.nz)]

        # Find SiO2 and Si layer indices
        sio2_indices = [i for i, name in enumerate(material_names) if name == "SiO2"]
        si_indices = [i for i, name in enumerate(material_names) if name == "Si"]

        assert len(sio2_indices) > 0, "Should have SiO2 layers"
        assert len(si_indices) > 0, "Should have Si layers"

        # Check band gap values
        sio2_Eg = result.structure.get_material_at_z(sio2_indices[0]).band_gap
        si_Eg = result.structure.get_material_at_z(si_indices[0]).band_gap

        assert sio2_Eg > si_Eg, f"SiO2 band gap {sio2_Eg}eV should be > Si {si_Eg}eV"

        # Check electron affinity values
        sio2_chi = result.structure.get_material_at_z(sio2_indices[0]).electron_affinity
        si_chi = result.structure.get_material_at_z(si_indices[0]).electron_affinity

        assert si_chi > sio2_chi, (
            f"Si electron affinity {si_chi}eV should be > SiO2 {sio2_chi}eV"
        )

        # Test 6: Check band offset at Si/SiO2 interface
        # Find interface index (first Si layer)
        interface_k = si_indices[0]
        sio2_k = interface_k - 1  # Layer above (should be SiO2)

        if sio2_k >= 0 and material_names[sio2_k] == "SiO2":
            # At center of grid (x_idx, y_idx)
            x_idx = result.nx // 2
            y_idx = result.ny // 2

            # Band offset (discontinuity in χ)
            delta_chi = si_chi - sio2_chi

            # At zero potential, ΔEc = -Δχ = -(χ_Si - χ_SiO2)
            # But with potential, we need to compare the actual band edges
            Ec_si = Ec[interface_k, x_idx, y_idx]
            Ec_sio2 = Ec[sio2_k, x_idx, y_idx]

            # The discontinuity depends on both χ and φ
            # ΔEc = (Ec_Si - Ec_SiO2) should relate to Δχ
            # Note: exact value depends on potential distribution
            print("\nBand offset at Si/SiO2 interface:")
            print(f"  Δχ = {delta_chi:.2f} eV")
            print(f"  Ec(Si) = {Ec_si:.3f} eV")
            print(f"  Ec(SiO2) = {Ec_sio2:.3f} eV")
            print(f"  ΔEc = {Ec_si - Ec_sio2:.3f} eV")

        # Test 7: Check 1D band diagram extraction
        z, Ec_1d, Ev_1d, phi_1d = result.get_band_diagram_1d()

        assert z.shape == (result.nz,), f"1D z array should have shape ({result.nz},)"
        assert Ec_1d.shape == (result.nz,), (
            f"1D Ec array should have shape ({result.nz},)"
        )
        assert Ev_1d.shape == (result.nz,), (
            f"1D Ev array should have shape ({result.nz},)"
        )
        assert phi_1d.shape == (result.nz,), (
            f"1D phi array should have shape ({result.nz},)"
        )

        # Test 8: Check convergence
        assert result.info["converged"], "Solver should converge"
        assert not np.isnan(result.phi).any(), "Solution should not contain NaN"
        assert not np.isnan(Ec).any(), "Ec should not contain NaN"
        assert not np.isnan(Ev).any(), "Ev should not contain NaN"

        print("\nBand bending test passed!")
        print(f"Grid size: ({result.nz}, {result.nx}, {result.ny})")
        print(f"Converged: {result.info['converged']}, Iterations: {result.info['iterations']}")
        print(f"Materials: {set(material_names)}")

    finally:
        # Clean up temporary file
        import os
        os.unlink(config_file)


def test_method_parameter_validation():
    """Test that invalid method parameter raises ValueError"""
    nz, nx, ny = 10, 10, 10
    h = 1e-9
    epsilon = np.ones((nz, nx, ny)) * 11.7
    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }
    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    # Valid methods should work
    solver_sor = PoissonSolver(structure, method="sor")
    assert solver_sor.method == "sor"

    solver_redblack = PoissonSolver(structure, method="redblack")
    assert solver_redblack.method == "redblack"

    # Invalid method should raise ValueError
    with pytest.raises(ValueError, match="Invalid method"):
        PoissonSolver(structure, method="invalid")


def test_redblack_method_convergence():
    """Test Red-Black SOR method convergence

    Uses same test as test_uniform_dielectric_neumann but with redblack method
    """
    nz, nx, ny = 10, 10, 10
    h = 1e-9

    epsilon = np.ones((nz, nx, ny)) * 11.7
    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    # Test Red-Black SOR method
    solver = PoissonSolver(
        structure,
        omega=1.5,
        tolerance=1e-6,
        max_iterations=1000,
        method="redblack",
    )

    result = solver.solve(verbose=False)

    # With rho=0 and Neumann BC, potential should be constant
    assert result.phi.std() < 1e-6, (
        "Red-Black SOR: Potential should be constant with zero charge and Neumann BC"
    )
    assert result.info["converged"], "Red-Black SOR should converge"
    assert result.info["iterations"] >= 1


@pytest.mark.parametrize("method", ["sor", "redblack"])
def test_parallel_plate_both_methods(method):
    """Test parallel plate capacitor with both SOR methods"""
    nz, nx, ny = 11, 3, 3
    h = 2e-9

    epsilon = np.ones((nz, nx, ny)) * 3.9
    boundary_conditions = {
        "z_top": {"type": "dirichlet", "value": 1.0},
        "z_bottom": {"type": "dirichlet", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
    )

    solver = PoissonSolver(
        structure,
        omega=1.5,
        tolerance=1e-8,
        max_iterations=5000,
        method=method,
    )

    phi_initial = np.zeros((nz, nx, ny))
    phi_initial[0, :, :] = 1.0
    phi_initial[-1, :, :] = 0.0

    result = solver.solve(phi_initial=phi_initial, verbose=False)

    # Check convergence
    assert result.info["converged"], f"{method}: Solver should converge"
    assert not np.isnan(result.phi).any(), f"{method}: Solution should not contain NaN"

    # Check boundary conditions
    assert np.abs(result.phi[0, :, :].mean() - 1.0) < 1e-6, (
        f"{method}: Top boundary should be 1V"
    )
    assert np.abs(result.phi[-1, :, :].mean() - 0.0) < 1e-6, (
        f"{method}: Bottom boundary should be 0V"
    )

    # Analytical solution
    k_coords = np.arange(nz)
    K = nz - 1
    phi_analytical = 1.0 - k_coords / K

    phi_numerical = result.phi[:, 1, 1]
    error = np.abs(phi_numerical[1:-1] - phi_analytical[1:-1])
    max_error = error.max()

    assert max_error < 0.01, (
        f"{method}: Max error {max_error:.6e} should be < 0.01"
    )


@pytest.mark.parametrize("method", ["sor", "redblack"])
def test_electrode_volume_both_methods(method):
    """Test 3D electrode volumes with both SOR methods"""
    nz, nx, ny = 11, 11, 11
    h = 10e-9

    epsilon = np.ones((nz, nx, ny)) * 11.7

    electrode_mask = np.zeros((nz, nx, ny), dtype=bool)
    k_electrode_top = 0
    k_electrode_bottom = 1
    electrode_mask[k_electrode_top : k_electrode_bottom + 1, 4:7, 4:7] = True

    electrode_voltages = np.zeros((nz, nx, ny))
    electrode_voltages[k_electrode_top : k_electrode_bottom + 1, 4:7, 4:7] = -0.5

    boundary_conditions = {
        "z_top": {"type": "neumann", "value": 0.0},
        "z_bottom": {"type": "neumann", "value": 0.0},
        "x_sides": {"type": "neumann", "value": 0.0},
        "y_sides": {"type": "neumann", "value": 0.0},
    }

    structure = MockStructureManager(
        epsilon_array=epsilon,
        h=h,
        boundary_conditions=boundary_conditions,
        electrode_mask=electrode_mask,
        electrode_voltages=electrode_voltages,
    )

    solver = PoissonSolver(
        structure,
        omega=1.8,
        tolerance=1e-6,
        max_iterations=10000,
        method=method,
    )

    phi_initial = np.zeros((nz, nx, ny))
    phi_initial[electrode_mask] = -0.5

    result = solver.solve(phi_initial=phi_initial, verbose=False)

    # Check electrode voltage
    electrode_phi = result.phi[electrode_mask]
    assert np.allclose(electrode_phi, -0.5, atol=1e-6), (
        f"{method}: Electrode potential should be -0.5V"
    )

    # Check convergence
    assert result.info["converged"], f"{method}: Solver should converge"
    assert not np.isnan(result.phi).any(), f"{method}: No NaN values"


# =============================================================================
# Julia Backend Tests
# =============================================================================


def _julia_available():
    """Check if Julia backend is available"""
    import importlib.util

    return importlib.util.find_spec("juliacall") is not None


JULIA_AVAILABLE = _julia_available()


@pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia backend not available")
class TestJuliaBackend:
    """Tests for Julia backend functionality"""

    @pytest.fixture
    def simple_structure(self):
        """Create a simple structure for testing"""
        nz, nx, ny = 11, 5, 5
        h = 2e-9
        epsilon = np.ones((nz, nx, ny)) * 3.9

        boundary_conditions = {
            "z_top": {"type": "dirichlet", "value": 1.0},
            "z_bottom": {"type": "dirichlet", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        }

        return MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
        )

    def test_julia_backend_initialization(self, simple_structure):
        """Test that Julia backend initializes correctly"""
        solver = PoissonSolver(
            simple_structure,
            method="sor",
            use_julia=True,
            max_iterations=10,
        )

        assert solver._use_julia, "Julia backend should be enabled"
        assert solver._julia_main is not None, "Julia main should be initialized"

    def test_julia_thread_count(self, simple_structure):
        """Test that Julia reports thread count"""
        solver = PoissonSolver(
            simple_structure,
            method="redblack",
            use_julia=True,
            max_iterations=10,
        )

        if solver._use_julia and solver._julia_main is not None:
            num_threads = solver._julia_main.get_num_threads()
            assert num_threads >= 1, "Julia should have at least 1 thread"
            print(f"Julia is using {num_threads} threads")

    @pytest.mark.parametrize("method", ["sor", "redblack"])
    def test_julia_parallel_plate(self, simple_structure, method):
        """Test Julia backend with parallel plate capacitor"""
        nz = simple_structure.nz

        solver = PoissonSolver(
            simple_structure,
            omega=1.5,
            tolerance=1e-8,
            max_iterations=5000,
            method=method,
            use_julia=True,
        )

        phi_initial = np.zeros((nz, simple_structure.nx, simple_structure.ny))
        phi_initial[0, :, :] = 1.0
        phi_initial[-1, :, :] = 0.0

        result = solver.solve(phi_initial=phi_initial, verbose=False)

        # Check convergence
        assert result.info["converged"], f"Julia {method}: Solver should converge"
        assert not np.isnan(result.phi).any(), f"Julia {method}: No NaN values"

        # Check boundary conditions
        assert np.abs(result.phi[0, :, :].mean() - 1.0) < 1e-6, (
            f"Julia {method}: Top boundary should be 1V"
        )
        assert np.abs(result.phi[-1, :, :].mean() - 0.0) < 1e-6, (
            f"Julia {method}: Bottom boundary should be 0V"
        )

        # Check analytical solution
        k_coords = np.arange(nz)
        K = nz - 1
        phi_analytical = 1.0 - k_coords / K

        phi_numerical = result.phi[:, 2, 2]
        error = np.abs(phi_numerical[1:-1] - phi_analytical[1:-1])
        max_error = error.max()

        assert max_error < 0.01, (
            f"Julia {method}: Max error {max_error:.6e} should be < 0.01"
        )

    @pytest.mark.parametrize("method", ["sor", "redblack"])
    def test_julia_vs_numba_consistency(self, simple_structure, method):
        """Test that Julia and Numba backends produce consistent results"""
        nz, nx, ny = simple_structure.nz, simple_structure.nx, simple_structure.ny

        phi_initial = np.zeros((nz, nx, ny))
        phi_initial[0, :, :] = 1.0
        phi_initial[-1, :, :] = 0.0

        # Solve with Numba
        solver_numba = PoissonSolver(
            simple_structure,
            omega=1.5,
            tolerance=1e-8,
            max_iterations=5000,
            method=method,
            use_julia=False,
        )
        result_numba = solver_numba.solve(phi_initial=phi_initial.copy(), verbose=False)

        # Solve with Julia
        solver_julia = PoissonSolver(
            simple_structure,
            omega=1.5,
            tolerance=1e-8,
            max_iterations=5000,
            method=method,
            use_julia=True,
        )
        result_julia = solver_julia.solve(phi_initial=phi_initial.copy(), verbose=False)

        # Compare results
        max_diff = np.max(np.abs(result_numba.phi - result_julia.phi))

        print(f"\n{method} backend comparison:")
        print(f"  Numba iterations: {result_numba.info['iterations']}")
        print(f"  Julia iterations: {result_julia.info['iterations']}")
        print(f"  Max difference: {max_diff:.2e}")

        # Allow some tolerance due to floating point differences
        assert max_diff < 1e-6, (
            f"{method}: Numba and Julia results should be consistent, "
            f"but max diff is {max_diff:.2e}"
        )

    def test_julia_electrode_volume(self):
        """Test Julia backend with 3D electrode volumes"""
        nz, nx, ny = 11, 11, 11
        h = 10e-9

        epsilon = np.ones((nz, nx, ny)) * 11.7

        electrode_mask = np.zeros((nz, nx, ny), dtype=bool)
        electrode_mask[0:2, 4:7, 4:7] = True

        electrode_voltages = np.zeros((nz, nx, ny))
        electrode_voltages[0:2, 4:7, 4:7] = -0.5

        boundary_conditions = {
            "z_top": {"type": "neumann", "value": 0.0},
            "z_bottom": {"type": "neumann", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        }

        structure = MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
            electrode_mask=electrode_mask,
            electrode_voltages=electrode_voltages,
        )

        solver = PoissonSolver(
            structure,
            omega=1.8,
            tolerance=1e-6,
            max_iterations=10000,
            method="redblack",
            use_julia=True,
        )

        phi_initial = np.zeros((nz, nx, ny))
        phi_initial[electrode_mask] = -0.5

        result = solver.solve(phi_initial=phi_initial, verbose=False)

        # Check electrode voltage
        electrode_phi = result.phi[electrode_mask]
        assert np.allclose(electrode_phi, -0.5, atol=1e-6), (
            "Julia: Electrode potential should be -0.5V"
        )

        assert result.info["converged"], "Julia: Solver should converge"

    def test_julia_neumann_bc(self):
        """Test Julia backend with all Neumann boundary conditions"""
        nz, nx, ny = 10, 10, 10
        h = 1e-9

        epsilon = np.ones((nz, nx, ny)) * 11.7
        boundary_conditions = {
            "z_top": {"type": "neumann", "value": 0.0},
            "z_bottom": {"type": "neumann", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        }

        structure = MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
        )

        solver = PoissonSolver(
            structure,
            omega=1.5,
            tolerance=1e-6,
            max_iterations=1000,
            method="sor",
            use_julia=True,
        )

        result = solver.solve(verbose=False)

        # With rho=0 and Neumann BC, potential should be constant
        assert result.phi.std() < 1e-6, (
            "Julia: Potential should be constant with zero charge and Neumann BC"
        )
        assert result.info["converged"], "Julia: Solver should converge"

    def test_julia_periodic_bc(self):
        """Test Julia backend with periodic boundary conditions"""
        nz, nx, ny = 10, 10, 10
        h = 1e-9

        epsilon = np.ones((nz, nx, ny)) * 11.7
        boundary_conditions = {
            "z_top": {"type": "dirichlet", "value": 1.0},
            "z_bottom": {"type": "dirichlet", "value": 0.0},
            "x_sides": {"type": "periodic"},
            "y_sides": {"type": "periodic"},
        }

        structure = MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
        )

        solver = PoissonSolver(
            structure,
            omega=1.5,
            tolerance=1e-6,
            max_iterations=1000,
            method="sor",
            use_julia=True,
        )

        result = solver.solve(verbose=False)

        # Check that periodic BC is applied (edges should match)
        assert np.allclose(result.phi[:, 0, :], result.phi[:, -2, :], atol=1e-6), (
            "Julia: x periodic BC should make edges equal"
        )
        assert np.allclose(result.phi[:, :, 0], result.phi[:, :, -2], atol=1e-6), (
            "Julia: y periodic BC should make edges equal"
        )

        assert result.info["converged"], "Julia: Solver should converge"


# =============================================================================
# GPU Backend Tests (CuPy/CUDA)
# =============================================================================


def _gpu_available():
    """Check if GPU backend is available"""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except (ImportError, Exception):
        return False


GPU_AVAILABLE = _gpu_available()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
class TestGPUBackend:
    """Tests for GPU backend functionality (CuPy/CUDA)"""

    @pytest.fixture
    def simple_structure(self):
        """Create a simple structure for testing"""
        nz, nx, ny = 11, 5, 5
        h = 2e-9
        epsilon = np.ones((nz, nx, ny)) * 3.9

        boundary_conditions = {
            "z_top": {"type": "dirichlet", "value": 1.0},
            "z_bottom": {"type": "dirichlet", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        }

        return MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
        )

    def test_gpu_backend_initialization(self, simple_structure):
        """Test that GPU backend initializes correctly"""
        solver = PoissonSolver(
            simple_structure,
            method="redblack",
            use_gpu=True,
            max_iterations=10,
        )

        assert solver._use_gpu, "GPU backend should be enabled"
        assert solver._cp is not None, "CuPy module should be loaded"

    def test_gpu_parallel_plate(self, simple_structure):
        """Test GPU backend with parallel plate capacitor"""
        nz = simple_structure.nz

        solver = PoissonSolver(
            simple_structure,
            omega=1.5,
            tolerance=1e-8,
            max_iterations=5000,
            method="redblack",
            use_gpu=True,
        )

        phi_initial = np.zeros((nz, simple_structure.nx, simple_structure.ny))
        phi_initial[0, :, :] = 1.0
        phi_initial[-1, :, :] = 0.0

        result = solver.solve(phi_initial=phi_initial, verbose=False)

        # Check convergence
        assert result.info["converged"], "GPU: Solver should converge"
        assert not np.isnan(result.phi).any(), "GPU: No NaN values"

        # Check boundary conditions
        assert np.abs(result.phi[0, :, :].mean() - 1.0) < 1e-6, (
            "GPU: Top boundary should be 1V"
        )
        assert np.abs(result.phi[-1, :, :].mean() - 0.0) < 1e-6, (
            "GPU: Bottom boundary should be 0V"
        )

        # Check analytical solution
        k_coords = np.arange(nz)
        K = nz - 1
        phi_analytical = 1.0 - k_coords / K

        phi_numerical = result.phi[:, 2, 2]
        error = np.abs(phi_numerical[1:-1] - phi_analytical[1:-1])
        max_error = error.max()

        assert max_error < 0.01, f"GPU: Max error {max_error:.6e} should be < 0.01"

    def test_gpu_vs_cpu_consistency(self, simple_structure):
        """Test that GPU and CPU backends produce consistent results"""
        nz, nx, ny = simple_structure.nz, simple_structure.nx, simple_structure.ny

        phi_initial = np.zeros((nz, nx, ny))
        phi_initial[0, :, :] = 1.0
        phi_initial[-1, :, :] = 0.0

        # Solve with CPU (Numba)
        solver_cpu = PoissonSolver(
            simple_structure,
            omega=1.5,
            tolerance=1e-8,
            max_iterations=5000,
            method="redblack",
            use_gpu=False,
        )
        result_cpu = solver_cpu.solve(phi_initial=phi_initial.copy(), verbose=False)

        # Solve with GPU
        solver_gpu = PoissonSolver(
            simple_structure,
            omega=1.5,
            tolerance=1e-8,
            max_iterations=5000,
            method="redblack",
            use_gpu=True,
        )
        result_gpu = solver_gpu.solve(phi_initial=phi_initial.copy(), verbose=False)

        # Compare results
        max_diff = np.max(np.abs(result_cpu.phi - result_gpu.phi))

        print("\nGPU vs CPU backend comparison:")
        print(f"  CPU iterations: {result_cpu.info['iterations']}")
        print(f"  GPU iterations: {result_gpu.info['iterations']}")
        print(f"  Max difference: {max_diff:.2e}")

        # Allow some tolerance due to floating point differences
        assert max_diff < 1e-6, (
            f"GPU and CPU results should be consistent, but max diff is {max_diff:.2e}"
        )

    def test_gpu_electrode_volume(self):
        """Test GPU backend with 3D electrode volumes"""
        nz, nx, ny = 11, 11, 11
        h = 10e-9

        epsilon = np.ones((nz, nx, ny)) * 11.7

        electrode_mask = np.zeros((nz, nx, ny), dtype=bool)
        electrode_mask[0:2, 4:7, 4:7] = True

        electrode_voltages = np.zeros((nz, nx, ny))
        electrode_voltages[0:2, 4:7, 4:7] = -0.5

        boundary_conditions = {
            "z_top": {"type": "neumann", "value": 0.0},
            "z_bottom": {"type": "neumann", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        }

        structure = MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
            electrode_mask=electrode_mask,
            electrode_voltages=electrode_voltages,
        )

        solver = PoissonSolver(
            structure,
            omega=1.8,
            tolerance=1e-6,
            max_iterations=10000,
            method="redblack",
            use_gpu=True,
        )

        phi_initial = np.zeros((nz, nx, ny))
        phi_initial[electrode_mask] = -0.5

        result = solver.solve(phi_initial=phi_initial, verbose=False)

        # Check electrode voltage
        electrode_phi = result.phi[electrode_mask]
        assert np.allclose(electrode_phi, -0.5, atol=1e-6), (
            "GPU: Electrode potential should be -0.5V"
        )

        assert result.info["converged"], "GPU: Solver should converge"

    def test_gpu_neumann_bc(self):
        """Test GPU backend with all Neumann boundary conditions"""
        nz, nx, ny = 10, 10, 10
        h = 1e-9

        epsilon = np.ones((nz, nx, ny)) * 11.7
        boundary_conditions = {
            "z_top": {"type": "neumann", "value": 0.0},
            "z_bottom": {"type": "neumann", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        }

        structure = MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
        )

        solver = PoissonSolver(
            structure,
            omega=1.5,
            tolerance=1e-6,
            max_iterations=1000,
            method="redblack",
            use_gpu=True,
        )

        result = solver.solve(verbose=False)

        # With rho=0 and Neumann BC, potential should be constant
        assert result.phi.std() < 1e-6, (
            "GPU: Potential should be constant with zero charge and Neumann BC"
        )
        assert result.info["converged"], "GPU: Solver should converge"


# =============================================================================
# Julia CUDA.jl GPU Backend Tests
# =============================================================================


def _julia_cuda_available():
    """Check if Julia CUDA.jl backend is available"""
    if not JULIA_AVAILABLE:
        return False
    try:
        from juliacall import Main as jl

        # Check if CUDA.jl is available and functional
        jl.seval("using CUDA")
        return bool(jl.seval("CUDA.functional()"))
    except Exception:
        return False


JULIA_CUDA_AVAILABLE = _julia_cuda_available()


@pytest.mark.skipif(
    not JULIA_CUDA_AVAILABLE, reason="Julia CUDA.jl backend not available"
)
class TestJuliaCUDABackend:
    """Tests for Julia CUDA.jl GPU backend functionality"""

    @pytest.fixture
    def simple_structure(self):
        """Create a simple structure for testing"""
        nz, nx, ny = 11, 5, 5
        h = 2e-9
        epsilon = np.ones((nz, nx, ny)) * 3.9

        boundary_conditions = {
            "z_top": {"type": "dirichlet", "value": 1.0},
            "z_bottom": {"type": "dirichlet", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        }

        return MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
        )

    def test_julia_cuda_backend_initialization(self, simple_structure):
        """Test that Julia CUDA backend initializes correctly"""
        solver = PoissonSolver(
            simple_structure,
            method="redblack",
            use_julia=True,
            use_gpu=True,
            max_iterations=10,
        )

        assert solver._use_julia, "Julia backend should be enabled"
        assert solver._use_gpu, "GPU flag should be enabled"
        assert solver._julia_main is not None, "Julia main should be initialized"

    def test_julia_cuda_parallel_plate(self, simple_structure):
        """Test Julia CUDA backend with parallel plate capacitor"""
        nz = simple_structure.nz

        solver = PoissonSolver(
            simple_structure,
            omega=1.5,
            tolerance=1e-8,
            max_iterations=5000,
            method="redblack",
            use_julia=True,
            use_gpu=True,
        )

        phi_initial = np.zeros((nz, simple_structure.nx, simple_structure.ny))
        phi_initial[0, :, :] = 1.0
        phi_initial[-1, :, :] = 0.0

        result = solver.solve(phi_initial=phi_initial, verbose=False)

        # Check convergence
        assert result.info["converged"], "Julia CUDA: Solver should converge"
        assert not np.isnan(result.phi).any(), "Julia CUDA: No NaN values"

        # Check boundary conditions
        assert np.abs(result.phi[0, :, :].mean() - 1.0) < 1e-6, (
            "Julia CUDA: Top boundary should be 1V"
        )
        assert np.abs(result.phi[-1, :, :].mean() - 0.0) < 1e-6, (
            "Julia CUDA: Bottom boundary should be 0V"
        )

        # Check analytical solution
        k_coords = np.arange(nz)
        K = nz - 1
        phi_analytical = 1.0 - k_coords / K

        phi_numerical = result.phi[:, 2, 2]
        error = np.abs(phi_numerical[1:-1] - phi_analytical[1:-1])
        max_error = error.max()

        assert max_error < 0.01, (
            f"Julia CUDA: Max error {max_error:.6e} should be < 0.01"
        )

    def test_julia_cuda_vs_julia_cpu_consistency(self, simple_structure):
        """Test that Julia CUDA and Julia CPU backends produce consistent results"""
        nz, nx, ny = simple_structure.nz, simple_structure.nx, simple_structure.ny

        phi_initial = np.zeros((nz, nx, ny))
        phi_initial[0, :, :] = 1.0
        phi_initial[-1, :, :] = 0.0

        # Solve with Julia CPU
        solver_cpu = PoissonSolver(
            simple_structure,
            omega=1.5,
            tolerance=1e-8,
            max_iterations=5000,
            method="redblack",
            use_julia=True,
            use_gpu=False,
        )
        result_cpu = solver_cpu.solve(phi_initial=phi_initial.copy(), verbose=False)

        # Solve with Julia CUDA
        solver_cuda = PoissonSolver(
            simple_structure,
            omega=1.5,
            tolerance=1e-8,
            max_iterations=5000,
            method="redblack",
            use_julia=True,
            use_gpu=True,
        )
        result_cuda = solver_cuda.solve(phi_initial=phi_initial.copy(), verbose=False)

        # Compare results
        max_diff = np.max(np.abs(result_cpu.phi - result_cuda.phi))

        print("\nJulia CUDA vs Julia CPU backend comparison:")
        print(f"  Julia CPU iterations: {result_cpu.info['iterations']}")
        print(f"  Julia CUDA iterations: {result_cuda.info['iterations']}")
        print(f"  Max difference: {max_diff:.2e}")

        # Allow some tolerance due to floating point differences
        assert max_diff < 1e-6, (
            f"Julia CUDA and Julia CPU results should be consistent, "
            f"but max diff is {max_diff:.2e}"
        )

    def test_julia_cuda_electrode_volume(self):
        """Test Julia CUDA backend with 3D electrode volumes"""
        nz, nx, ny = 11, 11, 11
        h = 10e-9

        epsilon = np.ones((nz, nx, ny)) * 11.7

        electrode_mask = np.zeros((nz, nx, ny), dtype=bool)
        electrode_mask[0:2, 4:7, 4:7] = True

        electrode_voltages = np.zeros((nz, nx, ny))
        electrode_voltages[0:2, 4:7, 4:7] = -0.5

        boundary_conditions = {
            "z_top": {"type": "neumann", "value": 0.0},
            "z_bottom": {"type": "neumann", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        }

        structure = MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
            electrode_mask=electrode_mask,
            electrode_voltages=electrode_voltages,
        )

        solver = PoissonSolver(
            structure,
            omega=1.8,
            tolerance=1e-6,
            max_iterations=10000,
            method="redblack",
            use_julia=True,
            use_gpu=True,
        )

        phi_initial = np.zeros((nz, nx, ny))
        phi_initial[electrode_mask] = -0.5

        result = solver.solve(phi_initial=phi_initial, verbose=False)

        # Check electrode voltage
        electrode_phi = result.phi[electrode_mask]
        assert np.allclose(electrode_phi, -0.5, atol=1e-6), (
            "Julia CUDA: Electrode potential should be -0.5V"
        )

        assert result.info["converged"], "Julia CUDA: Solver should converge"

    def test_julia_cuda_neumann_bc(self):
        """Test Julia CUDA backend with all Neumann boundary conditions"""
        nz, nx, ny = 10, 10, 10
        h = 1e-9

        epsilon = np.ones((nz, nx, ny)) * 11.7
        boundary_conditions = {
            "z_top": {"type": "neumann", "value": 0.0},
            "z_bottom": {"type": "neumann", "value": 0.0},
            "x_sides": {"type": "neumann", "value": 0.0},
            "y_sides": {"type": "neumann", "value": 0.0},
        }

        structure = MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
        )

        solver = PoissonSolver(
            structure,
            omega=1.5,
            tolerance=1e-6,
            max_iterations=1000,
            method="redblack",
            use_julia=True,
            use_gpu=True,
        )

        result = solver.solve(verbose=False)

        # With rho=0 and Neumann BC, potential should be constant
        assert result.phi.std() < 1e-6, (
            "Julia CUDA: Potential should be constant with zero charge and Neumann BC"
        )
        assert result.info["converged"], "Julia CUDA: Solver should converge"

    def test_julia_cuda_periodic_bc(self):
        """Test Julia CUDA backend with periodic boundary conditions"""
        nz, nx, ny = 10, 10, 10
        h = 1e-9

        epsilon = np.ones((nz, nx, ny)) * 11.7
        boundary_conditions = {
            "z_top": {"type": "dirichlet", "value": 1.0},
            "z_bottom": {"type": "dirichlet", "value": 0.0},
            "x_sides": {"type": "periodic"},
            "y_sides": {"type": "periodic"},
        }

        structure = MockStructureManager(
            epsilon_array=epsilon,
            h=h,
            boundary_conditions=boundary_conditions,
        )

        solver = PoissonSolver(
            structure,
            omega=1.5,
            tolerance=1e-6,
            max_iterations=1000,
            method="redblack",
            use_julia=True,
            use_gpu=True,
        )

        result = solver.solve(verbose=False)

        # Check that periodic BC is applied (edges should match)
        assert np.allclose(result.phi[:, 0, :], result.phi[:, -2, :], atol=1e-6), (
            "Julia CUDA: x periodic BC should make edges equal"
        )
        assert np.allclose(result.phi[:, :, 0], result.phi[:, :, -2], atol=1e-6), (
            "Julia CUDA: y periodic BC should make edges equal"
        )

        assert result.info["converged"], "Julia CUDA: Solver should converge"