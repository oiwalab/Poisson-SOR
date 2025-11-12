"""Example usage of PotentialInterpolator

Demonstrates how to use linear interpolation for fast computation
of potential distributions at different electrode voltages.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from structure_manager import StructureManager
from poisson_solver import PoissonSolver
from potential_interpolator import PotentialInterpolator


def main():
    """Main example demonstrating PotentialInterpolator usage"""

    print("=" * 70)
    print("PotentialInterpolator Example")
    print("=" * 70)

    # Load structure from YAML
    config_path = Path(__file__).parent.parent / "configs" / "example.yaml"
    print(f"\nLoading configuration from: {config_path}")

    manager = StructureManager(str(config_path))
    print(f"Grid: ({manager.nx}, {manager.ny}, {manager.nz})")
    print(f"Electrodes: {[e['name'] for e in manager.electrodes]}")

    # Create Poisson solver
    solver = PoissonSolver(manager.params, omega=1.8, tolerance=1e-6)

    # Create interpolator at Si/SiO2 interface (z = -20nm)
    print("\n" + "=" * 70)
    print("Creating PotentialInterpolator at Si/SiO2 interface (z=-20nm)")
    print("=" * 70)

    z_interface = -20e-9  # Si/SiO2 interface position
    interp = PotentialInterpolator(
        manager,
        solver,
        z_position=z_interface,
        charge_density=None,  # No charge density (ρ=0)
        verbose=True
    )

    print(f"\nInterpolator: {interp}")

    # Example 1: Compute potential for specific voltages
    print("\n" + "=" * 70)
    print("Example 1: Compute potential for specific voltages")
    print("=" * 70)

    voltages_1 = {
        "finger_gate_1": 0.5,
        "finger_gate_2": 1.0,
        "finger_gate_3": 0.5,
    }
    print(f"Voltages: {voltages_1}")

    phi_1 = interp(voltages_1)
    print(f"Potential shape: {phi_1.shape}")
    print(f"Potential range: [{phi_1.min():.4f}, {phi_1.max():.4f}] V")

    # Example 2: Sweep middle gate voltage
    print("\n" + "=" * 70)
    print("Example 2: Sweep middle gate voltage")
    print("=" * 70)

    gate2_voltages = np.linspace(-0.5, 1.5, 11)
    potentials = []

    print("Computing potentials for 11 different voltages...")
    for V2 in gate2_voltages:
        voltages = {
            "finger_gate_1": 0.5,
            "finger_gate_2": V2,
            "finger_gate_3": 0.5,
        }
        phi = interp(voltages)
        potentials.append(phi)
        print(f"  V2={V2:+.2f} V: φ range = [{phi.min():+.4f}, {phi.max():+.4f}] V")

    potentials = np.array(potentials)  # Shape: (11, nx, ny)

    # Example 3: Verify linearity
    print("\n" + "=" * 70)
    print("Example 3: Verify linearity (φ(2V) vs 2·φ(1V))")
    print("=" * 70)

    V_test = {"finger_gate_1": 1.0, "finger_gate_2": 0.0, "finger_gate_3": 0.0}
    phi_1V = interp(V_test)

    V_test_2 = {"finger_gate_1": 2.0, "finger_gate_2": 0.0, "finger_gate_3": 0.0}
    phi_2V = interp(V_test_2)

    # Check linearity (accounting for particular solution)
    phi_1V_scaled = interp.particular_potential + 2.0 * (phi_1V - interp.particular_potential)
    max_error = np.max(np.abs(phi_2V - phi_1V_scaled))

    print(f"Max error in linearity: {max_error:.2e} V")
    print(f"Linearity verified: {max_error < 1e-10}")

    # Visualization
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)

    # Plot 1: Potential distribution for example 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Convert to nm for plotting
    x_nm = interp.x * 1e9
    y_nm = interp.y * 1e9

    # 2D potential map
    im1 = axes[0].pcolormesh(x_nm, y_nm, phi_1.T, shading='auto', cmap='RdBu_r')
    axes[0].set_xlabel('x (nm)')
    axes[0].set_ylabel('y (nm)')
    axes[0].set_title(f'Potential at z={z_interface*1e9:.1f} nm\n' +
                      f'V=[{voltages_1["finger_gate_1"]}, ' +
                      f'{voltages_1["finger_gate_2"]}, ' +
                      f'{voltages_1["finger_gate_3"]}] V')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], label='Potential (V)')

    # 1D cross-section at y = center
    y_center_idx = manager.ny // 2
    axes[1].plot(x_nm, phi_1[:, y_center_idx], 'b-', linewidth=2, label='Interpolated')
    axes[1].set_xlabel('x (nm)')
    axes[1].set_ylabel('Potential (V)')
    axes[1].set_title(f'Cross-section at y={y_nm[y_center_idx]:.1f} nm')
    axes[1].grid(True, alpha=0.3)

    # Mark electrode positions (approximate)
    electrode_regions = [
        (10, 30, 'Gate 1'),
        (40, 60, 'Gate 2'),
        (70, 90, 'Gate 3'),
    ]
    for x_start, x_end, label in electrode_regions:
        axes[1].axvspan(x_start, x_end, alpha=0.2, color='gray')
        axes[1].text((x_start + x_end) / 2, axes[1].get_ylim()[1] * 0.95,
                     label, ha='center', va='top', fontsize=8)

    axes[1].legend()

    plt.tight_layout()
    output_path_1 = Path(__file__).parent / "interpolator_example_1.png"
    plt.savefig(output_path_1, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path_1}")

    # Plot 2: Gate voltage sweep
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract potential at center
    x_center_idx = manager.nx // 2
    y_center_idx = manager.ny // 2
    phi_center = potentials[:, x_center_idx, y_center_idx]

    axes[0].plot(gate2_voltages, phi_center, 'o-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Gate 2 voltage (V)')
    axes[0].set_ylabel('Potential at center (V)')
    axes[0].set_title(f'Potential vs Gate Voltage\n(x, y, z) = center, {z_interface*1e9:.1f} nm')
    axes[0].grid(True, alpha=0.3)

    # Heatmap: potential vs (x, gate voltage)
    phi_vs_x_V2 = potentials[:, :, y_center_idx]  # Shape: (11, nx)
    im2 = axes[1].pcolormesh(x_nm, gate2_voltages, phi_vs_x_V2, shading='auto', cmap='RdBu_r')
    axes[1].set_xlabel('x (nm)')
    axes[1].set_ylabel('Gate 2 voltage (V)')
    axes[1].set_title(f'Potential vs (x, V₂) at y={y_nm[y_center_idx]:.1f} nm')
    plt.colorbar(im2, ax=axes[1], label='Potential (V)')

    # Mark gate 2 position
    axes[1].axvline(40, color='white', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].axvline(60, color='white', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    output_path_2 = Path(__file__).parent / "interpolator_example_2.png"
    plt.savefig(output_path_2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path_2}")

    # Optional: Save interpolator for later use
    print("\n" + "=" * 70)
    print("Saving interpolator to file...")
    print("=" * 70)

    interp_save_path = Path(__file__).parent / "interpolator_z-20nm.npz"
    interp.save(str(interp_save_path))

    # Demonstrate loading
    print("\nLoading interpolator from file...")
    interp_loaded = PotentialInterpolator.load(str(interp_save_path))
    print(f"Loaded: {interp_loaded}")

    # Verify loaded interpolator works
    phi_test = interp_loaded(voltages_1)
    print(f"Loaded interpolator works: {np.allclose(phi_test, phi_1)}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. PotentialInterpolator pre-computes basis functions once")
    print("  2. Interpolation is ~1000x faster than full Poisson solve")
    print("  3. Results are exact for linear systems (ρ=0)")
    print("  4. Save/load functionality avoids re-computation")

    plt.show()


if __name__ == "__main__":
    main()
