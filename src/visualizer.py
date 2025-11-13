"""Visualization module

Visualizes potential distribution, electrode patterns, convergence history, etc.
"""

import numpy as np
import matplotlib.pyplot as plt

# from pathlib import Path
from typing import Optional, Tuple


def plot_potential_slice(
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z_index: Optional[int] = None,
    electrode_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Potential Distribution",
) -> None:
    """Plot 2D slice of potential distribution

    Parameters
    ----------
    phi : np.ndarray
        Potential distribution (nz, nx, ny)
    x, y, z : np.ndarray
        Coordinate arrays (m)
    z_index : int, optional
        Slice position in z direction (index). Center if None
    electrode_mask : np.ndarray, optional
        Electrode mask (nz, nx, ny)
    save_path : str, optional
        Path to save file
    title : str
        Graph title
    """
    if z_index is None:
        z_index = phi.shape[0] // 2

    # Get slice
    phi_slice = phi[z_index, :, :]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Potential distribution
    im = ax.pcolormesh(x * 1e9, y * 1e9, phi_slice.T, cmap="RdBu_r", shading="auto")

    # Overlay electrode positions
    if electrode_mask is not None:
        electrode_slice = electrode_mask[z_index, :, :]
        if electrode_slice.any():
            ax.contour(
                x * 1e9,
                y * 1e9,
                electrode_slice.T,
                colors="black",
                linewidths=2,
                levels=[0.5],
            )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"{title} at z={z[z_index] * 1e9:.1f} nm")
    ax.set_aspect("equal")

    # Color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Potential (V)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def plot_multiple_slices(
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z_indices: Optional[list] = None,
    electrode_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot potential distribution at multiple z positions

    Parameters
    ----------
    phi : np.ndarray
        Potential distribution (nz, nx, ny)
    x, y, z : np.ndarray
        Coordinate arrays (m)
    z_indices : list, optional
        List of slice positions in z direction. 4 evenly spaced if None
    electrode_mask : np.ndarray, optional
        Electrode mask (nz, nx, ny)
    save_path : str, optional
        Path to save file
    """
    if z_indices is None:
        nz = phi.shape[0]
        z_indices = [nz // 4, nz // 2, 3 * nz // 4, -1]

    n_slices = len(z_indices)
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 4))

    if n_slices == 1:
        axes = [axes]

    vmin = phi.min()
    vmax = phi.max()

    for i, z_idx in enumerate(z_indices):
        ax = axes[i]
        phi_slice = phi[z_idx, :, :]

        # Potential distribution
        im = ax.pcolormesh(
            x * 1e9,
            y * 1e9,
            phi_slice.T,
            cmap="RdBu_r",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        # Electrode positions
        if electrode_mask is not None:
            electrode_slice = electrode_mask[z_idx, :, :]
            if electrode_slice.any():
                ax.contour(
                    x * 1e9,
                    y * 1e9,
                    electrode_slice.T,
                    colors="black",
                    linewidths=2,
                    levels=[0.5],
                )

        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(f"z={z[z_idx] * 1e9:.1f} nm")
        ax.set_aspect("equal")

    # Color bar (shared)
    fig.colorbar(im, ax=axes, label="Potential (V)", shrink=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def plot_electrode_pattern(
    electrode_mask: np.ndarray,
    electrode_voltages: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z_index: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """Visualize electrode pattern

    Parameters
    ----------
    electrode_mask : np.ndarray
        Electrode mask (nz, nx, ny)
    electrode_voltages : np.ndarray
        Electrode voltage (nz, nx, ny)
    x, y : np.ndarray
        Coordinate arrays (m)
    z_index : int
        Index in z direction (default is surface)
    save_path : str, optional
        Path to save file
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Slice of electrode mask and data
    mask_slice = electrode_mask[z_index, :, :]
    voltage_slice = electrode_voltages[z_index, :, :].copy()

    # Set non-electrode regions to NaN
    voltage_slice[~mask_slice] = np.nan

    # Plot
    im = ax.pcolormesh(
        x * 1e9, y * 1e9, voltage_slice.T, cmap="viridis", shading="auto"
    )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title("Electrode Pattern and Voltages")
    ax.set_aspect("equal")

    # Color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Voltage (V)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def plot_convergence(
    convergence_history: list = None,
    residual_history: list = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot convergence history

    Parameters
    ----------
    convergence_history : list, optional
        History of phi changes (preferred)
    residual_history : list, optional
        History of residuals (deprecated, for backward compatibility)
    save_path : str, optional
        Path to save file
    """
    # Handle backward compatibility
    if convergence_history is None and residual_history is None:
        raise ValueError("Either convergence_history or residual_history must be provided")

    history = convergence_history if convergence_history is not None else residual_history
    ylabel = "Max |Δφ| (V)" if convergence_history is not None else "Residual (L2 norm)"

    fig, ax = plt.subplots(figsize=(8, 6))

    iterations = range(1, len(history) + 1)
    ax.semilogy(iterations, history, "b-", linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title("Convergence History")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def save_results(
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    info: dict,
    save_path: str,
) -> None:
    """Save calculation results to file

    Parameters
    ----------
    phi : np.ndarray
        Potential distribution
    x, y, z : np.ndarray
        Coordinate arrays
    info : dict
        Convergence information
    save_path : str
        Path to save file (.npz format)
    """
    np.savez(
        save_path,
        phi=phi,
        x=x,
        y=y,
        z=z,
        converged=info.get("converged", False),
        iterations=info.get("iterations", 0),
        final_residual=info.get("final_residual", 0.0),
    )
    print(f"Results saved to: {save_path}")


def load_results(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load saved results

    Parameters
    ----------
    file_path : str
        File path (.npz format)

    Returns
    -------
    phi, x, y, z : np.ndarray
        Potential distribution and coordinates
    info : dict
        Convergence information
    """
    data = np.load(file_path)

    phi = data["phi"]
    x = data["x"]
    y = data["y"]
    z = data["z"]

    info = {
        "converged": bool(data["converged"]),
        "iterations": int(data["iterations"]),
        "final_residual": float(data["final_residual"]),
    }

    return phi, x, y, z, info


def plot_band_diagram_1d(
    result,
    x_idx: Optional[int] = None,
    y_idx: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot 1D band diagram along z-direction

    Parameters
    ----------
    result : SolverResult
        Solver result containing potential and material information
    x_idx : int, optional
        Index in x-direction (default: center)
    y_idx : int, optional
        Index in y-direction (default: center)
    save_path : str, optional
        Path to save file

    Notes
    -----
    Plots:
    - Ec(z): Conduction band edge (solid blue line)
    - Ev(z): Valence band edge (solid red line)
    - -φ(z): Fermi level shift (dashed black line)
    - Material boundaries: vertical gray lines
    """
    # Extract 1D band diagram data
    z, Ec, Ev, phi = result.get_band_diagram_1d(x_idx=x_idx, y_idx=y_idx)

    # Convert z to nm for plotting
    z_nm = z * 1e9

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot band edges
    ax.plot(z_nm, Ec, "b-", linewidth=2, label="$E_c$ (conduction band)")
    ax.plot(z_nm, Ev, "r-", linewidth=2, label="$E_v$ (valence band)")

    # Plot -φ (Fermi level shift)
    ax.plot(z_nm, -phi, "k--", linewidth=1.5, label=r"$-\phi$ (Fermi shift)")

    # Mark material boundaries
    if result.materials is not None:
        # Find where material changes
        material_names = [mat.name for mat in result.materials]
        for k in range(1, len(material_names)):
            if material_names[k] != material_names[k - 1]:
                z_boundary = z_nm[k]
                ax.axvline(
                    z_boundary,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    alpha=0.7,
                )

        # Add material labels at the top
        current_material = material_names[0]
        start_idx = 0
        for k in range(1, len(material_names) + 1):
            if k == len(material_names) or material_names[k] != current_material:
                # Add label at middle of region
                mid_idx = (start_idx + k - 1) // 2
                z_label = z_nm[mid_idx]
                y_label = ax.get_ylim()[1] * 0.95  # Near top of plot
                ax.text(
                    z_label,
                    y_label,
                    current_material,
                    ha="center",
                    va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
                )
                if k < len(material_names):
                    current_material = material_names[k]
                    start_idx = k

    # Labels and formatting
    ax.set_xlabel("z position (nm)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("1D Band Diagram (z-direction)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


# ===========================================================================
# Wavefunction Visualization
# ===========================================================================


def plot_wavefunction(
    wavefunction,
    x,
    y,
    title='Wavefunction ψ(x,y)',
    cmap='RdBu',
    ax=None,
    colorbar=True,
    save_path=None,
):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    psi_plot = wavefunction.real
    X, Y = np.meshgrid(x * 1e9, y * 1e9, indexing='ij')
    im = ax.pcolormesh(X, Y, psi_plot, cmap=cmap, shading='auto')
    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('ψ (a.u.)')
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_title(title)
    ax.set_aspect('equal')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return ax
