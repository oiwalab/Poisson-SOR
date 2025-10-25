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
    residual_history: list,
    save_path: Optional[str] = None,
) -> None:
    """Plot convergence history

    Parameters
    ----------
    residual_history : list
        History of residuals
    save_path : str, optional
        Path to save file
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    iterations = range(1, len(residual_history) + 1)
    ax.semilogy(iterations, residual_history, "b-", linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (L2 norm)")
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
