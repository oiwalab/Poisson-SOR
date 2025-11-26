"""Visualization module

Visualizes potential distribution, electrode patterns, convergence history, etc.
"""

import numpy as np
import matplotlib.pyplot as plt

# from pathlib import Path
from typing import Optional, Tuple

# Optional plotly import for 3D interactive visualization
try:
    import plotly.graph_objects as go  # noqa: F401

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


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
    fig, ax = plt.subplots(figsize=(6, 4))

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
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))

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
    fig.colorbar(
        im, ax=axes, label="Potential (V)", shrink=0.8, orientation="horizontal"
    )

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
    fig, ax = plt.subplots(figsize=(6, 4))

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
        raise ValueError(
            "Either convergence_history or residual_history must be provided"
        )

    history = (
        convergence_history if convergence_history is not None else residual_history
    )
    ylabel = "Max |Δφ| (V)" if convergence_history is not None else "Residual (L2 norm)"

    fig, ax = plt.subplots(figsize=(6, 4))

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
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot band edges
    ax.plot(z_nm, Ec, "b-", linewidth=2, label="$E_c$ (conduction band)")
    ax.plot(z_nm, Ev, "r-", linewidth=2, label="$E_v$ (valence band)")

    # Plot -φ (Fermi level shift)
    ax.plot(z_nm, -phi, "k--", linewidth=1.5, label=r"$-\phi$ (Fermi shift)")

    # Mark material boundaries
    # Find where material changes
    material_names = [
        result.structure.unique_materials[mat_idx].name
        for mat_idx in result.structure.material_indices
    ]
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
    title="Wavefunction ψ(x,y)",
    cmap="RdBu",
    ax=None,
    colorbar=True,
    save_path=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
    psi_plot = wavefunction.real
    X, Y = np.meshgrid(x * 1e9, y * 1e9, indexing="ij")
    im = ax.pcolormesh(X, Y, psi_plot, cmap=cmap, shading="auto")
    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("ψ (a.u.)")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return ax


# ===========================================================================
# 3D Interactive Visualization with Plotly
# ===========================================================================


def plot_structure_3d(
    manager,
    material_colors: Optional[dict] = None,
    electrode_color: str = "gold",
    show_axes: bool = True,
    width: int = 900,
    height: int = 700,
) -> None:
    """Plot 3D device structure showing material layers and electrodes

    Parameters
    ----------
    manager : StructureManager
        Structure manager containing device geometry
    material_colors : dict, optional
        Colors for each material (default: {"SiO2": "lightblue", "Si": "gray"})
    electrode_color : str
        Color for electrodes (default: "gold")
    show_axes : bool
        Whether to show coordinate axes (default: True)
    width : int
        Figure width in pixels
    height : int
        Figure height in pixels

    Raises
    ------
    ImportError
        If plotly is not installed
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for 3D visualization. Install with: pip install plotly"
        )

    if material_colors is None:
        material_colors = {"SiO2": "lightblue", "Si": "gray"}

    fig = go.Figure()

    # Convert to nm for visualization
    size_x_nm = manager.size_x * 1e9
    size_y_nm = manager.size_y * 1e9

    # Helper function to create 3D box
    def create_box(x_range, y_range, z_range, color, name, opacity=0.5):
        """Create a 3D box using Mesh3d"""
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range

        # 8 vertices of the box
        vertices_x = [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min]
        vertices_y = [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max]
        vertices_z = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]

        # 12 triangles (2 per face, 6 faces)
        i = [0, 0, 0, 0, 4, 4, 6, 6, 0, 0, 2, 2]
        j = [1, 2, 3, 4, 5, 6, 7, 2, 5, 1, 6, 3]
        k = [2, 3, 4, 5, 6, 7, 4, 1, 4, 5, 7, 7]

        return go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            name=name,
            showlegend=True,
        )

    # Add material layers
    for layer in manager.layers:
        z_max, z_min = layer["z_range"]
        material = layer["material"]

        box = create_box(
            x_range=[0, size_x_nm],
            y_range=[0, size_y_nm],
            z_range=[z_min * 1e9, z_max * 1e9],
            color=material_colors.get(material, "lightgray"),
            name=material,
            opacity=0.3,
        )
        fig.add_trace(box)

    # Add electrodes
    for electrode in manager.electrodes:
        x_min, x_max = electrode["x_range"]
        y_min, y_max = electrode["y_range"]
        z_position = electrode["z_position"]
        voltage = electrode["voltage"]

        box = create_box(
            x_range=[x_min * 1e9, x_max * 1e9],
            y_range=[y_min * 1e9, y_max * 1e9],
            z_range=[z_position * 1e9, 0],
            color=electrode_color,
            name=f"{electrode['name']} ({voltage}V)",
            opacity=0.8,
        )
        fig.add_trace(box)

    # Add coordinate axes
    if show_axes:
        axis_length = max(size_x_nm, size_y_nm, abs(manager.size_z * 1e9)) * 0.3

        # X-axis (red)
        fig.add_trace(
            go.Scatter3d(
                x=[0, axis_length],
                y=[0, 0],
                z=[0, 0],
                mode="lines+text",
                line=dict(color="red", width=4),
                text=["", "x"],
                textposition="top center",
                showlegend=False,
            )
        )

        # Y-axis (green)
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, axis_length],
                z=[0, 0],
                mode="lines+text",
                line=dict(color="green", width=4),
                text=["", "y"],
                textposition="top center",
                showlegend=False,
            )
        )

        # Z-axis (blue)
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[0, -axis_length],
                mode="lines+text",
                line=dict(color="blue", width=4),
                text=["", "z"],
                textposition="middle right",
                showlegend=False,
            )
        )

    # Update layout
    fig.update_layout(
        title="3D Device Structure (Interactive)",
        scene=dict(
            xaxis_title="x (nm)",
            yaxis_title="y (nm)",
            zaxis_title="z (nm)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        width=width,
        height=height,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    fig.show()

    print("3D structure visualization created")
    print(f"  Materials: {', '.join([layer['material'] for layer in manager.layers])}")
    print(f"  Electrodes: {len(manager.electrodes)}")


def plot_potential_3d(
    result,
    stride: int = 2,
    colorscale: str = "RdBu_r",
    isosurface_values: Optional[list] = None,
    volume_opacity: float = 0.1,
    surface_count: int = 15,
    isosurface_opacity: float = 0.5,
    width: int = 900,
    height: int = 700,
) -> None:
    """Plot 3D potential distribution with volume rendering and isosurfaces

    Parameters
    ----------
    result : SolverResult
        Solver result containing potential distribution
    stride : int
        Sampling stride (every N points). Use larger values for big grids
    colorscale : str
        Plotly colorscale name (default: "RdBu_r")
    isosurface_values : list, optional
        List of potential values for isosurfaces (default: None, auto-generate)
    volume_opacity : float
        Opacity of volume rendering (0-1)
    surface_count : int
        Number of surfaces for volume rendering
    isosurface_opacity : float
        Opacity of isosurfaces (0-1)
    width : int
        Figure width in pixels
    height : int
        Figure height in pixels

    Raises
    ------
    ImportError
        If plotly is not installed

    Notes
    -----
    For large grids (>100^3), increase stride to reduce memory usage and
    improve performance. stride=2 samples every 2nd point in each dimension.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for 3D visualization. Install with: pip install plotly"
        )

    # Sample the data to reduce size
    # result.phi shape is (nz, nx, ny), transpose to (nx, ny, nz) for meshgrid
    phi_reordered = result.phi.transpose(1, 2, 0)
    phi_sampled = phi_reordered[::stride, ::stride, ::stride]

    x_sampled = result.x[::stride]
    y_sampled = result.y[::stride]
    z_sampled = result.z[::stride]

    # Create meshgrid for 3D coordinates
    X, Y, Z = np.meshgrid(
        x_sampled * 1e9, y_sampled * 1e9, z_sampled * 1e9, indexing="ij"
    )

    # Flatten arrays for scatter plot
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    phi_flat = phi_sampled.flatten()

    # Define colorscale range
    vmin = result.phi.min()
    vmax = result.phi.max()

    # Create volume rendering
    fig = go.Figure(
        data=go.Volume(
            x=X_flat,
            y=Y_flat,
            z=Z_flat,
            value=phi_flat,
            isomin=vmin,
            isomax=vmax,
            opacity=volume_opacity,
            surface_count=surface_count,
            colorscale=colorscale,
            colorbar=dict(title="Potential (V)", x=1.1),
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    # Add isosurfaces at specific potential values
    if isosurface_values is None:
        # Auto-generate isosurface values
        isosurface_values = [
            vmin + 0.25 * (vmax - vmin),
            vmin + 0.50 * (vmax - vmin),
            vmin + 0.75 * (vmax - vmin),
        ]

    for iso_val in isosurface_values:
        fig.add_trace(
            go.Isosurface(
                x=X_flat,
                y=Y_flat,
                z=Z_flat,
                value=phi_flat,
                isomin=iso_val - 0.05,
                isomax=iso_val + 0.05,
                opacity=isosurface_opacity,
                surface_count=1,
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                showscale=False,
                name=f"φ = {iso_val:.3f} V",
            )
        )

    # Update layout
    fig.update_layout(
        title="3D Potential Distribution (Interactive)",
        scene=dict(
            xaxis_title="x (nm)",
            yaxis_title="y (nm)",
            zaxis_title="z (nm)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode="cube",
        ),
        width=width,
        height=height,
    )

    fig.show()

    print("3D potential visualization created")
    print("  Use mouse to rotate, zoom, and pan")
    print("  - Click and drag to rotate")
    print("  - Scroll to zoom")
    print("  - Right-click and drag to pan")
    print(f"\n  Colorscale: {colorscale}")
    print(f"  Range: [{vmin:.3f}, {vmax:.3f}] V")
    print(f"  Isosurfaces at: {[f'{v:.3f}' for v in isosurface_values]} V")  # noqa: F541
    print("\n  Array shapes:")
    print(f"  - Original phi: {result.phi.shape} (nz, nx, ny)")
    print(f"  - Reordered phi: {phi_reordered.shape} (nx, ny, nz)")
    print(f"  - Sampled phi: {phi_sampled.shape}")
