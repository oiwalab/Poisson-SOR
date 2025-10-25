"""Semiconductor structure management class

Manages material layer structure, electrode placement, and permittivity distribution
"""

import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
# from pathlib import Path


class StructureManager:
    """Semiconductor structure management class

    Loads structure definitions from YAML file,
    generates permittivity distribution, electrode mask, and voltage distribution
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config: Dict = {}
        self.layers: List[Dict] = []
        self.electrodes: List[Dict] = []

        # Computational grid
        self.nx: int = 0
        self.ny: int = 0
        self.nz: int = 0
        self.h: float = 0.0  # Isotropic grid spacing

        # Size of computational domain
        self.size_x: float = 0.0
        self.size_y: float = 0.0
        self.size_z: float = 0.0

        # Permittivity distribution (nx, ny, nz)
        self.epsilon_array: Optional[np.ndarray] = None

        # Electrode mask (nx, ny, nz) - True=electrode present
        self.electrode_mask: Optional[np.ndarray] = None

        # Electrode voltage (nx, ny, nz)
        self.electrode_voltages: Optional[np.ndarray] = None

        # Charge density (nx, ny, nz)
        self.charge_density: Optional[np.ndarray] = None

        if config_path is not None:
            self.load_from_yaml(config_path)

    def load_from_yaml(self, yaml_path: str) -> None:
        """Load structure definition from YAML file

        Parameters
        ----------
        yaml_path : str
            Path to YAML configuration file
        """
        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Extract settings
        self.layers = self.config.get("layers", [])
        self.electrodes = self.config.get("electrodes", [])

        # Set computational domain
        domain = self.config.get("domain", {})
        size = domain["size"]
        grid_spacing = domain["grid_spacing"]

        self.size_x, self.size_y, self.size_z = size
        self.h = grid_spacing

        # Calculate number of grid points
        self.nx = int(self.size_x / self.h) + 1
        self.ny = int(self.size_y / self.h) + 1
        self.nz = int(self.size_z / self.h) + 1

        # Initialize arrays
        self._initialize_arrays()

        # Validate layer structure
        self._validate_layers()

        # Generate permittivity distribution
        self.generate_epsilon_array()

        # Generate electrode structure
        if self.electrodes:
            self.generate_electrode_mask()
            self.get_electrode_voltages()
            self.check_electrode_overlap()

    def _initialize_arrays(self) -> None:
        """Initialize internal arrays

        Array shape: (nz, nx, ny)
        """
        self.epsilon_array = np.ones((self.nz, self.nx, self.ny))
        self.electrode_mask = np.zeros((self.nz, self.nx, self.ny), dtype=bool)
        self.electrode_voltages = np.zeros((self.nz, self.nx, self.ny))
        self.charge_density = np.zeros((self.nz, self.nx, self.ny))

    def _validate_layers(self) -> None:
        """Check validity of layer structure

        New coordinate system: z = 0 (surface) -> z = -size_z (bottom)
        z_range is in [z_max, z_min] format (left value is larger)

        - Check if z-direction ranges overlap
        - Check if there are gaps in z-direction ranges
        - Check if z-direction ranges are within computational domain

        Raises
        ------
        ValueError
            If there are problems with layer structure
        """
        if not self.layers:
            return

        # Sort in descending order by z_range[0] (z_max) (from surface to bottom)
        sorted_layers = sorted(
            self.layers, key=lambda x: x.get("z_range", [0, 0])[0], reverse=True
        )

        # Computational domain range
        domain_z_top = 0.0  # Surface
        domain_z_bottom = -self.size_z  # Bottom

        # Check if first layer starts at 0 (from surface)
        first_z_max = sorted_layers[0].get("z_range", [0, 0])[0]
        if abs(first_z_max - domain_z_top) > 1e-12:
            raise ValueError(
                f"First layer must start at z=0 (surface), but starts at z={first_z_max * 1e9:.2f} nm"
            )

        # Check each layer
        for i, layer in enumerate(sorted_layers):
            z_range = layer.get("z_range", [0, 0])
            material = layer.get("material", "Unknown")

            # Check range validity (in new coordinate system z_range[0] > z_range[1])
            if z_range[0] <= z_range[1]:
                raise ValueError(
                    f"Layer '{material}': Invalid z_range {z_range}. "
                    f"z_range must be [z_max, z_min] with z_max > z_min"
                )

            # Check if within computational domain
            if z_range[0] > domain_z_top or z_range[1] < domain_z_bottom:
                raise ValueError(
                    f"Layer '{material}': z_range [{z_range[0] * 1e9:.2f}, {z_range[1] * 1e9:.2f}] nm "
                    f"is outside domain [{domain_z_top * 1e9:.2f}, {domain_z_bottom * 1e9:.2f}] nm"
                )

            # Check continuity with next layer
            if i < len(sorted_layers) - 1:
                next_layer = sorted_layers[i + 1]
                next_z_max = next_layer.get("z_range", [0, 0])[0]
                current_z_min = z_range[1]

                # Check gap (allow if <= 1e-12 m = 1e-3 nm considering numerical errors)
                gap = current_z_min - next_z_max
                if abs(gap) > 1e-12:
                    if gap > 0:
                        raise ValueError(
                            f"Gap detected between layer '{material}' "
                            f"(ends at {current_z_min * 1e9:.2f} nm) and "
                            f"next layer (starts at {next_z_max * 1e9:.2f} nm). "
                            f"Gap size: {gap * 1e9:.2f} nm"
                        )
                    else:
                        raise ValueError(
                            f"Overlap detected between layer '{material}' "
                            f"(ends at {current_z_min * 1e9:.2f} nm) and "
                            f"next layer (starts at {next_z_max * 1e9:.2f} nm). "
                            f"Overlap size: {-gap * 1e9:.2f} nm"
                        )

        # Check if last layer covers to the end of computational domain
        last_z_min = sorted_layers[-1].get("z_range", [0, 0])[1]
        if abs(last_z_min - domain_z_bottom) > 1e-12:
            raise ValueError(
                f"Last layer must end at z={domain_z_bottom * 1e9:.2f} nm, "
                f"but ends at z={last_z_min * 1e9:.2f} nm"
            )

    def generate_epsilon_array(self) -> np.ndarray:
        """Generate permittivity distribution

        Generate 3D permittivity array based on layer structure

        New coordinate system: z = 0 (surface, k=0) -> z = -size_z (bottom, k=nz-1)

        Returns
        -------
        epsilon_array : np.ndarray
            Relative permittivity distribution (nz, nx, ny)
        """
        if self.epsilon_array is None:
            self._initialize_arrays()

        # Default is vacuum (epsilon_r=1)
        self.epsilon_array[:, :, :] = 1.0

        # Set permittivity for each layer
        for layer in self.layers:
            # material = layer.get('material', 'Unknown')
            z_range = layer.get("z_range", [0, 0])  # [z_max, z_min] where z_max > z_min
            epsilon_r = layer.get("epsilon_r", 1.0)

            # Convert z coordinate to k index
            # z = 0 -> k = 0 (surface)
            # z = -size_z -> k = nz-1 (bottom)
            # k = -z / h
            k_top = int(-z_range[0] / self.h)  # z_max (larger) -> smaller k
            k_bottom = int(-z_range[1] / self.h)  # z_min (smaller) -> larger k

            # Range check
            k_top = max(0, min(k_top, self.nz - 1))
            k_bottom = max(0, min(k_bottom, self.nz - 1))

            # Set permittivity (array shape: (nz, nx, ny))
            self.epsilon_array[k_top : k_bottom + 1, :, :] = epsilon_r

        return self.epsilon_array

    def generate_electrode_mask(self) -> np.ndarray:
        """Generate electrode mask

        3D boolean array with True at electrode positions

        Returns
        -------
        electrode_mask : np.ndarray
            Electrode mask (nz, nx, ny), dtype=bool
        """
        if self.electrode_mask is None:
            self._initialize_arrays()

        self.electrode_mask[:, :, :] = False

        for electrode in self.electrodes:
            shape = electrode.get("shape", "rectangle")

            if shape == "rectangle":
                self._add_rectangle_electrode(electrode)
            elif shape == "from_file":
                raise NotImplementedError(
                    "Loading electrode pattern from file is not yet implemented. "
                    "Use 'rectangle' shape for now."
                )
            else:
                raise ValueError(f"Unknown electrode shape: {shape}")

        return self.electrode_mask

    def _add_rectangle_electrode(self, electrode: Dict) -> None:
        """Add rectangular electrode to mask (as 3D volume)

        New coordinate system: z = 0 (surface, k=0) -> z = -size_z (bottom, k=nz-1)

        Parameters
        ----------
        electrode : Dict
            Electrode definition (x_range, y_range, z_position, etc.)
            z_position is electrode bottom position (negative value), electrode region extends from there to surface (z=0)
        """
        x_range = electrode.get("x_range", [0, 0])
        y_range = electrode.get("y_range", [0, 0])
        z_position = electrode.get("z_position", 0)  # Electrode bottom (negative value)

        # Convert to indices
        i_min = int(x_range[0] / self.h)
        i_max = int(x_range[1] / self.h)
        j_min = int(y_range[0] / self.h)
        j_max = int(y_range[1] / self.h)

        # Electrode extends from surface (k=0, z=0) to z_position (negative value)
        # Since k = -z / h, z_position (negative) -> k_bottom (positive)
        k_top = 0  # Surface (z=0)
        k_bottom = int(-z_position / self.h)  # Electrode bottom

        # Range check
        i_min = max(0, min(i_min, self.nx - 1))
        i_max = max(0, min(i_max, self.nx - 1))
        j_min = max(0, min(j_min, self.ny - 1))
        j_max = max(0, min(j_max, self.ny - 1))
        k_top = max(0, min(k_top, self.nz - 1))
        k_bottom = max(0, min(k_bottom, self.nz - 1))

        # Set mask (array shape: (nz, nx, ny))
        # Electrode region: from k_top to k_bottom
        self.electrode_mask[
            k_top : k_bottom + 1, i_min : i_max + 1, j_min : j_max + 1
        ] = True

    def get_electrode_voltages(self) -> np.ndarray:
        """Get electrode voltage distribution

        Set electrode voltage at each grid point

        Returns
        -------
        electrode_voltages : np.ndarray
            Electrode voltage (V), shape=(nz, nx, ny)
        """
        if self.electrode_voltages is None:
            self._initialize_arrays()

        self.electrode_voltages[:, :, :] = 0.0

        for electrode in self.electrodes:
            voltage = electrode.get("voltage", 0.0)
            shape = electrode.get("shape", "rectangle")

            if shape == "rectangle":
                x_range = electrode.get("x_range", [0, 0])
                y_range = electrode.get("y_range", [0, 0])
                z_position = electrode.get(
                    "z_position", 0
                )  # Electrode bottom (negative value)

                # Convert to indices
                i_min = int(x_range[0] / self.h)
                i_max = int(x_range[1] / self.h)
                j_min = int(y_range[0] / self.h)
                j_max = int(y_range[1] / self.h)

                # Electrode extends from surface (k=0, z=0) to z_position (negative value)
                k_top = 0  # Surface (z=0)
                k_bottom = int(-z_position / self.h)  # Electrode bottom

                # Range check
                i_min = max(0, min(i_min, self.nx - 1))
                i_max = max(0, min(i_max, self.nx - 1))
                j_min = max(0, min(j_min, self.ny - 1))
                j_max = max(0, min(j_max, self.ny - 1))
                k_top = max(0, min(k_top, self.nz - 1))
                k_bottom = max(0, min(k_bottom, self.nz - 1))

                # Set voltage (array shape: (nz, nx, ny))
                self.electrode_voltages[
                    k_top : k_bottom + 1, i_min : i_max + 1, j_min : j_max + 1
                ] = voltage

        return self.electrode_voltages

    def check_electrode_overlap(self) -> None:
        """Check for electrode overlap

        Raises error if overlap exists

        Raises
        ------
        ValueError
            If electrodes overlap
        """
        # Count number of electrodes at each position (array shape: (nz, nx, ny))
        electrode_count = np.zeros((self.nz, self.nx, self.ny), dtype=int)

        for electrode in self.electrodes:
            shape = electrode.get("shape", "rectangle")

            if shape == "rectangle":
                x_range = electrode.get("x_range", [0, 0])
                y_range = electrode.get("y_range", [0, 0])
                z_position = electrode.get(
                    "z_position", 0
                )  # Electrode bottom (negative value)

                # Convert to indices
                i_min = int(x_range[0] / self.h)
                i_max = int(x_range[1] / self.h)
                j_min = int(y_range[0] / self.h)
                j_max = int(y_range[1] / self.h)
                k = int(-z_position / self.h)  # k index of electrode bottom

                # Range check
                i_min = max(0, min(i_min, self.nx - 1))
                i_max = max(0, min(i_max, self.nx - 1))
                j_min = max(0, min(j_min, self.ny - 1))
                j_max = max(0, min(j_max, self.ny - 1))
                k = max(0, min(k, self.nz - 1))

                # Increment count (array shape: (nz, nx, ny))
                electrode_count[k, i_min : i_max + 1, j_min : j_max + 1] += 1

        # Check for overlap
        if np.any(electrode_count > 1):
            overlapping_positions = np.where(electrode_count > 1)
            raise ValueError(
                f"Electrode overlap detected at {len(overlapping_positions[0])} positions. "
                f"First overlap at grid indices (k, i, j): "
                f"({overlapping_positions[0][0]}, {overlapping_positions[1][0]}, {overlapping_positions[2][0]})"
            )

    def set_charge_density(self, rho: np.ndarray) -> None:
        """Set charge density distribution

        Parameters
        ----------
        rho : np.ndarray
            Charge density distribution (C/m^3), shape=(nz, nx, ny)
        """
        if rho.shape != (self.nz, self.nx, self.ny):
            raise ValueError(
                f"Charge density shape {rho.shape} does not match "
                f"grid size ({self.nz}, {self.nx}, {self.ny})"
            )

        self.charge_density = rho.copy()

    def get_grid_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get grid coordinates

        New coordinate system: z = 0 (surface, k=0) -> z = -size_z (bottom, k=nz-1)

        Returns
        -------
        x, y, z : np.ndarray
            Coordinate arrays for each direction (m)
        """
        x = np.arange(self.nx) * self.h
        y = np.arange(self.ny) * self.h
        z = -np.arange(self.nz) * self.h  # z = -k * h (negative direction)

        return x, y, z

    def get_summary(self) -> str:
        """Get structure summary as string

        Returns
        -------
        summary : str
            Structure summary
        """
        summary = []
        summary.append("=== Structure Summary ===")
        summary.append(f"Grid size (nz, nx, ny): ({self.nz}, {self.nx}, {self.ny})")
        summary.append(f"Grid spacing (h): {self.h * 1e9:.2f} nm (isotropic)")
        summary.append(f"Number of layers: {len(self.layers)}")
        summary.append(f"Number of electrodes: {len(self.electrodes)}")

        summary.append("\n--- Layers ---")
        for i, layer in enumerate(self.layers):
            material = layer.get("material", "Unknown")
            z_range = layer.get("z_range", [0, 0])
            epsilon_r = layer.get("epsilon_r", 1.0)
            summary.append(
                f"  {i + 1}. {material}: "
                f"z=[{z_range[0] * 1e9:.1f}, {z_range[1] * 1e9:.1f}] nm, "
                f"εr={epsilon_r:.2f}"
            )

        summary.append("\n--- Electrodes ---")
        for i, electrode in enumerate(self.electrodes):
            name = electrode.get("name", f"electrode_{i + 1}")
            voltage = electrode.get("voltage", 0.0)
            shape = electrode.get("shape", "rectangle")
            summary.append(f"  {i + 1}. {name}: V={voltage:.3f} V, shape={shape}")

        return "\n".join(summary)
