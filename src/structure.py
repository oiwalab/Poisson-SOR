"""Semiconductor structure management class

Manages material layer structure, electrode placement, and permittivity distribution
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from materials import Material, get_material


class StructureManager:
    """Semiconductor structure management class

    Loads structure definitions from YAML file,
    generates permittivity distribution, electrode mask, and voltage distribution
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config: Dict = {}
        self.layers: List[Dict] = []
        self.electrodes: List[Dict] = []
        self.boundary_conditions: Dict = {}

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

        # Material data (efficient storage)
        self.unique_materials: List[Material] = []  # Unique materials only
        self.material_indices: Optional[np.ndarray] = None  # Index array (nz,)

        # Path to config file (for resolving relative paths)
        self._config_dir: Optional[Path] = None

        if config_path is not None:
            self.load_from_yaml(config_path)

    def load_from_yaml(self, yaml_path: str | Path) -> None:
        """Load structure definition from YAML file

        Parameters
        ----------
        yaml_path : str or Path
            Path to YAML configuration file
        """
        yaml_path = Path(yaml_path)
        self._config_dir = yaml_path.parent.resolve()

        with open(yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Extract settings and convert string numbers to float
        self.layers = self._convert_to_numeric(self.config.get("layers", []))
        self.electrodes = self._convert_to_numeric(self.config.get("electrodes", []))
        self.boundary_conditions = self._convert_to_numeric(
            self.config.get("boundary_conditions", {})
        )
        domain = self._convert_to_numeric(self.config.get("domain", {}))

        # Set computational domain
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

        # Generate material data (unique materials + indices)
        self.generate_materials_data()

        # Generate permittivity distribution
        self.generate_epsilon_array()

        # Generate electrode structure
        if self.electrodes:
            self.generate_electrode_mask()
            self.get_electrode_voltages()
            self.check_electrode_overlap()

    def _convert_to_numeric(self, obj):
        """Recursively convert string numbers to float in nested structures

        Converts all numeric strings to float in dictionaries and lists.
        Handles nested structures recursively.

        Parameters
        ----------
        obj : any
            Object to convert (can be dict, list, str, or any other type)

        Returns
        -------
        converted : any
            Object with numeric strings converted to float
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_numeric(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_numeric(item) for item in obj]
        elif isinstance(obj, str):
            try:
                return float(obj)
            except ValueError:
                return obj
        else:
            return obj

    def generate_materials_data(self) -> Tuple[List[Material], np.ndarray]:
        """Generate unique materials and index mapping

        Creates efficient material storage:
        - unique_materials: List of unique Material objects only
        - material_indices: Array mapping each z-layer to material index

        Returns
        -------
        unique_materials : List[Material]
            List of unique Material objects
        material_indices : np.ndarray
            Index array mapping z-layer to material, shape=(nz,), dtype=int

        Notes
        -----
        This is much more efficient than storing nz Material objects.
        For example, with 3 layers across 100 z-points:
        - Old: 100 Material objects
        - New: 3 Material objects + 100 integers
        """
        unique_materials = []
        material_map = {}  # key: (material_name, overrides_tuple) -> index
        material_indices = np.zeros(self.nz, dtype=int)

        for k in range(self.nz):
            z_coord = -k * self.h

            # Find which layer this z coordinate belongs to
            # Convention: z_min < z <= z_max (upper layer includes boundary)
            # Exception: Bottom-most layer includes both boundaries
            material_found = False
            for i, layer in enumerate(self.layers):
                z_range = layer.get("z_range", [0, 0])
                z_max, z_min = z_range[0], z_range[1]

                # Check if this is the bottom-most layer
                is_bottom_layer = (i == len(self.layers) - 1)

                # Upper layer includes upper boundary, excludes lower boundary
                # Bottom layer includes both boundaries
                # Tolerance for numerical errors (1e-15 for float64 precision)
                if is_bottom_layer:
                    # Bottom layer: z_min <= z <= z_max
                    if z_min - 1e-15 <= z_coord <= z_max + 1e-15:
                        material_found = True
                else:
                    # Other layers: z_min < z <= z_max
                    if z_min + 1e-15 < z_coord <= z_max + 1e-15:
                        material_found = True

                if material_found:
                    material_name = layer.get("material", "Unknown")

                    # Collect overrides from YAML
                    overrides = {}
                    for key in [
                        "epsilon_r",
                        "electron_affinity",
                        "band_gap",
                        "effective_mass_e",
                        "effective_mass_h",
                    ]:
                        if key in layer:
                            overrides[key] = layer[key]

                    # Create unique key for this material configuration
                    override_key = tuple(sorted(overrides.items()))
                    mat_key = (material_name, override_key)

                    # Check if this material configuration already exists
                    if mat_key not in material_map:
                        # Create new unique material
                        mat = get_material(
                            material_name, overrides if overrides else None
                        )
                        material_map[mat_key] = len(unique_materials)
                        unique_materials.append(mat)

                    # Store index for this z-layer
                    material_indices[k] = material_map[mat_key]
                    break

            if not material_found:
                raise ValueError(
                    f"No material found for z={z_coord * 1e9:.2f} nm (k={k})"
                )

        self.unique_materials = unique_materials
        self.material_indices = material_indices

        return unique_materials, material_indices

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
            layer_name = layer.get("name", "Unknown")

            # Check range validity (in new coordinate system z_range[0] > z_range[1])
            if z_range[0] <= z_range[1]:
                raise ValueError(
                    f"Layer '{layer_name}': Invalid z_range {z_range}. "
                    f"z_range must be [z_max, z_min] with z_max > z_min"
                )

            # Check if within computational domain
            if z_range[0] > domain_z_top or z_range[1] < domain_z_bottom:
                raise ValueError(
                    f"Layer '{layer_name}': z_range [{z_range[0] * 1e9:.2f}, {z_range[1] * 1e9:.2f}] nm "
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
                            f"Gap detected between layer '{layer_name}' "
                            f"(ends at {current_z_min * 1e9:.2f} nm) and "
                            f"next layer (starts at {next_z_max * 1e9:.2f} nm). "
                            f"Gap size: {gap * 1e9:.2f} nm"
                        )
                    else:
                        raise ValueError(
                            f"Overlap detected between layer '{layer_name}' "
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

        # Set permittivity for each z-layer using material indices
        for k in range(self.nz):
            mat_idx = self.material_indices[k]
            epsilon_r = self.unique_materials[mat_idx].epsilon_r
            self.epsilon_array[k, :, :] = epsilon_r

        return self.epsilon_array

    def get_material_at_z(self, k: int) -> Material:
        """Get material at specific z-layer index

        Parameters
        ----------
        k : int
            z-layer index (0 to nz-1)

        Returns
        -------
        material : Material
            Material object at this z-layer

        Raises
        ------
        IndexError
            If k is out of range
        """
        if not (0 <= k < self.nz):
            raise IndexError(f"z-layer index k={k} out of range [0, {self.nz - 1}]")

        mat_idx = self.material_indices[k]
        return self.unique_materials[mat_idx]

    def generate_electrode_mask(self) -> np.ndarray:
        """Generate electrode mask

        3D boolean array with True at electrode positions.
        Supports both rectangle shapes and GDS file sources.

        Returns
        -------
        electrode_mask : np.ndarray
            Electrode mask (nz, nx, ny), dtype=bool
        """
        if self.electrode_mask is None:
            self._initialize_arrays()

        self.electrode_mask[:, :, :] = False

        for electrode in self.electrodes:
            source = electrode.get("source", "yaml")

            if source == "gds":
                self._add_gds_electrodes(electrode)
            else:
                shape = electrode.get("shape", "rectangle")

                if shape == "rectangle":
                    self._add_rectangle_electrode(electrode)
                elif shape == "from_file":
                    raise NotImplementedError(
                        "shape='from_file' is deprecated. Use source='gds' instead."
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

    def _add_gds_electrodes(self, gds_config: Dict) -> None:
        """Add electrodes from GDS file

        Parameters
        ----------
        gds_config : Dict
            GDS electrode configuration containing:
            - gds_file: path to GDS file (relative to YAML or absolute)
            - gds_unit: conversion factor (default: 1e-6 for um)
            - origin: [x, y] offset in meters
            - cell_name: optional cell name (default: top-level cell)
            - layer_mapping: list of layer configurations
        """
        from gds_loader import GDSLoader

        gds_file = gds_config.get("gds_file")
        if not gds_file:
            raise ValueError("gds_file must be specified for GDS electrodes")

        # Resolve path relative to YAML config file
        gds_path = Path(gds_file)
        if not gds_path.is_absolute() and self._config_dir is not None:
            gds_path = self._config_dir / gds_path

        gds_unit = gds_config.get("gds_unit", 1e-6)
        origin = tuple(gds_config.get("origin", [0.0, 0.0]))
        cell_name = gds_config.get("cell_name")

        loader = GDSLoader(
            gds_path=gds_path,
            gds_unit=gds_unit,
            origin=origin,
            cell_name=cell_name,
        )

        # Get grid coordinates
        x_coords = np.arange(self.nx) * self.h
        y_coords = np.arange(self.ny) * self.h

        # Process each layer mapping
        for layer_config in gds_config.get("layer_mapping", []):
            layer = layer_config.get("layer")
            if layer is None:
                raise ValueError("layer must be specified in layer_mapping")

            datatype = layer_config.get("datatype", 0)
            z_position = layer_config.get("z_position", 0.0)

            # Rasterize this layer to 2D mask
            xy_mask = loader.rasterize_layer(layer, datatype, x_coords, y_coords)

            # Convert to 3D mask (extend in z direction from surface to z_position)
            k_top = 0
            k_bottom = int(-z_position / self.h)
            k_bottom = max(0, min(k_bottom, self.nz - 1))

            # Set electrode mask for this layer
            for k in range(k_top, k_bottom + 1):
                self.electrode_mask[k, :, :] |= xy_mask

    def get_electrode_voltages(self) -> np.ndarray:
        """Get electrode voltage distribution

        Set electrode voltage at each grid point.
        Supports both rectangle shapes and GDS file sources.

        Returns
        -------
        electrode_voltages : np.ndarray
            Electrode voltage (V), shape=(nz, nx, ny)
        """
        if self.electrode_voltages is None:
            self._initialize_arrays()

        self.electrode_voltages[:, :, :] = 0.0

        for electrode in self.electrodes:
            source = electrode.get("source", "yaml")

            if source == "gds":
                self._set_gds_electrode_voltages(electrode)
            else:
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

    def _set_gds_electrode_voltages(self, gds_config: Dict) -> None:
        """Set voltages for GDS-based electrodes

        Parameters
        ----------
        gds_config : Dict
            GDS electrode configuration (same as _add_gds_electrodes)
        """
        from gds_loader import GDSLoader

        gds_file = gds_config.get("gds_file")
        if not gds_file:
            return

        # Resolve path relative to YAML config file
        gds_path = Path(gds_file)
        if not gds_path.is_absolute() and self._config_dir is not None:
            gds_path = self._config_dir / gds_path

        gds_unit = gds_config.get("gds_unit", 1e-6)
        origin = tuple(gds_config.get("origin", [0.0, 0.0]))
        cell_name = gds_config.get("cell_name")

        loader = GDSLoader(
            gds_path=gds_path,
            gds_unit=gds_unit,
            origin=origin,
            cell_name=cell_name,
        )

        # Get grid coordinates
        x_coords = np.arange(self.nx) * self.h
        y_coords = np.arange(self.ny) * self.h

        # Process each layer mapping
        for layer_config in gds_config.get("layer_mapping", []):
            layer = layer_config.get("layer")
            if layer is None:
                continue

            datatype = layer_config.get("datatype", 0)
            voltage = layer_config.get("voltage", 0.0)
            z_position = layer_config.get("z_position", 0.0)

            # Rasterize this layer to 2D mask
            xy_mask = loader.rasterize_layer(layer, datatype, x_coords, y_coords)

            # Set voltages in 3D
            k_top = 0
            k_bottom = int(-z_position / self.h)
            k_bottom = max(0, min(k_bottom, self.nz - 1))

            for k in range(k_top, k_bottom + 1):
                self.electrode_voltages[k, xy_mask] = voltage

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

        summary.append("\n--- Unique Materials ---")
        for i, mat in enumerate(self.unique_materials):
            summary.append(
                f"  {i}. {mat.name}: εr={mat.epsilon_r:.2f}, "
                f"χ={mat.electron_affinity:.3f} eV, Eg={mat.band_gap:.3f} eV"
            )

        summary.append("\n--- Layers ---")
        for i, layer in enumerate(self.layers):
            material = layer.get("material", "Unknown")
            z_range = layer.get("z_range", [0, 0])
            summary.append(
                f"  {i + 1}. {material}: "
                f"z=[{z_range[0] * 1e9:.1f}, {z_range[1] * 1e9:.1f}] nm"
            )

        summary.append("\n--- Electrodes ---")
        for i, electrode in enumerate(self.electrodes):
            name = electrode.get("name", f"electrode_{i + 1}")
            voltage = electrode.get("voltage", 0.0)
            shape = electrode.get("shape", "rectangle")
            summary.append(f"  {i + 1}. {name}: V={voltage:.3f} V, shape={shape}")

        return "\n".join(summary)

