"""Time-dependent potential calculator for gate voltage dynamics

Combines spatial interpolation (PotentialInterpolator) with temporal interpolation
to efficiently compute 2D potential distributions at arbitrary time points.

Mathematical Background
-----------------------
For time-dependent gate voltages V_i(t), the 2D potential distribution is:

    φ(x, y, t) = φ_particular + Σᵢ Vᵢ(t)·φᵢ(x, y)

where:
- φ_particular: Particular solution (constant in time)
- φᵢ(x, y): Basis functions from PotentialInterpolator
- Vᵢ(t): Time-dependent voltage for electrode i

This allows O(1) computation for any time point after initial basis setup.

Example
-------
>>> from structure_manager import StructureManager
>>> from potential_interpolator import PotentialInterpolator
>>> from time_dependent_potential import TimeDependentPotential
>>> import numpy as np
>>>
>>> # Setup spatial interpolator
>>> manager = StructureManager("configs/example.yaml")
>>> interp = PotentialInterpolator(manager, z_position=-20e-9)
>>>
>>> # Define time-dependent voltages
>>> voltages = {
...     "gate1": lambda t: 0.5 * np.sin(2*np.pi*1e9*t),  # 1 GHz sine wave
...     "gate2": ([0, 1e-9, 2e-9], [0.0, 1.0, 0.0]),    # Pulse (discrete)
...     "gate3": 0.3  # Constant voltage
... }
>>>
>>> # Create time-dependent potential
>>> td_pot = TimeDependentPotential(interp, voltages)
>>>
>>> # Get potential at specific time
>>> phi = td_pot(t=0.5e-9)  # t=0.5 ns
>>>
>>> # Get time series
>>> t_array = np.linspace(0, 2e-9, 100)
>>> phi_series = td_pot.get_time_series(t_array)  # shape: (100, nx, ny)
"""

import numpy as np
from typing import Dict, Union, Tuple, Callable, Optional, List
from scipy.interpolate import interp1d
import warnings


class VoltageFunction:
    """Helper class to represent time-dependent voltage for a single electrode

    Supports both analytical functions and discrete data with interpolation.

    Parameters
    ----------
    data : Union[Callable, Tuple[np.ndarray, np.ndarray], float]
        Voltage specification:
        - Callable: Function of time, e.g., lambda t: V(t)
        - Tuple: (t_array, V_array) for discrete data
        - float: Constant voltage (time-independent)
    kind : str, optional
        Interpolation kind for discrete data (default: 'linear')
        Options: 'linear', 'cubic', 'nearest', 'previous', 'next'

    Attributes
    ----------
    t_range : Optional[Tuple[float, float]]
        Valid time range for discrete data (None for analytical functions)

    Examples
    --------
    >>> # Analytical function
    >>> vf1 = VoltageFunction(lambda t: 0.5 * np.sin(t))
    >>> vf1(0.5)
    0.23971...

    >>> # Discrete data
    >>> t = np.array([0, 1e-9, 2e-9])
    >>> V = np.array([0.0, 1.0, 0.0])
    >>> vf2 = VoltageFunction((t, V))
    >>> vf2(0.5e-9)
    0.5

    >>> # Constant voltage
    >>> vf3 = VoltageFunction(0.3)
    >>> vf3(1e-9)
    0.3
    """

    def __init__(
        self,
        data: Union[Callable, Tuple[np.ndarray, np.ndarray], float],
        kind: str = "linear",
    ):
        self.t_range: Optional[Tuple[float, float]] = None

        # Case 1: Constant voltage (float or int)
        if isinstance(data, (float, int)):
            self._func = lambda t: float(data)
            self._is_constant = True
            self._is_callable = False

        # Case 2: Analytical function
        elif callable(data):
            self._func = data
            self._is_constant = False
            self._is_callable = True

        # Case 3: Discrete data (tuple/list of arrays)
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            t_array, V_array = data
            t_array = np.asarray(t_array, dtype=float)
            V_array = np.asarray(V_array, dtype=float)

            if len(t_array) != len(V_array):
                raise ValueError(
                    f"Time and voltage arrays must have same length: "
                    f"{len(t_array)} != {len(V_array)}"
                )

            if len(t_array) < 2:
                raise ValueError("Need at least 2 data points for interpolation")

            # Check if sorted
            if not np.all(np.diff(t_array) > 0):
                raise ValueError("Time array must be strictly increasing")

            self._func = interp1d(
                t_array,
                V_array,
                kind=kind,
                bounds_error=False,
                fill_value=(V_array[0], V_array[-1]),  # Constant extrapolation
            )
            self.t_range = (float(t_array[0]), float(t_array[-1]))
            self._is_constant = False
            self._is_callable = False

        else:
            raise TypeError(
                f"data must be callable, tuple of arrays, or float, got {type(data)}"
            )

    def __call__(self, t: float) -> float:
        """Evaluate voltage at time t

        Parameters
        ----------
        t : float
            Time point (seconds)

        Returns
        -------
        V : float
            Voltage at time t (volts)
        """
        result = self._func(t)
        # Ensure scalar return
        if isinstance(result, np.ndarray):
            return float(result)
        return float(result)

    def __repr__(self) -> str:
        """String representation"""
        if self._is_constant:
            return f"VoltageFunction(constant={self._func(0):.3f} V)"
        elif self._is_callable:
            return "VoltageFunction(analytical function)"
        else:
            return f"VoltageFunction(discrete data, t∈[{self.t_range[0] * 1e9:.2f}, {self.t_range[1] * 1e9:.2f}] ns)"


class TimeDependentPotential:
    """Time-dependent 2D potential distribution calculator

    Combines PotentialInterpolator (spatial) with VoltageFunction (temporal)
    to efficiently compute potential distributions at arbitrary time points.

    Parameters
    ----------
    potential_interpolator : PotentialInterpolator
        Pre-computed spatial interpolator with basis functions
    voltage_functions : Dict[str, Union[Callable, Tuple, float]]
        Time-dependent voltages for each electrode:
        - Callable: lambda t: V(t)
        - Tuple: (t_array, V_array)
        - float: Constant voltage
    interpolation_kind : str, optional
        Interpolation method for discrete data (default: 'linear')
    validate_electrodes : bool, optional
        Check that voltage_functions keys match interpolator electrodes (default: True)

    Attributes
    ----------
    interpolator : PotentialInterpolator
        Spatial interpolator
    voltage_functions : Dict[str, VoltageFunction]
        Time-dependent voltage functions for each electrode
    electrode_names : List[str]
        Names of electrodes
    t_range : Optional[Tuple[float, float]]
        Valid time range (intersection of all discrete data ranges)

    Raises
    ------
    ValueError
        If voltage_functions keys don't match interpolator electrode names

    Examples
    --------
    >>> # Pulse and sine wave combination
    >>> voltages = {
    ...     "gate1": lambda t: 0.5 if t < 1e-9 else 0.0,
    ...     "gate2": lambda t: 1.0 * np.sin(2*np.pi*1e9*t),
    ...     "gate3": 0.3
    ... }
    >>> td_pot = TimeDependentPotential(interp, voltages)
    >>> phi = td_pot(t=0.5e-9)

    >>> # Measured data
    >>> t_meas = np.linspace(0, 2e-9, 50)
    >>> V_meas = 0.5 + 0.3*np.sin(2*np.pi*1e9*t_meas)
    >>> voltages = {"gate1": (t_meas, V_meas), "gate2": 1.0, "gate3": 0.0}
    >>> td_pot = TimeDependentPotential(interp, voltages, interpolation_kind='cubic')
    """

    def __init__(
        self,
        potential_interpolator,  # PotentialInterpolator type
        voltage_functions: Dict[str, Union[Callable, Tuple, float]],
        interpolation_kind: str = "linear",
        validate_electrodes: bool = True,
    ):
        self.interpolator = potential_interpolator
        self.electrode_names = potential_interpolator.electrode_names

        # Validate electrode names
        if validate_electrodes:
            voltage_keys = set(voltage_functions.keys())
            electrode_keys = set(self.electrode_names)

            if voltage_keys != electrode_keys:
                missing = electrode_keys - voltage_keys
                extra = voltage_keys - electrode_keys
                error_msg = "Voltage function keys do not match electrode names.\n"
                if missing:
                    error_msg += f"  Missing: {missing}\n"
                if extra:
                    error_msg += f"  Extra: {extra}\n"
                error_msg += f"  Expected: {electrode_keys}"
                raise ValueError(error_msg)

        # Create VoltageFunction objects
        self.voltage_functions: Dict[str, VoltageFunction] = {}
        for name, data in voltage_functions.items():
            self.voltage_functions[name] = VoltageFunction(
                data, kind=interpolation_kind
            )

        # Determine valid time range (intersection of all discrete ranges)
        self.t_range = self._compute_time_range()

    def _compute_time_range(self) -> Optional[Tuple[float, float]]:
        """Compute valid time range from all voltage functions

        Returns
        -------
        t_range : Optional[Tuple[float, float]]
            Intersection of all discrete data ranges, or None if all analytical
        """
        ranges = [
            vf.t_range
            for vf in self.voltage_functions.values()
            if vf.t_range is not None
        ]

        if not ranges:
            return None  # All analytical or constant

        # Intersection: max of t_min, min of t_max
        t_min = max(r[0] for r in ranges)
        t_max = min(r[1] for r in ranges)

        if t_min >= t_max:
            warnings.warn(
                f"Time ranges have no overlap: t_min={t_min * 1e9:.2f} ns >= "
                f"t_max={t_max * 1e9:.2f} ns"
            )
            return (t_min, t_min)  # Degenerate range

        return (t_min, t_max)

    def get_voltage_at_time(self, t: float) -> Dict[str, float]:
        """Get electrode voltages at specific time

        Parameters
        ----------
        t : float
            Time point (seconds)

        Returns
        -------
        voltages : Dict[str, float]
            Voltage for each electrode at time t

        Examples
        --------
        >>> voltages = td_pot.get_voltage_at_time(t=0.5e-9)
        >>> print(voltages)
        {'gate1': 0.25, 'gate2': 0.95, 'gate3': 0.3}
        """
        return {name: vf(t) for name, vf in self.voltage_functions.items()}

    def __call__(self, t: float) -> np.ndarray:
        """Compute 2D potential distribution at time t

        Parameters
        ----------
        t : float
            Time point (seconds)

        Returns
        -------
        phi_2d : np.ndarray
            2D potential distribution at z=z_position, shape=(nx, ny)

        Examples
        --------
        >>> phi = td_pot(t=1.5e-9)
        >>> phi.shape
        (100, 100)
        """
        voltages = self.get_voltage_at_time(t)
        return self.interpolator(voltages)

    def get_time_series(
        self, t_array: np.ndarray, progress: bool = False
    ) -> np.ndarray:
        """Compute potential distribution time series

        Parameters
        ----------
        t_array : np.ndarray
            Array of time points (seconds)
        progress : bool, optional
            Show progress bar (default: False)

        Returns
        -------
        phi_series : np.ndarray
            Time series of 2D potentials, shape=(len(t_array), nx, ny)

        Examples
        --------
        >>> t_array = np.linspace(0, 2e-9, 100)
        >>> phi_series = td_pot.get_time_series(t_array, progress=True)
        >>> phi_series.shape
        (100, 100, 100)
        """
        n_times = len(t_array)
        nx, ny = self.interpolator.nx, self.interpolator.ny
        phi_series = np.zeros((n_times, nx, ny))

        iterator = range(n_times)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Computing time series")
            except ImportError:
                warnings.warn("tqdm not installed, progress bar disabled")

        for i in iterator:
            phi_series[i] = self(t_array[i])

        return phi_series

    def animate(
        self,
        t_array: np.ndarray,
        save_path: Optional[str] = None,
        fps: int = 30,
        show_voltages: bool = True,
        show_colorbar: bool = True,
        figsize: Tuple[int, int] = (14, 6),
        track_minimum: Optional[Dict] = None,
        show_trajectory: bool = True,
        trajectory_length: int = 10,
    ):
        """Create animation of time-dependent potential

        Parameters
        ----------
        t_array : np.ndarray
            Array of time points (seconds)
        save_path : str, optional
            Path to save animation (e.g., 'animation.gif', 'animation.mp4')
            If None, animation is shown but not saved
        fps : int, optional
            Frames per second for animation (default: 30)
        show_voltages : bool, optional
            Show voltage time series subplot (default: True)
        show_colorbar : bool, optional
            Show colorbar for potential (default: True)
        figsize : Tuple[int, int], optional
            Figure size (width, height) in inches (default: (14, 6))
        track_minimum : dict, optional
            Dictionary from track_local_minimum() to overlay minimum position
            If provided, shows current position, trajectory, start/end markers
        show_trajectory : bool, optional
            Show trajectory trail (last N points). Only used if track_minimum given.
            Default: True
        trajectory_length : int, optional
            Number of past points to show in trajectory. Default: 10

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            Animation object

        Notes
        -----
        Requires matplotlib. For saving to file:
        - .gif: requires pillow
        - .mp4: requires ffmpeg

        Examples
        --------
        >>> t_array = np.linspace(0, 2e-9, 60)
        >>> anim = td_pot.animate(t_array, save_path='potential.gif', fps=20)
        >>>
        >>> # With minimum tracking
        >>> tracking = td_pot.track_local_minimum(t_array, (50e-9, 50e-9))
        >>> anim = td_pot.animate(t_array, track_minimum=tracking,
        ...                        show_trajectory=True, trajectory_length=10)
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib import colormaps

        # Convert coordinates to nm for plotting
        x_nm = self.interpolator.x * 1e9
        y_nm = self.interpolator.y * 1e9

        # Setup figure
        if show_voltages:
            fig, (ax_pot, ax_volt) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax_pot = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))

        # Pre-compute potential range for consistent colorscale
        phi_min, phi_max = self._compute_potential_range(
            t_array[: min(10, len(t_array))]
        )

        # Initialize potential plot
        phi_init = self(t_array[0])
        im = ax_pot.pcolormesh(
            x_nm,
            y_nm,
            phi_init.T,
            shading="auto",
            cmap=colormaps.get_cmap("RdBu_r"),
            vmin=phi_min,
            vmax=phi_max,
        )
        ax_pot.set_xlabel("x (nm)")
        ax_pot.set_ylabel("y (nm)")
        ax_pot.set_aspect("equal")
        ax_pot.set_title(f"t = 0.00 ns")

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax_pot)
            cbar.set_label("Potential (V)")

        # Initialize voltage plot if requested
        if show_voltages:
            voltage_lines = {}
            for i, name in enumerate(self.electrode_names):
                (line,) = ax_volt.plot(
                    [], [], "o-", label=name, linewidth=2, markersize=6
                )
                voltage_lines[name] = line

            ax_volt.set_xlabel("Time (ns)")
            ax_volt.set_ylabel("Voltage (V)")
            ax_volt.set_xlim(t_array[0] * 1e9, t_array[-1] * 1e9)

            # Compute voltage range
            V_min, V_max = self._compute_voltage_range(t_array)
            margin = 0.1 * (V_max - V_min) if V_max > V_min else 0.1
            ax_volt.set_ylim(V_min - margin, V_max + margin)
            ax_volt.grid(True, alpha=0.3)
            ax_volt.legend(loc="upper right")

            # Storage for voltage history
            t_history = []
            V_history = {name: [] for name in self.electrode_names}

        # Initialize minimum tracking plot objects
        minimum_artists = []
        if track_minimum is not None:
            # Start marker (green circle)
            (start_marker,) = ax_pot.plot(
                track_minimum["x"][0] * 1e9,
                track_minimum["y"][0] * 1e9,
                "go",
                markersize=8,
                label="Start",
                zorder=10,
            )
            minimum_artists.append(start_marker)

            # Current position marker (red X)
            (current_marker,) = ax_pot.plot(
                [],
                [],
                "rx",
                markersize=12,
                markeredgewidth=2.5,
                label="Current min",
                zorder=12,
            )
            minimum_artists.append(current_marker)

            # Trajectory line
            if show_trajectory:
                (trajectory_line,) = ax_pot.plot(
                    [], [], "r-", linewidth=1.5, alpha=0.7, zorder=11
                )
                minimum_artists.append(trajectory_line)

            # End marker (red square) - will be shown only on last frame
            (end_marker,) = ax_pot.plot(
                [], [], "rs", markersize=8, label="End", zorder=10
            )
            minimum_artists.append(end_marker)

            ax_pot.legend(loc="upper right")

        def update(frame):
            """Update function for animation"""
            t = t_array[frame]

            # Update potential
            phi = self(t)
            im.set_array(phi.T.ravel())
            ax_pot.set_title(f"t = {t * 1e9:.2f} ns")

            # Update voltages
            if show_voltages:
                voltages = self.get_voltage_at_time(t)
                t_history.append(t * 1e9)

                for name in self.electrode_names:
                    V_history[name].append(voltages[name])
                    voltage_lines[name].set_data(t_history, V_history[name])

            # Update minimum tracking markers
            artists_to_return = [im]
            if show_voltages:
                artists_to_return.extend(voltage_lines.values())

            if track_minimum is not None:
                # Update current position (index 1 in minimum_artists)
                x_current = track_minimum["x"][frame] * 1e9  # Convert to nm
                y_current = track_minimum["y"][frame] * 1e9
                current_marker = minimum_artists[1]
                current_marker.set_data([x_current], [y_current])

                # Update trajectory
                if show_trajectory and len(minimum_artists) > 2:
                    trajectory_line = minimum_artists[2]
                    # Get trajectory slice (last trajectory_length points up to current frame)
                    start_idx = max(0, frame - trajectory_length + 1)
                    end_idx = frame + 1

                    x_traj = track_minimum["x"][start_idx:end_idx] * 1e9
                    y_traj = track_minimum["y"][start_idx:end_idx] * 1e9
                    trajectory_line.set_data(x_traj, y_traj)

                # Show end marker on last frame
                if frame == len(t_array) - 1:
                    end_marker = minimum_artists[-1]  # Last element is end marker
                    x_end = track_minimum["x"][-1] * 1e9
                    y_end = track_minimum["y"][-1] * 1e9
                    end_marker.set_data([x_end], [y_end])
                else:
                    # Hide end marker on other frames
                    end_marker = minimum_artists[-1]
                    end_marker.set_data([], [])

                artists_to_return.extend(minimum_artists)

            return tuple(artists_to_return)

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(t_array),
            interval=1000 / fps,
            blit=False,
            repeat=True,
        )

        plt.tight_layout()

        # Save if path provided
        if save_path:
            print(f"Saving animation to: {save_path}")
            if save_path.endswith(".gif"):
                anim.save(save_path, writer="pillow", fps=fps)
            elif save_path.endswith(".mp4"):
                anim.save(save_path, writer="ffmpeg", fps=fps)
            else:
                anim.save(save_path, fps=fps)
            print("Animation saved successfully")

        return anim

    def _compute_potential_range(self, t_sample: np.ndarray) -> Tuple[float, float]:
        """Compute potential range for colorscale

        Parameters
        ----------
        t_sample : np.ndarray
            Sample of time points for range estimation

        Returns
        -------
        phi_min, phi_max : float
            Minimum and maximum potential values
        """
        phi_min = float("inf")
        phi_max = float("-inf")

        for t in t_sample:
            phi = self(t)
            phi_min = min(phi_min, phi.min())
            phi_max = max(phi_max, phi.max())

        return phi_min, phi_max

    def _compute_voltage_range(self, t_array: np.ndarray) -> Tuple[float, float]:
        """Compute voltage range for all electrodes

        Parameters
        ----------
        t_array : np.ndarray
            Time array

        Returns
        -------
        V_min, V_max : float
            Minimum and maximum voltage values across all electrodes and times
        """
        V_min = float("inf")
        V_max = float("-inf")

        for t in t_array:
            voltages = self.get_voltage_at_time(t)
            V_min = min(V_min, min(voltages.values()))
            V_max = max(V_max, max(voltages.values()))

        return V_min, V_max

    def save(self, filepath: str) -> None:
        """Save TimeDependentPotential to file

        Parameters
        ----------
        filepath : str
            Path to save file (.npz format)

        Notes
        -----
        Saves all voltage function data and interpolator info.
        Analytical functions are NOT saved (will raise error).
        """
        # Check if all voltage functions can be saved
        has_analytical = any(vf._is_callable for vf in self.voltage_functions.values())
        if has_analytical:
            raise ValueError(
                "Cannot save TimeDependentPotential with analytical voltage functions. "
                "Only discrete data and constants can be saved."
            )

        # Save voltage function data
        voltage_data = {}
        for name, vf in self.voltage_functions.items():
            if vf._is_constant:
                voltage_data[f"{name}_type"] = "constant"
                voltage_data[f"{name}_value"] = vf(0)
            else:
                # Discrete data - reconstruct from interpolator
                voltage_data[f"{name}_type"] = "discrete"
                # Note: We need to store original t, V arrays
                # This is a limitation - we store them during creation
                raise NotImplementedError(
                    "Saving discrete data not yet implemented. "
                    "Consider saving original data separately."
                )

        # Save interpolator reference info
        voltage_data["z_position"] = self.interpolator.z_position
        voltage_data["z_index"] = self.interpolator.z_index
        voltage_data["electrode_names"] = np.array(self.electrode_names, dtype=object)

        np.savez(filepath, **voltage_data)
        print(f"TimeDependentPotential saved to: {filepath}")

    def track_local_minimum(
        self,
        t_array: np.ndarray,
        initial_position: Tuple[float, float],
        search_radius: float = 20e-9,
        interpolate_trajectory: bool = True,
        return_curvature: bool = True,
        effective_mass: Optional[float] = None,
        max_steps: int = 1000,
    ) -> Dict:
        """
        Track the position of a local minimum over time.

        Uses a simple steepest descent algorithm: moves to the lowest neighboring
        grid point until a local minimum is reached. The minimum position is then
        refined using parabolic interpolation.

        Useful for shuttling simulations where a quantum dot position follows
        a local minimum in the potential landscape.

        Parameters
        ----------
        t_array : np.ndarray
            Time points to track (s)
        initial_position : tuple[float, float]
            Initial (x, y) position to start search (m)
        search_radius : float, optional
            Maximum allowed distance from previous position (m). Default: 20e-9 (20nm)
        interpolate_trajectory : bool, optional
            Smooth trajectory using spline interpolation. Default: True
        return_curvature : bool, optional
            Return curvature (confinement strength). Default: True
        effective_mass : float, optional
            Effective mass (kg) for eigenfrequency calculation.
            If None, eigenfrequencies are not computed.
        max_steps : int, optional
            Maximum steps for gradient descent. Default: 1000

        Returns
        -------
        dict
            Dictionary with keys:
            - 't': time array (s)
            - 'x': x positions (m)
            - 'y': y positions (m)
            - 'phi_min': potential at minimum (V)
            - 'curvature_x': d²φ/dx² at minimum (V/m²)
            - 'curvature_y': d²φ/dy² at minimum (V/m²)
            - 'eigenfreq_x': ωx = sqrt(q·d²φ/dx² / m*) (rad/s) [if effective_mass given]
            - 'eigenfreq_y': ωy (rad/s) [if effective_mass given]

        Raises
        ------
        ValueError
            If fails to find minimum
        ValueError
            If minimum moves outside search radius

        Examples
        --------
        >>> from scipy.constants import electron_mass
        >>> tracking = td_pot.track_local_minimum(
        ...     t_array,
        ...     initial_position=(50e-9, 50e-9),
        ...     effective_mass=0.19 * electron_mass
        ... )
        >>> print(f"Final position: {tracking['x'][-1]*1e9:.1f} nm")
        >>> print(f"Confinement: {tracking['eigenfreq_x'][-1]/1e9:.2f} GHz")
        """
        # Initialize storage
        x_positions = []
        y_positions = []
        phi_values = []
        curvature_x_values = []
        curvature_y_values = []

        # Find initial minimum
        phi_0 = self(t_array[0])
        try:
            x_min, y_min, phi_min, curv_x, curv_y = (
                self._find_local_minimum_with_parabolic_fit(
                    phi_0,
                    initial_position[0],
                    initial_position[1],
                    max_steps=max_steps,
                )
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to find initial minimum at t={t_array[0]:.2e}s: {e}"
            )

        x_positions.append(x_min)
        y_positions.append(y_min)
        phi_values.append(phi_min)

        if return_curvature:
            curvature_x_values.append(curv_x)
            curvature_y_values.append(curv_y)

        # Track minimum over time
        for t in t_array[1:]:
            phi_t = self(t)

            # Search near previous position
            try:
                x_min, y_min, phi_min, curv_x, curv_y = (
                    self._find_local_minimum_with_parabolic_fit(
                        phi_t,
                        x_positions[-1],
                        y_positions[-1],
                        max_steps=max_steps,
                    )
                )
            except ValueError as e:
                raise ValueError(
                    f"Failed to track minimum at t={t:.2e}s: {e}\n"
                    f"Previous position: ({x_positions[-1] * 1e9:.2f}, {y_positions[-1] * 1e9:.2f}) nm"
                )

            # Check if moved too far (outside search radius)
            dist = np.sqrt(
                (x_min - x_positions[-1]) ** 2 + (y_min - y_positions[-1]) ** 2
            )
            if dist > search_radius:
                raise ValueError(
                    f"Minimum moved {dist * 1e9:.2f} nm, exceeding search radius {search_radius * 1e9:.2f} nm at t={t:.2e}s"
                )

            x_positions.append(x_min)
            y_positions.append(y_min)
            phi_values.append(phi_min)

            if return_curvature:
                curvature_x_values.append(curv_x)
                curvature_y_values.append(curv_y)

        # Convert to arrays
        x_positions = np.array(x_positions)
        y_positions = np.array(y_positions)
        phi_values = np.array(phi_values)

        # Optional trajectory smoothing
        if interpolate_trajectory and len(t_array) > 3:
            from scipy.interpolate import UnivariateSpline

            # Smoothing parameter: smaller = smoother
            s_factor = len(t_array) * 1e-20  # Very small smoothing
            try:
                x_spline = UnivariateSpline(t_array, x_positions, s=s_factor)
                y_spline = UnivariateSpline(t_array, y_positions, s=s_factor)
                x_positions = x_spline(t_array)
                y_positions = y_spline(t_array)
            except Exception:
                # If spline fails, use original positions
                pass

        # Build result dictionary
        result = {
            "t": t_array,
            "x": x_positions,
            "y": y_positions,
            "phi_min": phi_values,
        }

        if return_curvature:
            curvature_x_values = np.array(curvature_x_values)
            curvature_y_values = np.array(curvature_y_values)
            result["curvature_x"] = curvature_x_values
            result["curvature_y"] = curvature_y_values

            # Compute eigenfrequencies if effective mass given
            if effective_mass is not None:
                from scipy.constants import elementary_charge

                # ω = sqrt(q·d²φ/dx² / m*)
                # Handle negative curvature (unconfined direction)
                omega_x = np.sqrt(
                    np.abs(elementary_charge * curvature_x_values / effective_mass)
                ) * np.sign(curvature_x_values)
                omega_y = np.sqrt(
                    np.abs(elementary_charge * curvature_y_values / effective_mass)
                ) * np.sign(curvature_y_values)
                result["eigenfreq_x"] = omega_x
                result["eigenfreq_y"] = omega_y

        return result

    def _find_local_minimum_with_parabolic_fit(
        self,
        phi: np.ndarray,
        x_init: float,
        y_init: float,
        max_steps: int = 1000,
    ) -> Tuple[float, float, float, float, float]:
        """
        Find local minimum using steepest descent + parabolic interpolation.

        First finds the minimum grid point using steepest descent, then refines
        the position using parabolic interpolation in each direction.

        Parameters
        ----------
        phi : np.ndarray
            2D potential array, shape (nx, ny)
        x_init, y_init : float
            Initial position (m)
        max_steps : int
            Maximum number of steps for descent

        Returns
        -------
        x_min : float
            x position of minimum (m)
        y_min : float
            y position of minimum (m)
        phi_min : float
            Potential at minimum (V)
        curv_x : float
            Curvature d²φ/dx² at minimum (V/m²)
        curv_y : float
            Curvature d²φ/dy² at minimum (V/m²)

        Raises
        ------
        ValueError
            If fails to converge
        """
        # Step 1: Find minimum grid point using steepest descent
        x_grid = self.interpolator.x
        y_grid = self.interpolator.y

        # Find nearest grid point to initial position
        x_idx = np.argmin(np.abs(x_grid - x_init))
        y_idx = np.argmin(np.abs(y_grid - y_init))

        # Iterative descent to find minimum grid point
        for step in range(max_steps):
            current_phi = phi[x_idx, y_idx]

            # Check 8 neighbors (including diagonals)
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue

                    # Handle periodic boundaries
                    ni = (x_idx + di) % phi.shape[0]
                    nj = (y_idx + dj) % phi.shape[1]

                    neighbors.append((phi[ni, nj], ni, nj))

            # Find lowest neighbor
            min_phi, min_i, min_j = min(neighbors, key=lambda p: p[0])

            # If current point is lowest, we found minimum grid point
            if current_phi <= min_phi:
                break

            # Move to lowest neighbor
            x_idx, y_idx = min_i, min_j
        else:
            raise ValueError(
                f"Steepest descent did not converge after {max_steps} steps"
            )

        # Step 2: Refine using parabolic interpolation
        # Get 3 points in x direction (centered at minimum)
        if x_idx == 0:
            x_indices = [0, 1, 2]
        elif x_idx == phi.shape[0] - 1:
            x_indices = [phi.shape[0] - 3, phi.shape[0] - 2, phi.shape[0] - 1]
        else:
            x_indices = [x_idx - 1, x_idx, x_idx + 1]

        x_points = x_grid[x_indices]
        phi_x = phi[x_indices, y_idx]

        # Fit parabola: φ(x) = a*(x - x0)² + b
        # Use 3-point formula for parabolic minimum
        x_min, phi_min_x, curv_x = self._parabolic_minimum_1d(x_points, phi_x)

        # Get 3 points in y direction (at refined x position)
        if y_idx == 0:
            y_indices = [0, 1, 2]
        elif y_idx == phi.shape[1] - 1:
            y_indices = [phi.shape[1] - 3, phi.shape[1] - 2, phi.shape[1] - 1]
        else:
            y_indices = [y_idx - 1, y_idx, y_idx + 1]

        y_points = y_grid[y_indices]
        phi_y = phi[x_idx, y_indices]

        # Fit parabola in y direction
        y_min, phi_min_y, curv_y = self._parabolic_minimum_1d(y_points, phi_y)

        # Final potential value at (x_min, y_min)
        # Average the two estimates
        phi_min = (phi_min_x + phi_min_y) / 2

        return x_min, y_min, phi_min, curv_x, curv_y

    def _parabolic_minimum_1d(
        self,
        x_points: np.ndarray,
        phi_points: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Find minimum of parabola fitted to 3 points.

        Fits φ(x) = a*(x - x0)² + b to 3 points and returns minimum.

        Parameters
        ----------
        x_points : np.ndarray
            3 x-coordinates
        phi_points : np.ndarray
            3 φ values

        Returns
        -------
        x_min : float
            Position of minimum
        phi_min : float
            Potential at minimum
        curvature : float
            Second derivative d²φ/dx² = 2a

        Notes
        -----
        Uses 3-point parabolic fit formula:
        x_min = x1 - 0.5 * (x1-x0)² * (φ1-φ2) - (x1-x2)² * (φ1-φ0)
                     / ((x1-x0)*(φ1-φ2) - (x1-x2)*(φ1-φ0))
        """
        x0, x1, x2 = x_points
        phi0, phi1, phi2 = phi_points

        # Parabolic fit using 3 points
        # https://en.wikipedia.org/wiki/Successive_parabolic_interpolation
        denom = (x0 - x1) * (phi0 - phi2) - (x0 - x2) * (phi0 - phi1)

        if abs(denom) < 1e-15:
            # Points are collinear or nearly so, return center point
            return x1, phi1, (phi2 - 2 * phi1 + phi0) / (x2 - x0) ** 2

        # Position of parabolic minimum
        numer = (x0 - x1) ** 2 * (phi0 - phi2) - (x0 - x2) ** 2 * (phi0 - phi1)
        x_min = x0 - 0.5 * numer / denom

        # Clamp to stay within interval [x0, x2]
        x_min = np.clip(x_min, min(x_points), max(x_points))

        # Evaluate parabola at minimum
        # Fit coefficients: φ = a*(x-x1)² + b*(x-x1) + c
        # Using Lagrange interpolation
        L0 = ((x_min - x1) * (x_min - x2)) / ((x0 - x1) * (x0 - x2))
        L1 = ((x_min - x0) * (x_min - x2)) / ((x1 - x0) * (x1 - x2))
        L2 = ((x_min - x0) * (x_min - x1)) / ((x2 - x0) * (x2 - x1))
        phi_min = L0 * phi0 + L1 * phi1 + L2 * phi2

        # Curvature (second derivative): d²φ/dx²
        # From finite difference: d²φ/dx² ≈ (φ2 - 2*φ1 + φ0) / h²
        h = x1 - x0  # Assume equal spacing
        curvature = (phi2 - 2 * phi1 + phi0) / h**2

        return x_min, phi_min, curvature

    def __repr__(self) -> str:
        """String representation"""
        n_analytical = sum(vf._is_callable for vf in self.voltage_functions.values())
        n_discrete = sum(
            vf.t_range is not None for vf in self.voltage_functions.values()
        )
        n_constant = sum(vf._is_constant for vf in self.voltage_functions.values())

        t_range_str = (
            "all times"
            if self.t_range is None
            else f"[{self.t_range[0] * 1e9:.2f}, {self.t_range[1] * 1e9:.2f}] ns"
        )

        return (
            f"TimeDependentPotential(\n"
            f"  electrodes: {len(self.electrode_names)}\n"
            f"  analytical: {n_analytical}, discrete: {n_discrete}, constant: {n_constant}\n"
            f"  valid range: {t_range_str}\n"
            f"  z_position: {self.interpolator.z_position * 1e9:.2f} nm\n"
            f")"
        )
