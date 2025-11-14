"""Example: Track local minimum in time-dependent potential

Demonstrates how to use TimeDependentPotential.track_local_minimum()
to track a potential minimum as it moves during shuttling.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from structure_manager import StructureManager
from potential_interpolator import PotentialInterpolator
from time_dependent_potential import TimeDependentPotential
from scipy.constants import electron_mass

# Load structure
manager = StructureManager("configs/example.yaml")

# Create interpolator at Si/SiO2 interface (z=-20nm)
print("Creating potential interpolator...")
interp = PotentialInterpolator(
    manager,
    z_position=-20e-9,
    omega=1.8,
    tolerance=1e-6,
    verbose=1
)

# Define time-dependent voltages for shuttling
# Simple example: move minimum from gate 1 to gate 3
t_array = np.linspace(0, 10e-9, 100)  # 0 to 10 ns, 100 steps

def voltage_func_1(t):
    """Gate 1: High -> Low"""
    return 0.5 * (1 - t / 10e-9)  # 0.5V -> 0V

def voltage_func_2(t):
    """Gate 2: Constant"""
    return 0.25  # Stay at 0.25V

def voltage_func_3(t):
    """Gate 3: Low -> High"""
    return 0.5 * (t / 10e-9)  # 0V -> 0.5V

voltage_functions = {
    "finger_gate_1": voltage_func_1,
    "finger_gate_2": voltage_func_2,
    "finger_gate_3": voltage_func_3,
}

# Create time-dependent potential
print("\nCreating time-dependent potential...")
td_pot = TimeDependentPotential(interp, voltage_functions)

# Initial guess for minimum position (near gate 1)
# Estimate from domain: assuming gate 1 is on the left
initial_position = (-30e-9, 0.0)  # x=-30nm, y=0

# Track minimum
print("\nTracking local minimum...")
track_result = td_pot.track_local_minimum(
    t_array=t_array,
    initial_position=initial_position,
    search_radius=20e-9,  # 20nm search radius
    interpolate_trajectory=True,
    return_curvature=True,
    effective_mass=0.19 * electron_mass,  # Si effective mass
)

# Print results
print("\nResults:")
print(f"  Time points: {len(track_result['t'])}")
print(f"  Initial position: ({track_result['x'][0]*1e9:.2f}, {track_result['y'][0]*1e9:.2f}) nm")
print(f"  Final position: ({track_result['x'][-1]*1e9:.2f}, {track_result['y'][-1]*1e9:.2f}) nm")
print(f"  Displacement: {np.linalg.norm([track_result['x'][-1] - track_result['x'][0], track_result['y'][-1] - track_result['y'][0]])*1e9:.2f} nm")
print(f"  Initial potential: {track_result['phi_min'][0]:.4f} V")
print(f"  Final potential: {track_result['phi_min'][-1]:.4f} V")

if 'eigenfreq_x' in track_result:
    print(f"  Initial eigenfreq (x): {track_result['eigenfreq_x'][0]/1e9:.2f} GHz")
    print(f"  Final eigenfreq (x): {track_result['eigenfreq_x'][-1]/1e9:.2f} GHz")

# Create animation with minimum tracking
print("\nCreating animation...")
td_pot.animate(
    t_array=t_array,
    save_path="minimum_tracking_animation.mp4",
    fps=30,
    show_voltages=True,
    track_minimum=track_result,
    show_trajectory=True,
    trajectory_length=20,  # Show last 20 points in trajectory
)

print("\nAnimation saved to: minimum_tracking_animation.mp4")
