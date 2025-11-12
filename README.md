# SOR Poisson Solver

3D Poisson equation solver using the Successive Over-Relaxation (SOR) method for semiconductor heterostructure simulations with band structure analysis.

## Overview

This solver solves the Poisson equation $-\nabla \cdot (\varepsilon \nabla \phi) = \rho$ in three dimensions for systems with non-uniform permittivity distributions. Designed for semiconductor heterostructure simulations at cryogenic temperatures (4K) with multiple material layers and electrode configurations.

## Features

- **3D Poisson Solver**: SOR method with Numba JIT compilation for performance
- **Heterostructure Support**: Multi-layer semiconductor structures (Si, SiO2, etc.)
- **Band Structure Calculation**: Automatic computation of conduction and valence band edges
- **Electrode Configuration**: 3D volumetric electrodes with voltage control
- **Boundary Conditions**: Dirichlet, Neumann, and periodic options
- **Fast Interpolation**: Linear interpolation for rapid potential computation at different voltages
- **Visualization**: Potential distribution, band diagrams, convergence history
- **YAML Configuration**: Human-readable structure definitions

## Requirements

- Python 3.12+
- numpy 2.3.4+
- numba 0.62.1+
- matplotlib 3.10.7+
- pyyaml 6.0.3+
- pytest 8.4.2+

## Installation

Install dependencies using uv:

```bash
uv sync
```

For development dependencies:

```bash
uv sync --dev
```

## Quick Start

### Basic Usage

```python
from structure_manager import StructureManager
from poisson_solver import PoissonSolver
import visualizer as vis

# Load structure from YAML
manager = StructureManager("configs/example.yaml")

# Create solver
solver = PoissonSolver(manager.params, omega=1.8, tolerance=1e-6)

# Solve Poisson equation
result = solver.solve()

# Access results
phi = result.phi          # Potential distribution (nz, nx, ny)
Ec = result.compute_Ec()  # Conduction band edge (eV)
Ev = result.compute_Ev()  # Valence band edge (eV)

# Visualize
vis.plot_band_diagram_1d(result)
vis.plot_potential_slice(result.phi, result.x, result.y, result.z, z_index=20)
```

### Fast Voltage Interpolation

For rapid computation of potentials at different electrode voltages:

```python
from potential_interpolator import PotentialInterpolator

# Create interpolator at Si/SiO2 interface (z=-20nm)
interp = PotentialInterpolator(
    manager,
    solver,
    z_position=-20e-9,  # Physical coordinate
    charge_density=None  # Optional charge density
)

# Compute potential for any voltage combination (fast!)
voltages = {
    "finger_gate_1": 0.5,
    "finger_gate_2": 1.0,
    "finger_gate_3": 0.5
}
phi_2d = interp(voltages)  # Returns (nx, ny) array at z=-20nm

# Save interpolator for later use
interp.save("interpolator_z-20nm.npz")

# Load and reuse
interp = PotentialInterpolator.load("interpolator_z-20nm.npz")
```

The interpolator pre-computes basis functions (one per electrode) and uses linear superposition:
```
φ(V₁, V₂, ..., Vₙ) = φ_particular + Σᵢ Vᵢ·φᵢ
```
This is mathematically exact for linear systems and ~1000× faster than re-solving.

## Configuration File

Define structures in YAML format ([configs/example.yaml](configs/example.yaml)):

```yaml
# Computational domain
domain:
  size: [100e-9, 100e-9, 100e-9]  # [x, y, z] in meters
  grid_spacing: 1e-9               # Isotropic grid spacing (1nm)

# Material layers (z=0 is surface, extends in negative direction)
layers:
  - material: "SiO2"
    z_range: [0, -20e-9]           # 0nm to -20nm: oxide layer
    epsilon_r: 3.9                  # Optional override

  - material: "Si"
    z_range: [-20e-9, -100e-9]     # -20nm to -100nm: silicon substrate

# Electrodes (3D volumes)
electrodes:
  - name: "finger_gate_1"
    shape: "rectangle"
    x_range: [10e-9, 30e-9]
    y_range: [0, 100e-9]
    z_position: -5e-9              # Electrode bottom (extends to z=0)
    voltage: 0.5

  - name: "finger_gate_2"
    shape: "rectangle"
    x_range: [40e-9, 60e-9]
    y_range: [0, 100e-9]
    z_position: -5e-9
    voltage: 1.0

# Boundary conditions
boundary_conditions:
  z_top:
    type: "neumann"
    value: 0.0                     # ∂φ/∂z = 0
  z_bottom:
    type: "dirichlet"
    value: 0.0                     # φ = 0
  x_sides:
    type: "periodic"
    value: 0.0
  y_sides:
    type: "periodic"
    value: 0.0
```

## Examples

Run the main example:

```bash
uv run python examples/example.py
```

Run the interpolator example:

```bash
uv run python examples/interpolator_example.py
```

Run the tutorial notebook:

```bash
uv run jupyter notebook examples/tutorial.ipynb
```

## Project Structure

```
.
├── src/
│   ├── materials.py              # Material database (Si, SiO2, etc.)
│   ├── structure_manager.py      # Structure and grid management
│   ├── poisson_solver.py         # SOR Poisson solver (JIT compiled)
│   ├── solver_result.py          # Results container with band structure
│   ├── potential_interpolator.py # Fast voltage interpolation
│   └── visualizer.py             # Visualization utilities
├── configs/
│   └── example.yaml              # Example configuration (Si/SiO2 with finger gates)
├── examples/
│   ├── example.py                # Basic usage example
│   ├── interpolator_example.py   # Interpolation demo
│   └── tutorial.ipynb            # Interactive tutorial
├── tests/
│   ├── test_config_small.yaml    # Small test configuration (21³ grid)
│   ├── test_materials.py         # Material database tests
│   ├── test_structure_manager.py # Structure manager tests
│   ├── test_solver.py            # Solver tests with physics validation
│   └── test_interpolator.py     # Interpolation tests (17 tests)
└── README.md
```

## Testing

Run all tests:

```bash
uv run pytest -v
```

Run specific test module:

```bash
uv run pytest tests/test_interpolator.py -v
```

Run with coverage report:

```bash
uv run pytest --cov=src --cov-report=html
```

Test results (as of latest commit):
- `test_materials.py`: Material database validation
- `test_structure_manager.py`: Structure loading and grid generation
- `test_solver.py`: Physics-based solver validation (parallel plate, point charge, band bending)
- `test_interpolator.py`: 17 tests covering initialization, interpolation, accuracy, save/load (1.7s runtime)

## Development

This project uses:
- **uv** for package management
- **ruff** for linting and formatting
- **pytest** for testing
- **numba** for JIT compilation

Development commands (see [CLAUDE.md](CLAUDE.md) for full details):

```bash
# Linting and formatting
uv run ruff check src/
uv run ruff format src/

# Testing
uv run pytest -v
uv run pytest --lf                # Run last failed
uv run pytest -k "test_pattern"   # Run tests matching pattern
```

Development principles:
- YAGNI (You Aren't Gonna Need It): implement only when needed
- Early development: breaking changes acceptable for better design
- English-only code comments (docstrings can contain Unicode for notation)

## API Reference

### Core Classes

#### `StructureManager`
Manages structure definitions, grid generation, and material properties.

```python
manager = StructureManager("config.yaml")
params = manager.params  # Dictionary for PoissonSolver
x, y, z = manager.get_grid_coordinates()
```

#### `PoissonSolver`
Solves the Poisson equation using SOR method.

```python
solver = PoissonSolver(
    params,
    omega=1.8,           # SOR relaxation parameter (1 < ω < 2)
    tolerance=1e-6,      # Convergence threshold
    max_iterations=10000
)
result = solver.solve(rho=None, phi_initial=None, verbose=True)
```

#### `SolverResult`
Container for solution with band structure methods.

```python
result = solver.solve()
phi = result.phi                    # Potential (V), shape=(nz, nx, ny)
Ec = result.compute_Ec()            # Conduction band (eV)
Ev = result.compute_Ev()            # Valence band (eV)
z, Ec_1d, Ev_1d, phi_1d = result.get_band_diagram_1d(x_idx, y_idx)

# Save/load
result.save("result.npz")
result = SolverResult.load("result.npz")
```

#### `PotentialInterpolator`
Fast interpolation for different electrode voltages.

```python
interp = PotentialInterpolator(
    manager,
    solver,
    z_position=-20e-9,      # Physical z-coordinate (m)
    charge_density=None,    # Optional charge density
    verbose=True
)

# Interpolate
voltages = {"gate1": 0.5, "gate2": 1.0}
phi_2d = interp(voltages)           # or interp.interpolate(voltages)

# Save/load
interp.save("interp.npz")
interp = PotentialInterpolator.load("interp.npz")
```

### Material Database

Built-in materials (at 4K):
- **Si** (Silicon): εᵣ=11.7, χ=4.05 eV, Eg=1.12 eV
- **SiO2** (Silicon Dioxide): εᵣ=3.9, χ=0.9 eV, Eg=9.0 eV

```python
from materials import get_material, list_materials

mat = get_material("Si")
mat = get_material("Si", overrides={"epsilon_r": 11.9})  # Custom parameters
available = list_materials()
```

## Coordinate System

The solver uses a coordinate system where:
- **z = 0**: Surface (electrode side)
- **z-axis**: Extends in negative direction (z = -10nm is 10nm below surface)
- **Array indexing**: `(nz, nx, ny)` where `k=0` is the surface

Example:
```
z = 0nm      ← Surface (electrodes)
z = -10nm    ← 10nm depth
z = -100nm   ← 100nm depth (domain bottom)
```

## Band Structure Formulas

Conduction band edge:
```
Ec(r) = -q·φ(r) - χ(z)  [eV]
```

Valence band edge:
```
Ev(r) = Ec(r) - Eg(z)  [eV]
```

where:
- φ: Electrostatic potential (V)
- χ: Electron affinity (eV)
- Eg: Band gap (eV)

## Performance Notes

- **JIT Compilation**: First solve takes ~2-3× longer due to Numba compilation
- **Grid Size**: Computation time scales as O(N³) where N is grid points per dimension
- **Interpolation**: ~1000× faster than re-solving once basis functions are computed
- **Typical Runtime**: 100³ grid ~5-10s per solve, 20³ grid ~0.5s per solve

## Citation

If you use this code in your research, please cite:

```
(Add citation information if applicable)
```

## License

(Add license information if applicable)

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass (`uv run pytest`)
2. Code follows ruff formatting (`uv run ruff format src/`)
3. No linting errors (`uv run ruff check src/`)
4. New features include tests

See [CLAUDE.md](CLAUDE.md) for development guidelines.
