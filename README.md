# SOR Poisson Solver

3D Poisson equation solver using the Successive Over-Relaxation (SOR) method for semiconductor device simulations.

## Overview

This solver solves the Poisson equation $-\nabla \cdot (\varepsilon \nabla \phi) = \rho$ in three dimensions for systems with non-uniform permittivity distributions. Designed for semiconductor heterostructure simulations with multiple material layers and electrode configurations.

## Features

- Non-uniform permittivity distribution
- Multi-layer heterostructure support
- Electrode placement with voltage control
- Configurable boundary conditions (Dirichlet, Neumann, periodic)
- Result visualization (potential distribution, convergence history)
- YAML-based configuration

## Requirements

- Python 3.13+
- numpy
- matplotlib
- pyyaml
- pytest

## Installation

Install dependencies using uv:

```bash
uv sync
```

For development dependencies:

```bash
uv sync -U
```

## Usage

### Configuration File

Define your structure in a YAML file (see [configs/example.yaml](configs/example.yaml)):

```yaml
domain:
  size: [100e-9, 100e-9, 50e-9]  # [x, y, z] in meters
  grid_spacing: 5e-9              # Isotropic grid spacing

layers:
  - material: "SiO2"
    z_range: [0, -10e-9]          # z=0 (surface) to z=-10nm
    epsilon_r: 3.9

  - material: "Si"
    z_range: [-10e-9, -50e-9]
    epsilon_r: 11.7

electrodes:
  - name: "gate_1"
    shape: "rectangle"
    x_range: [10e-9, 30e-9]
    y_range: [0, 100e-9]
    z_position: -5e-9             # Electrode bottom position
    voltage: -0.5

solver:
  omega: 1.5                      # SOR relaxation parameter
  max_iterations: 10000
  tolerance: 1e-6

boundary_conditions:
  z_top:
    type: "neumann"
    value: 0.0                    # ∂φ/∂z = 0
  z_bottom:
    type: "neumann"
    value: 0.0
  x_sides:
    type: "neumann"
    value: 0.0
  y_sides:
    type: "neumann"
    value: 0.0
```

### Example Code

```python
from structure_manager import StructureManager
from poisson_solver import PoissonSolver
import visualizer as vis

# Load structure from YAML
manager = StructureManager("configs/example.yaml")

# Initialize solver
solver = PoissonSolver(
    epsilon=manager.epsilon_array,
    grid_spacing=manager.h,
    boundary_conditions=manager.config['boundary_conditions'],
    omega=1.5,
    tolerance=1e-6,
    max_iterations=10000,
    electrode_mask=manager.electrode_mask,
    electrode_voltages=manager.electrode_voltages,
)

# Solve Poisson equation
phi, info = solver.solve()

# Visualize results
x, y, z = manager.get_grid_coordinates()
vis.plot_potential_slice(phi, x, y, z)
vis.plot_convergence(solver.residual_history)
```

### Running Example

```bash
uv run python examples/example.py
```

Results are saved to the `results/` directory.

## Project Structure

```
.
├── src/
│   ├── poisson_solver.py      # SOR Poisson solver
│   ├── structure_manager.py   # Structure and electrode management
│   └── visualizer.py           # Visualization functions
├── configs/
│   └── example.yaml            # Example configuration
├── examples/
│   └── example.py              # Example script
├── tests/
│   └── test_solver.py          # Unit tests
└── README.md
```

## Testing

Run tests using pytest:

```bash
uv run pytest
```

For coverage report:

```bash
uv run pytest --cov=src --cov-report=html
```

## Development

This project uses:
- **ruff** for linting and formatting
- **uv** for package management
- **pytest** for testing

Linting and formatting:

```bash
uv run ruff check src/
uv run ruff format src/
```

## Coordinate System

The solver uses a coordinate system where:
- z = 0: Surface (electrode side)
- z-axis: Extends in negative direction
- Array indexing: (nz, nx, ny) where k=0 is the surface

## License

(Add license information if applicable)
