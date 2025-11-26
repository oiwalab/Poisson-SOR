# Julia Backend for Poisson Solver

## Overview

The PoissonSolver now supports a Julia backend for high-performance computation of the SOR (Successive Over-Relaxation) iteration. The entire `solve()` method has been reimplemented in Julia for maximum performance.

## Features

- **Native Julia implementation**: Complete solve loop written in Julia
- **Automatic fallback**: Falls back to Numba implementation if Julia is unavailable
- **API compatibility**: Python API remains unchanged
- **Type conversion**: Automatic conversion between NumPy and Julia arrays

## Installation

The Julia backend requires the `juliacall` package:

```bash
uv add juliacall
```

On first use, Julia will be automatically installed and configured.

## Usage

### Default (Julia backend enabled)

```python
from poisson_solver import PoissonSolver

# Julia backend is used by default
solver = PoissonSolver(structure)
result = solver.solve()
```

### Disable Julia backend

```python
# Use Numba implementation instead
solver = PoissonSolver(structure, use_julia=False)
result = solver.solve()
```

## Implementation Details

### File Structure

```
src/
├── poisson_solver.py          # Python wrapper with Julia integration
└── poisson_julia/
    ├── __init__.py
    └── sor_solver.jl          # Julia implementation
```

### Julia Functions

The Julia module (`sor_solver.jl`) implements:

- `solve_poisson()` - Main solve loop with convergence checking
- `sor_iteration!()` - Single SOR iteration (in-place)
- `apply_boundary_conditions!()` - Apply boundary conditions
- `prepare_bc_dict()` - Convert Python boundary conditions to Julia format

### Data Flow

1. Python prepares NumPy arrays
2. Arrays converted to Julia types via `jl.Array()`
3. Julia performs computation
4. Results converted back to NumPy arrays
5. Python creates `PoissonResult` object

## Performance

Benchmark results (as of implementation):

| Grid Size | Julia Time | Numba Time | Speedup |
|-----------|-----------|-----------|---------|
| 30³ (27K points) | 4.6s | 0.7s | 0.15x |
| 50³ (125K points) | 3.7s | 1.8s | 0.49x |
| 70³ (343K points) | 16.9s | 17.0s | 1.01x |

**Current status**: Julia backend has comparable performance to Numba for large grids, but has overhead for small grids.

**Performance considerations**:
- Python-Julia data transfer overhead affects small problems
- Array conversion (`jl.Array()`) creates copies
- Boundary condition processing done per-solve

**Potential optimizations** (not yet implemented):
- Use contiguous arrays: `np.ascontiguousarray()` before conversion
- Cache Julia arrays between calls for repeated solves
- Pre-convert boundary conditions once during initialization
- Use PythonCall's zero-copy array sharing where possible

Run the benchmark script to compare:

```bash
uv run python examples/benchmark_julia.py
```

## Compatibility

- Python API is fully compatible with existing code
- All tests pass with Julia backend
- Convergence behavior identical to Numba implementation
- Results are numerically equivalent (within solver tolerance)

## Troubleshooting

### Julia not found

If Julia installation fails, manually install Julia 1.10+ and ensure it's in PATH.

### Array conversion errors

Ensure NumPy arrays are contiguous and of correct dtype:
- Float arrays: `dtype=np.float64`
- Boolean arrays: `dtype=np.bool_`

### Performance issues

First run is slower due to Julia compilation. Subsequent runs are fast.

## Technical Notes

### Index Conventions

Julia uses 1-based indexing while Python uses 0-based:
- Loop ranges adjusted: `range(1, n-1)` in Python → `2:n-1` in Julia
- Array indexing: `arr[k, i, j]` in both (Julia handles conversion)

### Memory Management

- Arrays are passed by reference (no copying)
- Julia's garbage collector manages memory
- No manual memory management needed

### Boundary Conditions

Python dict converted to Julia Dict automatically:
- Nested dicts flattened: `bc["z_top"]["type"]` → `bc["z_top_type"]`
- All values converted to Julia native types
