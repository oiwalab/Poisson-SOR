# Performance Profiling

This directory contains tools for profiling the Poisson solver to identify performance bottlenecks.

## Quick Start

```bash
# Install profiling dependencies
uv add --dev snakeviz

# Run profiling
uv run python profiling/profile_solver.py

# View results in browser
uv run snakeviz profiling/results/solver_profile.prof
```

## Files

- `profile_solver.py` - Main profiling script
- `profile_config.yaml` - Simplified configuration for profiling
- `results/` - Output directory for profiling results

## Profiling Methods

### 1. Full Profiling with snakeviz (Recommended)

```bash
uv run python profiling/profile_solver.py
uv run snakeviz profiling/results/solver_profile.prof
```

This will:
1. Run the solver with profiling enabled
2. Save profile data to `results/solver_profile.prof`
3. Print summary statistics to console
4. Open an interactive visualization in your browser

### 2. Command-line Profiling

For quick profiling without visualization:

```bash
uv run python -m cProfile -s cumulative profiling/profile_solver.py | head -50
```

### 3. Line-by-line Profiling

To profile specific functions line-by-line:

```bash
# Install line_profiler
uv add --dev line-profiler

# Add @profile decorator to functions you want to profile
# Then run:
uv run kernprof -l -v profiling/profile_solver.py
```

## Understanding Results

### snakeviz View

The snakeviz browser interface shows:
- **Icicle chart**: Hierarchical view of function calls
- **Sunburst chart**: Circular hierarchical view
- **Function list**: Detailed statistics for each function

Click on functions to drill down into their call hierarchy.

### Key Metrics

- **tottime**: Total time spent in the function (excluding subcalls)
- **cumtime**: Total time spent in the function (including subcalls)
- **ncalls**: Number of times the function was called
- **percall**: Average time per call

## Configuration

`profile_config.yaml` uses a smaller grid (50x50x50 nm) compared to the example configuration for faster profiling. Adjust the grid size and structure as needed:

```yaml
domain:
  size: [50e-9, 50e-9, 50e-9]  # Adjust for different profiling scenarios
  grid_spacing: 1e-9
```

## Common Bottlenecks to Check

1. SOR iteration loop in `PoissonSolver.solve()`
2. Boundary condition application
3. Array indexing and memory access patterns
4. NumPy operations in the solver
5. Convergence checking

## Tips

- Start with small grid sizes to quickly identify bottlenecks
- Use `tottime` to find functions that are slow themselves
- Use `cumtime` to find expensive call chains
- Profile multiple times to account for variability
- Compare profiles before and after optimizations
