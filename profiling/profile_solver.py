"""Profile the Poisson solver to identify performance bottlenecks

This script runs the solver with cProfile and saves results for analysis with snakeviz.
"""

import sys
import cProfile
import pstats
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from structure_manager import StructureManager
from poisson_solver import PoissonSolver


def run_solver():
    """Run the Poisson solver for profiling"""
    print("=" * 60)
    print("Profiling Poisson Solver")
    print("=" * 60)

    # Configuration file path
    config_path = Path(__file__).parent / "profile_config.yaml"

    # Load structure
    print("\n[1] Loading structure definition...")
    manager = StructureManager(str(config_path))
    print(manager.get_summary())

    # Initialize solver
    print("\n[2] Initializing solver...")
    solver = PoissonSolver(manager.params, max_iterations=1000)

    print(f"  Grid size: ({manager.nx}, {manager.ny}, {manager.nz})")
    print(f"  Omega: {solver.omega}")
    print(f"  Tolerance: {solver.tolerance:.2e}")
    print(f"  Max iterations: {solver.max_iterations}")

    # Solve Poisson equation
    print("\n[3] Solving Poisson equation...")
    phi, info = solver.solve(rho=manager.charge_density)

    # Display results
    print("\n[4] Results:")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Final φ change: {info['final_phi_change']:.2e}")
    print(f"  Potential range: [{phi.min():.4f}, {phi.max():.4f}] V")

    return phi, info


def main():
    """Main profiling function"""
    # Setup profiler
    profiler = cProfile.Profile()

    # Run with profiling
    print("Starting profiling...\n")
    profiler.enable()
    phi, info = run_solver()
    profiler.disable()

    # Save profile results with timestamp
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_file = results_dir / f"solver_profile_{timestamp}.prof"

    profiler.dump_stats(str(profile_file))
    print(f"\n[5] Profile saved to: {profile_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Top 30 functions by cumulative time:")
    print("=" * 60)
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    print("\n" + "=" * 60)
    print("Top 30 functions by total time:")
    print("=" * 60)
    stats.sort_stats("tottime")
    stats.print_stats(30)

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print(f"View results with: uv run snakeviz {profile_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
