"""Benchmark convergence performance of SOR solver

Tests convergence speed and iteration counts
"""

import numpy as np


def test_convergence_time(benchmark, solver):
    """Measure total time to convergence"""
    rho = np.zeros(solver.epsilon.shape)

    phi, info = benchmark(solver.solve, rho=rho, verbose=False)

    assert info["converged"], "Solver should converge"
    assert info["iterations"] > 0


def test_convergence_iterations(solver):
    """Measure number of iterations to convergence"""
    rho = np.zeros(solver.epsilon.shape)

    phi, info = solver.solve(rho=rho, verbose=False)

    assert info["converged"], "Solver should converge"
    print(f"Grid size: {solver.nx}x{solver.ny}x{solver.nz}")
    print(f"Iterations: {info['iterations']}")
    print(f"Final residual: {info['final_residual']:.6e}")
