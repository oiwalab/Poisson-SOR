"""Red-Black SOR iteration JIT function

Separated for easier maintenance and testing
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _redblack_sor_iteration_jit(
    phi: np.ndarray,
    rho: np.ndarray,
    epsilon: np.ndarray,
    eps_z_array: np.ndarray,
    electrode_mask: np.ndarray,
    h: float,
    omega: float,
    epsilon_0: float,
    nz: int,
    nx: int,
    ny: int,
) -> np.ndarray:
    """JIT-compiled Red-Black SOR iteration with parallel execution

    Divides grid points into Red and Black groups for parallel updates.
    Red points: (k + i + j) % 2 == 0
    Black points: (k + i + j) % 2 == 1

    Parameters
    ----------
    phi : np.ndarray
        Potential distribution (nz, nx, ny)
    rho : np.ndarray
        Charge density distribution (nz, nx, ny)
    epsilon : np.ndarray
        Permittivity distribution (nz, nx, ny)
    eps_z_array : np.ndarray
        Harmonic mean at z-interfaces, shape (nz-1,)
        eps_z_array[k] is harmonic mean between layer k and k+1
    electrode_mask : np.ndarray
        Electrode mask (nz, nx, ny), True where electrodes exist
    h : float
        Grid spacing
    omega : float
        SOR relaxation parameter
    epsilon_0 : float
        Vacuum permittivity
    nz, nx, ny : int
        Grid dimensions

    Returns
    -------
    phi : np.ndarray
        Updated potential distribution

    Notes
    -----
    Uses Red-Black ordering for parallel execution:
    1. Update all Red points in parallel (neighbors are all Black)
    2. Update all Black points in parallel (neighbors are all Red)
    This ensures no data race conditions during parallel updates.
    """
    h2 = h * h

    # Update Red points (color = 0)
    for k in prange(1, nz - 1):
        eps_k = epsilon[k, 0, 0]
        eps_zp = eps_z_array[k]
        eps_zm = eps_z_array[k - 1]

        az = eps_zp / h2
        bz = eps_zm / h2
        axy = eps_k / h2
        A = 4 * axy + az + bz

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if (k + i + j) % 2 == 0:  # Red points
                    if electrode_mask[k, i, j]:
                        continue

                    B = (
                        axy
                        * (
                            phi[k, i + 1, j]
                            + phi[k, i - 1, j]
                            + phi[k, i, j + 1]
                            + phi[k, i, j - 1]
                        )
                        + az * phi[k + 1, i, j]
                        + bz * phi[k - 1, i, j]
                        + rho[k, i, j] / epsilon_0
                    )

                    phi[k, i, j] = (1 - omega) * phi[k, i, j] + omega * (B / A)

    # Update Black points (color = 1)
    for k in prange(1, nz - 1):
        eps_k = epsilon[k, 0, 0]
        eps_zp = eps_z_array[k]
        eps_zm = eps_z_array[k - 1]

        az = eps_zp / h2
        bz = eps_zm / h2
        axy = eps_k / h2
        A = 4 * axy + az + bz

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if (k + i + j) % 2 == 1:  # Black points
                    if electrode_mask[k, i, j]:
                        continue

                    B = (
                        axy
                        * (
                            phi[k, i + 1, j]
                            + phi[k, i - 1, j]
                            + phi[k, i, j + 1]
                            + phi[k, i, j - 1]
                        )
                        + az * phi[k + 1, i, j]
                        + bz * phi[k - 1, i, j]
                        + rho[k, i, j] / epsilon_0
                    )

                    phi[k, i, j] = (1 - omega) * phi[k, i, j] + omega * (B / A)

    return phi
