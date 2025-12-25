"""Normal SOR iteration JIT function

Separated for easier maintenance and testing
"""

import numpy as np
from numba import njit


@njit
def _sor_iteration_jit(
    phi: np.ndarray,
    rho: np.ndarray,
    epsilon: np.ndarray,
    electrode_mask: np.ndarray,
    h: float,
    omega: float,
    epsilon_0: float,
    nz: int,
    nx: int,
    ny: int,
) -> np.ndarray:
    """JIT-compiled SOR iteration core computation

    Parameters
    ----------
    phi : np.ndarray
        Potential distribution (nz, nx, ny)
    rho : np.ndarray
        Charge density distribution (nz, nx, ny)
    epsilon : np.ndarray
        Permittivity distribution (nz, nx, ny)
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
    """
    h2 = h * h

    for k in range(1, nz - 1):
        eps_k = epsilon[k, 0, 0]

        eps_zp = epsilon[k, 0, 0]
        eps_zm = epsilon[k - 1, 0, 0]

        az = eps_zp / h2
        bz = eps_zm / h2
        axy = eps_k / h2

        A = 4 * axy + az + bz

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
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
