"""CuPy-based GPU solver for Poisson equation using Red-Black SOR method"""

import cupy as cp

# CUDA kernel for Red-Black SOR iteration
_sor_kernel_code = r"""
extern "C" __global__
void sor_update_phase(
    double* phi,
    const double* rho,
    const double* epsilon,
    const bool* electrode_mask,
    const int nz, const int nx, const int ny,
    const double h2,
    const double omega,
    const double epsilon_0,
    const int phase
) {
    // Grid indices (1-indexed for interior points)
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // Bounds check (interior points only)
    if (k >= nz - 1 || i >= nx - 1 || j >= ny - 1) return;

    // Red-Black check: skip if not matching phase
    // Red: (k + i + j) % 2 == 0, Black: (k + i + j) % 2 == 1
    if ((k + i + j) % 2 != phase) return;

    // Linear index for 3D array (row-major: k * nx * ny + i * ny + j)
    int idx = k * nx * ny + i * ny + j;

    // Skip electrode points
    if (electrode_mask[idx]) return;

    // Permittivity coefficients (z-layer dependent)
    // epsilon array is (nz, nx, ny), we use epsilon[k, 0, 0] for layer k
    double eps_k = epsilon[k * nx * ny];
    double eps_zp = epsilon[k * nx * ny];        // Between k and k+1
    double eps_zm = epsilon[(k - 1) * nx * ny];  // Between k-1 and k

    // Finite difference coefficients
    double az = eps_zp / h2;
    double bz = eps_zm / h2;
    double axy = eps_k / h2;
    double A = 4.0 * axy + az + bz;

    // 6-point stencil: neighbors in x, y, z directions
    int idx_xp = idx + ny;       // phi[k, i+1, j]
    int idx_xm = idx - ny;       // phi[k, i-1, j]
    int idx_yp = idx + 1;        // phi[k, i, j+1]
    int idx_ym = idx - 1;        // phi[k, i, j-1]
    int idx_zp = idx + nx * ny;  // phi[k+1, i, j]
    int idx_zm = idx - nx * ny;  // phi[k-1, i, j]

    double B = axy * (phi[idx_xp] + phi[idx_xm] + phi[idx_yp] + phi[idx_ym])
             + az * phi[idx_zp] + bz * phi[idx_zm]
             + rho[idx] / epsilon_0;

    // SOR update
    phi[idx] = (1.0 - omega) * phi[idx] + omega * (B / A);
}
"""

# Compile the kernel
_sor_kernel = cp.RawKernel(_sor_kernel_code, "sor_update_phase")


def redblack_sor_iteration_gpu(
    phi: cp.ndarray,
    rho: cp.ndarray,
    epsilon: cp.ndarray,
    electrode_mask: cp.ndarray,
    h: float,
    omega: float,
    epsilon_0: float,
) -> None:
    """Perform one Red-Black SOR iteration on GPU (in-place).

    Parameters
    ----------
    phi : cp.ndarray
        Potential array (nz, nx, ny), modified in-place
    rho : cp.ndarray
        Charge density array (nz, nx, ny)
    epsilon : cp.ndarray
        Permittivity array (nz, nx, ny)
    electrode_mask : cp.ndarray
        Boolean mask for electrode points (nz, nx, ny)
    h : float
        Grid spacing
    omega : float
        SOR relaxation parameter (1 < omega < 2)
    epsilon_0 : float
        Vacuum permittivity
    """
    nz, nx, ny = phi.shape
    h2 = h * h

    # Thread block and grid dimensions
    # Use 8x8x8 threads per block (512 threads total)
    block = (8, 8, 8)
    grid = (
        (ny - 2 + block[0] - 1) // block[0],
        (nx - 2 + block[1] - 1) // block[1],
        (nz - 2 + block[2] - 1) // block[2],
    )

    # Red phase (phase = 0)
    _sor_kernel(
        grid,
        block,
        (phi, rho, epsilon, electrode_mask, nz, nx, ny, h2, omega, epsilon_0, 0),
    )

    # Black phase (phase = 1)
    _sor_kernel(
        grid,
        block,
        (phi, rho, epsilon, electrode_mask, nz, nx, ny, h2, omega, epsilon_0, 1),
    )


def apply_boundary_conditions_gpu(
    phi: cp.ndarray,
    boundary_conditions: dict,
) -> None:
    """Apply boundary conditions on GPU (in-place).

    Parameters
    ----------
    phi : cp.ndarray
        Potential array (nz, nx, ny), modified in-place
    boundary_conditions : dict
        Dictionary with boundary condition specifications
    """
    nz, nx, ny = phi.shape

    # Z-direction boundaries
    z_top = boundary_conditions.get("z_top", {})
    z_bottom = boundary_conditions.get("z_bottom", {})

    if z_top.get("type") == "dirichlet":
        phi[0, :, :] = z_top["value"]
    elif z_top.get("type") == "neumann":
        phi[0, :, :] = phi[1, :, :] + z_top.get("value", 0.0)

    if z_bottom.get("type") == "dirichlet":
        phi[-1, :, :] = z_bottom["value"]
    elif z_bottom.get("type") == "neumann":
        phi[-1, :, :] = phi[-2, :, :] + z_bottom.get("value", 0.0)

    # X-direction boundaries
    x_sides = boundary_conditions.get("x_sides", {})

    if x_sides.get("type") == "dirichlet":
        value = x_sides["value"]
        phi[:, 0, :] = value
        phi[:, -1, :] = value
    elif x_sides.get("type") == "neumann":
        phi[:, 0, :] = phi[:, 1, :]
        phi[:, -1, :] = phi[:, -2, :]
    elif x_sides.get("type") == "periodic":
        phi[:, 0, :] = phi[:, -2, :]
        phi[:, -1, :] = phi[:, 1, :]

    # Y-direction boundaries
    y_sides = boundary_conditions.get("y_sides", {})

    if y_sides.get("type") == "dirichlet":
        value = y_sides["value"]
        phi[:, :, 0] = value
        phi[:, :, -1] = value
    elif y_sides.get("type") == "neumann":
        phi[:, :, 0] = phi[:, :, 1]
        phi[:, :, -1] = phi[:, :, -2]
    elif y_sides.get("type") == "periodic":
        phi[:, :, 0] = phi[:, :, -2]
        phi[:, :, -1] = phi[:, :, 1]
