"""
    sor_iteration!(phi, rho, epsilon, electrode_mask, h, omega, epsilon_0, ::SOR)

Perform single SOR iteration in-place.

# Arguments
- `phi::Array{Float64,3}`: Potential distribution (nz, nx, ny)
- `rho::Array{Float64,3}`: Charge density distribution (nz, nx, ny)
- `epsilon::Array{Float64,3}`: Permittivity distribution (nz, nx, ny)
- `electrode_mask::Array{Bool,3}`: Electrode mask (nz, nx, ny)
- `h::Float64`: Grid spacing
- `omega::Float64`: SOR relaxation parameter
- `epsilon_0::Float64`: Vacuum permittivity
"""
function sor_iteration!(
    phi::Array{Float64,3},
    rho::Array{Float64,3},
    epsilon::Array{Float64,3},
    electrode_mask::Array{Bool,3},
    h::Float64,
    omega::Float64,
    epsilon_0::Float64,
    ::SOR
)
    nz, nx, ny = size(phi)
    h2 = h * h

    # Note: Julia uses 1-based indexing
    # Python range(1, nz-1) becomes 2:nz-1 in Julia
    for k in 2:nz-1
        eps_k = epsilon[k, 1, 1]

        eps_zp = epsilon[k, 1, 1]
        eps_zm = epsilon[k-1, 1, 1]

        az = eps_zp / h2
        bz = eps_zm / h2
        axy = eps_k / h2
        A = 4 * axy + az + bz

        for i in 2:nx-1
            for j in 2:ny-1
                # Skip electrode points
                if electrode_mask[k, i, j]
                    continue
                end

                # Compute stencil
                B = (axy * (phi[k, i+1, j] + phi[k, i-1, j] +
                            phi[k, i, j+1] + phi[k, i, j-1]) +
                     az * phi[k+1, i, j] +
                     bz * phi[k-1, i, j] +
                     rho[k, i, j] / epsilon_0)

                # SOR update
                phi[k, i, j] = (1 - omega) * phi[k, i, j] + omega * (B / A)
            end
        end
    end

    return nothing
end