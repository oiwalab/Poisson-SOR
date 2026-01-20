
const MINIMUM_GRID_SIZE = 8
const NUM_PRE_SMOOTH = 2
const NUM_POST_SMOOTH = 2

struct MultiGrid <: AbstractMethod end




function smoothing!(
    phi::Array{Float64,3},
    b::Array{Float64,3},
    epsilon::Array{Float64,3},
    electrode_mask::Array{Bool,3},
    h::Float64,
    omega::Float64,
)
    nz, nx, ny = size(phi)
    h2 = h * h

    # Note: Julia uses 1-based indexing
    # Python range(1, nz-1) becomes 2:nz-1 in Julia
    # Parallelize over k (z-direction) for independent red point updates

    # Update red points
    @threads for k in 2:nz-1
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
                # Update red point: (i + j + k) is odd
                if isodd(k + i + j)
                    # Compute stencil
                    B = (axy * (phi[k, i+1, j] + phi[k, i-1, j] +
                                phi[k, i, j+1] + phi[k, i, j-1]) +
                         az * phi[k+1, i, j] +
                         bz * phi[k-1, i, j] +
                         b[k, i, j])

                    # SOR update
                    phi[k, i, j] = (1 - omega) * phi[k, i, j] + omega * (B / A)
                end
            end
        end
    end

    # Update black points
    @threads for k in 2:nz-1
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
                # Update black point: (i + j + k) is even
                if iseven(k + i + j)
                    # Compute stencil
                    B = (axy * (phi[k, i+1, j] + phi[k, i-1, j] +
                                phi[k, i, j+1] + phi[k, i, j-1]) +
                         az * phi[k+1, i, j] +
                         bz * phi[k-1, i, j] +
                         b[k, i, j])

                    # SOR update
                    phi[k, i, j] = (1 - omega) * phi[k, i, j] + omega * (B / A)
                end
            end
        end
    end
end


function compute_residual(
    phi::Array{Float64,3},
    b::Array{Float64,3},
    epsilon::Array{Float64,3},
    electrode_mask::Array{Bool,3},
    h::Float64,
)
    nz, nx, ny = size(phi)
    h2 = h^2
    r = zeros(Float64, nz, nx, ny)

    @threads for k in 2:nz-1
        eps_k = epsilon[k, 1, 1]

        eps_zp = epsilon[k, 1, 1]
        eps_zm = epsilon[k-1, 1, 1]

        for i in 2:nx-1
            for j in 2:ny-1
                # Skip electrode points
                if electrode_mask[k, i, j]
                    continue
                end

                laplacian = (
                    eps_k * (
                        phi[k, i+1, j] + phi[k, i-1, j] +
                        phi[k, i, j+1] + phi[k, i, j-1] -
                        4 * phi[k, i, j]
                    )
                    + eps_zp * (phi[k+1, i, j] - phi[k, i, j])
                    + eps_zm * (phi[k-1, i, j] - phi[k, i, j])
                ) / h2
                r[k, i, j] = laplacian + b[k, i, j]
            end
        end
    end

    return r
end

function restrict(fine::Array{Float64,3})
    nz_f, nx_f, ny_f = size(fine)

    nz_c = div(nz_f, 2) + 1
    nx_c = div(nx_f, 2) + 1
    ny_c = div(ny_f, 2) + 1

    coarse = zeros(Float64, nz_c, nx_c, ny_c)


    @threads for k_c in 2:(nz_c-1)
        k_f = 2 * k_c - 1
        for i_c in 2:(nx_c-1)
            i_f = 2 * i_c - 1
            for j_c in 2:(ny_c-1)
                j_f = 2 * j_c - 1

                for dk in -1:1, di in -1:1, dj in -1:1
                    # Center: 1/8, Faces: 1/16, Edges: 1/32, Corners: 1/64
                    pow = 3 + abs(dk) + abs(di) + abs(dj)
                    weight = 1 / (2 ^ pow)
                    coarse[k_c, i_c, j_c] += weight * fine[k_f + dk, i_f + di, j_f + dj]
                end
            end
        end
    end

    # Ghost layers
    # Bottom/top in Z

    coarse[1, :, :] .= fine[1, 1:2:end, 1:2:end]
    coarse[end, :, :] .= fine[end, 1:2:end, 1:2:end]

    # Left/right in x
    coarse[:, 1, :] .= fine[1:2:end, 1, 1:2:end]
    coarse[:, end, :] .= fine[1:2:end, end, 1:2:end]

    # Front/back in y
    coarse[:, :, 1] .= fine[1:2:end, 1:2:end, 1]
    coarse[:, :, end] .= fine[1:2:end, 1:2:end, end]
    return coarse
end

function prolongate(coarse::Array{Float64,3}, fine_ref::Array{Float64,3})
    nz_c, nx_c, ny_c = size(coarse)
    nz_f, nx_f, ny_f = size(fine_ref)

    fine = zeros(Float64, nz_f, nx_f, ny_f)

    @threads for k_f in 2:(nz_f-1)
        for i_f in 2:(nx_f-1), j_f in 2:(ny_f-1)
            k_c, dk = divrem(k_f + 1, 2)
            i_c, di = divrem(i_f + 1, 2)
            j_c, dj = divrem(j_f + 1, 2)

            wk0 = 1.0 - 0.5 * dk
            wk1 = 0.5 * dk
            wi0 = 1.0 - 0.5 * di
            wi1 = 0.5 * di
            wj0 = 1.0 - 0.5 * dj
            wj1 = 0.5 * dj

            fine[k_f, i_f, j_f] = (
                wk0 * wi0 * wj0 * coarse[k_c, i_c, j_c] +
                wk0 * wi0 * wj1 * coarse[k_c, i_c, j_c + 1] +
                wk0 * wi1 * wj0 * coarse[k_c, i_c + 1, j_c] +
                wk0 * wi1 * wj1 * coarse[k_c, i_c + 1, j_c + 1] +
                wk1 * wi0 * wj0 * coarse[k_c + 1, i_c, j_c] +
                wk1 * wi0 * wj1 * coarse[k_c + 1, i_c, j_c + 1] +
                wk1 * wi1 * wj0 * coarse[k_c + 1, i_c + 1, j_c] +
                wk1 * wi1 * wj1 * coarse[k_c + 1, i_c + 1, j_c + 1]
            )
        end
    end
    # Ghost layers are remained zero
    return fine
end

function v_cycle!(
    phi::Array{Float64,3},
    b::Array{Float64,3},
    epsilon::Array{Float64,3},
    electrode_mask::Array{Bool,3},
    h::Float64,
    omega::Float64,
)
    for _ in 1:NUM_PRE_SMOOTH
        smoothing!(
            phi, b, epsilon, electrode_mask, h, omega
        )
    end

    r = compute_residual(
        phi, b, epsilon, electrode_mask, h
    )

    rhs = restrict(r)

    nz_c, nx_c, ny_c = size(rhs)
    err = zeros(Float64, nz_c, nx_c, ny_c)

    if min(nz_c, nx_c, ny_c) <= MINIMUM_GRID_SIZE
        # Solve directly on coarsest grid
        smoothing!(
            err, rhs, epsilon[1:2:end, 1:2:end, 1:2:end],
            electrode_mask[1:2:end, 1:2:end, 1:2:end],
            2*h, omega
        )
    else
        # Recursive call
        v_cycle!(
            err, rhs,
            epsilon[1:2:end, 1:2:end, 1:2:end],
            electrode_mask[1:2:end, 1:2:end, 1:2:end],
            2*h, omega,
        )
    end

    phi .+= prolongate(err, phi)

    for _ in 1:NUM_POST_SMOOTH
        smoothing!(
            phi, b, epsilon, electrode_mask, h, omega
        )
    end
end

function sor_iteration!(
    phi::Array{Float64,3},
    rho::Array{Float64,3},
    epsilon::Array{Float64,3},
    electrode_mask::Array{Bool,3},
    h::Float64,
    omega::Float64,
    epsilon_0::Float64,
    ::MultiGrid
)
    b = rho ./ epsilon_0
    v_cycle!(
        phi, b, epsilon, electrode_mask, h, omega
    )
end