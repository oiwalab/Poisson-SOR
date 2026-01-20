
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

    # Correct coarse grid size calculation
    nz_c = cld(nz_f, 2)
    nx_c = cld(nx_f, 2)
    ny_c = cld(ny_f, 2)

    coarse = zeros(Float64, nz_c, nx_c, ny_c)

    # Interior points with full weighting
    @threads for k_c in 2:(nz_c-1)
        k_f = 2 * k_c - 1
        if k_f < 2 || k_f > nz_f - 1
            continue
        end
        for i_c in 2:(nx_c-1)
            i_f = 2 * i_c - 1
            if i_f < 2 || i_f > nx_f - 1
                continue
            end
            for j_c in 2:(ny_c-1)
                j_f = 2 * j_c - 1
                if j_f < 2 || j_f > ny_f - 1
                    continue
                end
                # Full weighting stencil
                for dk in -1:1, di in -1:1, dj in -1:1
                    pow = 3 + abs(dk) + abs(di) + abs(dj)
                    weight = 1 / (2 ^ pow)
                    coarse[k_c, i_c, j_c] += weight * fine[k_f + dk, i_f + di, j_f + dj]
                end
            end
        end
    end

    # Boundary faces with explicit loops
    for i_c in 1:nx_c, j_c in 1:ny_c
        i_f = clamp(2 * i_c - 1, 1, nx_f)
        j_f = clamp(2 * j_c - 1, 1, ny_f)
        coarse[1, i_c, j_c] = fine[1, i_f, j_f]
        coarse[nz_c, i_c, j_c] = fine[nz_f, i_f, j_f]
    end
    for k_c in 1:nz_c, j_c in 1:ny_c
        k_f = clamp(2 * k_c - 1, 1, nz_f)
        j_f = clamp(2 * j_c - 1, 1, ny_f)
        coarse[k_c, 1, j_c] = fine[k_f, 1, j_f]
        coarse[k_c, nx_c, j_c] = fine[k_f, nx_f, j_f]
    end
    for k_c in 1:nz_c, i_c in 1:nx_c
        k_f = clamp(2 * k_c - 1, 1, nz_f)
        i_f = clamp(2 * i_c - 1, 1, nx_f)
        coarse[k_c, i_c, 1] = fine[k_f, i_f, 1]
        coarse[k_c, i_c, ny_c] = fine[k_f, i_f, ny_f]
    end

    return coarse
end

function prolongate(coarse::Array{Float64,3}, fine_size::NTuple{3,Int})
    nz_c, nx_c, ny_c = size(coarse)
    nz_f, nx_f, ny_f = fine_size

    fine = zeros(Float64, nz_f, nx_f, ny_f)

    @threads for k_f in 2:(nz_f-1)
        # Linear mapping: fine index -> coarse coordinate
        t_k = (k_f - 1) * (nz_c - 1) / (nz_f - 1)
        k_c_lo = floor(Int, t_k) + 1
        k_c_hi = min(k_c_lo + 1, nz_c)
        wk = t_k - (k_c_lo - 1)

        for i_f in 2:(nx_f-1)
            t_i = (i_f - 1) * (nx_c - 1) / (nx_f - 1)
            i_c_lo = floor(Int, t_i) + 1
            i_c_hi = min(i_c_lo + 1, nx_c)
            wi = t_i - (i_c_lo - 1)

            for j_f in 2:(ny_f-1)
                t_j = (j_f - 1) * (ny_c - 1) / (ny_f - 1)
                j_c_lo = floor(Int, t_j) + 1
                j_c_hi = min(j_c_lo + 1, ny_c)
                wj = t_j - (j_c_lo - 1)

                # Trilinear interpolation
                fine[k_f, i_f, j_f] = (
                    (1-wk) * (1-wi) * (1-wj) * coarse[k_c_lo, i_c_lo, j_c_lo] +
                    (1-wk) * (1-wi) * wj     * coarse[k_c_lo, i_c_lo, j_c_hi] +
                    (1-wk) * wi     * (1-wj) * coarse[k_c_lo, i_c_hi, j_c_lo] +
                    (1-wk) * wi     * wj     * coarse[k_c_lo, i_c_hi, j_c_hi] +
                    wk     * (1-wi) * (1-wj) * coarse[k_c_hi, i_c_lo, j_c_lo] +
                    wk     * (1-wi) * wj     * coarse[k_c_hi, i_c_lo, j_c_hi] +
                    wk     * wi     * (1-wj) * coarse[k_c_hi, i_c_hi, j_c_lo] +
                    wk     * wi     * wj     * coarse[k_c_hi, i_c_hi, j_c_hi]
                )
            end
        end
    end
    # Ghost layers remain zero
    return fine
end

function restrict_epsilon(epsilon::Array{Float64,3}, coarse_size::NTuple{3,Int})
    nz_c, _, _ = coarse_size
    nz_f = size(epsilon, 1)
    eps_c = zeros(Float64, nz_c, 1, 1)
    for k_c in 1:nz_c
        k_f = clamp(2 * k_c - 1, 1, nz_f)
        eps_c[k_c, 1, 1] = epsilon[k_f, 1, 1]
    end
    return eps_c
end

function restrict_mask(mask, coarse_size::NTuple{3,Int})
    nz_c, nx_c, ny_c = coarse_size
    nz_f, nx_f, ny_f = size(mask)
    mask_c = zeros(Bool, nz_c, nx_c, ny_c)
    for k_c in 1:nz_c, i_c in 1:nx_c, j_c in 1:ny_c
        k_f = clamp(2 * k_c - 1, 1, nz_f)
        i_f = clamp(2 * i_c - 1, 1, nx_f)
        j_f = clamp(2 * j_c - 1, 1, ny_f)
        mask_c[k_c, i_c, j_c] = mask[k_f, i_f, j_f]
    end
    return mask_c
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
        smoothing!(phi, b, epsilon, electrode_mask, h, omega)
    end

    r = compute_residual(phi, b, epsilon, electrode_mask, h)
    rhs = restrict(r)

    coarse_size = size(rhs)
    nz_c, nx_c, ny_c = coarse_size
    err = zeros(Float64, nz_c, nx_c, ny_c)

    # Use helper functions for coarsening
    epsilon_c = restrict_epsilon(epsilon, coarse_size)
    mask_c = restrict_mask(electrode_mask, coarse_size)

    if min(nz_c, nx_c, ny_c) <= MINIMUM_GRID_SIZE
        # Solve directly on coarsest grid
        smoothing!(err, rhs, epsilon_c, mask_c, 2*h, omega)
    else
        # Recursive call
        v_cycle!(err, rhs, epsilon_c, mask_c, 2*h, omega)
    end

    phi .+= prolongate(err, size(phi))

    for _ in 1:NUM_POST_SMOOTH
        smoothing!(phi, b, epsilon, electrode_mask, h, omega)
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