"""
CUDA.jl GPU Backend for Poisson Solver

This module provides GPU-accelerated Red-Black SOR iteration using CUDA.jl.
Requires NVIDIA GPU with CUDA support.

Note: CUDA is imported in base_solver.jl before including this file.
      GPUMethod abstract type is also defined in base_solver.jl.
"""

# Concrete GPU method type
struct CUDARedBlack <: GPUMethod end


"""
    _sor_kernel!(phi, rho, epsilon, electrode_mask, nz, nx, ny, h2, omega, epsilon_0, phase)

CUDA kernel for Red-Black SOR iteration.

# Arguments
- `phi`: Potential array (CuDeviceArray)
- `rho`: Charge density array (CuDeviceArray)
- `epsilon`: Permittivity array (CuDeviceArray)
- `electrode_mask`: Electrode mask (CuDeviceArray)
- `nz, nx, ny`: Grid dimensions
- `h2`: Grid spacing squared
- `omega`: SOR relaxation parameter
- `epsilon_0`: Vacuum permittivity
- `phase`: 0 for red points, 1 for black points
"""
function _sor_kernel!(
    phi, rho, epsilon, electrode_mask,
    nz, nx, ny, h2, omega, epsilon_0, phase
)
    # Thread indices (1-based in Julia)
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + 1
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y + 1
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + 1

    # Bounds check (interior points: 2 to n-1)
    if k < 2 || k > nz - 1 || i < 2 || i > nx - 1 || j < 2 || j > ny - 1
        return nothing
    end

    # Red-Black check
    # Red: (k + i + j) odd, Black: (k + i + j) even
    # phase=0 for red (odd), phase=1 for black (even)
    parity = (k + i + j) % 2
    if phase == 0 && parity != 1  # Red phase, want odd
        return nothing
    elseif phase == 1 && parity != 0  # Black phase, want even
        return nothing
    end

    # Skip electrode points
    if electrode_mask[k, i, j]
        return nothing
    end

    # Permittivity coefficients (z-layer dependent)
    eps_k = epsilon[k, 1, 1]
    eps_zp = epsilon[k, 1, 1]      # Between k and k+1
    eps_zm = epsilon[k-1, 1, 1]    # Between k-1 and k

    # Finite difference coefficients
    az = eps_zp / h2
    bz = eps_zm / h2
    axy = eps_k / h2
    A = 4.0 * axy + az + bz

    # 6-point stencil
    B = axy * (phi[k, i+1, j] + phi[k, i-1, j] +
               phi[k, i, j+1] + phi[k, i, j-1]) +
        az * phi[k+1, i, j] + bz * phi[k-1, i, j] +
        rho[k, i, j] / epsilon_0

    # SOR update
    phi[k, i, j] = (1.0 - omega) * phi[k, i, j] + omega * (B / A)

    return nothing
end


"""
    sor_iteration!(phi, rho, epsilon, electrode_mask, h, omega, epsilon_0, ::CUDARedBlack)

Perform one Red-Black SOR iteration on GPU using CUDA.jl.

# Arguments
- `phi::CuArray{Float64,3}`: Potential array (modified in-place)
- `rho::CuArray{Float64,3}`: Charge density array
- `epsilon::CuArray{Float64,3}`: Permittivity array
- `electrode_mask::CuArray{Bool,3}`: Electrode mask
- `h::Float64`: Grid spacing
- `omega::Float64`: SOR relaxation parameter
- `epsilon_0::Float64`: Vacuum permittivity
- `::CUDARedBlack`: Method dispatch tag
"""
function sor_iteration!(
    phi::CuArray{Float64,3},
    rho::CuArray{Float64,3},
    epsilon::CuArray{Float64,3},
    electrode_mask::CuArray{Bool,3},
    h::Float64,
    omega::Float64,
    epsilon_0::Float64,
    ::CUDARedBlack
)
    nz, nx, ny = size(phi)
    h2 = h * h

    # Thread block dimensions (8x8x8 = 512 threads)
    threads = (8, 8, 8)
    blocks = (
        cld(ny - 2, threads[1]),
        cld(nx - 2, threads[2]),
        cld(nz - 2, threads[3])
    )

    # Red phase (phase = 0, updates odd parity points)
    @cuda threads=threads blocks=blocks _sor_kernel!(
        phi, rho, epsilon, electrode_mask,
        nz, nx, ny, h2, omega, epsilon_0, 0
    )

    # Black phase (phase = 1, updates even parity points)
    @cuda threads=threads blocks=blocks _sor_kernel!(
        phi, rho, epsilon, electrode_mask,
        nz, nx, ny, h2, omega, epsilon_0, 1
    )

    return nothing
end


"""
    apply_boundary_conditions_gpu!(phi, bc_dict, h)

Apply boundary conditions on GPU arrays.

# Arguments
- `phi::CuArray{Float64,3}`: Potential array (modified in-place)
- `bc_dict::Dict`: Boundary conditions dictionary
- `h::Float64`: Grid spacing
"""
function apply_boundary_conditions_gpu!(
    phi::CuArray{Float64,3},
    bc_dict::Dict,
    h::Float64
)
    nz, nx, ny = size(phi)

    # Z-direction: z_top (k=1)
    if haskey(bc_dict, "z_top_type")
        bc_type = bc_dict["z_top_type"]
        bc_value = get(bc_dict, "z_top_value", 0.0)

        if bc_type == "neumann"
            phi[1, :, :] .= phi[2, :, :] .- bc_value * h
        elseif bc_type == "dirichlet"
            phi[1, :, :] .= bc_value
        end
    end

    # Z-direction: z_bottom (k=nz)
    if haskey(bc_dict, "z_bottom_type")
        bc_type = bc_dict["z_bottom_type"]
        bc_value = get(bc_dict, "z_bottom_value", 0.0)

        if bc_type == "neumann"
            phi[nz, :, :] .= phi[nz-1, :, :] .+ bc_value * h
        elseif bc_type == "dirichlet"
            phi[nz, :, :] .= bc_value
        end
    end

    # X-direction boundaries
    if haskey(bc_dict, "x_sides_type")
        bc_type = bc_dict["x_sides_type"]
        bc_value = get(bc_dict, "x_sides_value", 0.0)

        if bc_type == "neumann"
            phi[:, 1, :] .= phi[:, 2, :] .- bc_value * h
            phi[:, nx, :] .= phi[:, nx-1, :] .+ bc_value * h
        elseif bc_type == "dirichlet"
            phi[:, 1, :] .= bc_value
            phi[:, nx, :] .= bc_value
        elseif bc_type == "periodic"
            phi[:, 1, :] .= phi[:, nx-1, :]
            phi[:, nx, :] .= phi[:, 2, :]
        end
    end

    # Y-direction boundaries
    if haskey(bc_dict, "y_sides_type")
        bc_type = bc_dict["y_sides_type"]
        bc_value = get(bc_dict, "y_sides_value", 0.0)

        if bc_type == "neumann"
            phi[:, :, 1] .= phi[:, :, 2] .- bc_value * h
            phi[:, :, ny] .= phi[:, :, ny-1] .+ bc_value * h
        elseif bc_type == "dirichlet"
            phi[:, :, 1] .= bc_value
            phi[:, :, ny] .= bc_value
        elseif bc_type == "periodic"
            phi[:, :, 1] .= phi[:, :, ny-1]
            phi[:, :, ny] .= phi[:, :, 2]
        end
    end

    return nothing
end


"""
    solve_poisson(phi, rho, epsilon, electrode_mask, electrode_voltages,
                  boundary_conditions, h, omega, epsilon_0, tolerance,
                  max_iterations, method::GPUMethod, verbose)

Solve Poisson equation using GPU-accelerated SOR method.
This is a multiple dispatch specialization for GPUMethod subtypes.

# Arguments
- `phi::Array{Float64,3}`: Initial potential distribution
- `rho::Array{Float64,3}`: Charge density distribution (C/m^3)
- `epsilon::Array{Float64,3}`: Permittivity distribution
- `electrode_mask::Array{Bool,3}`: Electrode mask
- `electrode_voltages::Array{Float64,3}`: Electrode voltages
- `boundary_conditions`: Boundary conditions (Python dict or Julia Dict)
- `h::Float64`: Grid spacing
- `omega::Float64`: SOR relaxation parameter (1 < omega < 2)
- `epsilon_0::Float64`: Vacuum permittivity (F/m)
- `tolerance::Float64`: Convergence threshold
- `max_iterations::Int`: Maximum iterations
- `method::GPUMethod`: GPU method dispatch tag (e.g., CUDARedBlack())
- `verbose::Bool`: Print progress

# Returns
- `phi::Array{Float64,3}`: Converged potential distribution (CPU array)
- `info::Dict`: Convergence information
"""
function solve_poisson(
    phi::Array{Float64,3},
    rho::Array{Float64,3},
    epsilon::Array{Float64,3},
    electrode_mask::Array{Bool,3},
    electrode_voltages::Array{Float64,3},
    boundary_conditions,
    h::Float64,
    omega::Float64,
    epsilon_0::Float64,
    tolerance::Float64,
    max_iterations::Int,
    method::GPUMethod,
    verbose::Bool
)
    # Transfer arrays to GPU
    phi_gpu = CuArray(phi)
    rho_gpu = CuArray(rho)
    epsilon_gpu = CuArray(epsilon)
    electrode_mask_gpu = CuArray(electrode_mask)
    electrode_voltages_gpu = CuArray(electrode_voltages)

    # Prepare boundary conditions
    bc_julia = prepare_bc_dict(boundary_conditions)

    # Set electrode potentials
    phi_gpu[electrode_mask_gpu] .= electrode_voltages_gpu[electrode_mask_gpu]

    # Apply initial boundary conditions
    apply_boundary_conditions_gpu!(phi_gpu, bc_julia, h)

    # Convergence tracking
    convergence_history = Float64[]
    phi_prev = similar(phi_gpu)
    copyto!(phi_prev, phi_gpu)

    converged = false
    final_iteration = max_iterations

    # Main SOR iteration loop
    for iteration in 1:max_iterations
        # Perform GPU SOR iteration (dispatches to specific method)
        sor_iteration!(phi_gpu, rho_gpu, epsilon_gpu, electrode_mask_gpu,
                       h, omega, epsilon_0, method)

        # Reapply boundary conditions
        apply_boundary_conditions_gpu!(phi_gpu, bc_julia, h)

        # Reapply electrode voltages
        phi_gpu[electrode_mask_gpu] .= electrode_voltages_gpu[electrode_mask_gpu]

        # Compute convergence metric (GPU reduction)
        phi_diff = maximum(abs.(phi_gpu .- phi_prev))
        push!(convergence_history, Float64(phi_diff))

        # Verbose output
        if verbose
            if iteration % 1000 == 0
                println("="^40)
            end
            if iteration % 100 == 0
                println("Iteration $iteration: Max Δφ = $(phi_diff)")
            end
        end

        # Check convergence
        if phi_diff < tolerance
            converged = true
            final_iteration = iteration
            break
        end

        # Check for divergence
        if isnan(phi_diff) || isinf(phi_diff)
            error("Phi change became NaN or Inf, diverging solution.")
        end

        # Update phi_prev for next iteration
        copyto!(phi_prev, phi_gpu)
    end

    # Transfer result back to CPU
    phi_result = Array(phi_gpu)

    info = Dict(
        "converged" => converged,
        "iterations" => final_iteration,
        "final_phi_change" => length(convergence_history) > 0 ? convergence_history[end] : 0.0,
        "convergence_history" => convergence_history
    )

    return phi_result, info
end


"""
    cuda_available()

Check if CUDA is available and functional.

# Returns
- `Bool`: true if CUDA is available, false otherwise
"""
function cuda_available()
    return CUDA.functional()
end


"""
    get_cuda_device_info()

Get information about the current CUDA device.

# Returns
- `Dict`: Device information including name, memory, compute capability
"""
function get_cuda_device_info()
    if !cuda_available()
        return Dict("available" => false)
    end

    dev = CUDA.device()
    return Dict(
        "available" => true,
        "name" => CUDA.name(dev),
        "total_memory" => CUDA.totalmem(dev),
        "compute_capability" => CUDA.capability(dev)
    )
end
