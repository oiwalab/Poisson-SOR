

abstract type AbstractMethod end
struct RedBlack <: AbstractMethod end
struct SOR <: AbstractMethod end

include("rbsor.jl")
include("sor.jl")

# GPU method abstract type (defined here so cuda_solver.jl can use it)
abstract type GPUMethod <: AbstractMethod end

# Conditionally include CUDA backend if CUDA.jl is available
const CUDA_AVAILABLE = try
    using CUDA
    CUDA.functional()
catch
    false
end

if CUDA_AVAILABLE
    include("rbsor_cuda.jl")
end

"""
    prepare_bc_dict(bc_python)

Convert Python boundary conditions dict to Julia-friendly format.

# Arguments
- `bc_python`: Python boundary conditions dictionary (can be PyDict or Dict)

# Returns
- `bc_julia::Dict`: Flattened dictionary for Julia
"""
function prepare_bc_dict(bc_python)
    bc_julia = Dict{String,Any}()

    # Convert to Dict if it's a Python object
    # Handle both PyDict and Julia Dict
    try
        # Process each boundary
        for (key, val) in bc_python
            # Convert key to string
            key_str = string(key)

            # Check if val is dict-like
            try
                bc_type = get(val, "type", nothing)
                bc_value = get(val, "value", 0.0)

                if bc_type !== nothing
                    bc_julia["$(key_str)_type"] = string(bc_type)
                    bc_julia["$(key_str)_value"] = Float64(bc_value)
                end
            catch
                # val is not dict-like, skip
            end
        end
    catch e
        @warn "Error processing boundary conditions" exception = e
    end

    return bc_julia
end


"""
    apply_boundary_conditions!(phi, bc_dict, h)

Apply boundary conditions to potential array.

# Arguments
- `phi::Array{Float64,3}`: Potential distribution (nz, nx, ny)
- `bc_dict::Dict`: Boundary conditions dictionary
- `h::Float64`: Grid spacing
"""
function apply_boundary_conditions!(
    phi::Array{Float64,3},
    bc_dict::Dict,
    h::Float64
)
    nz, nx, ny = size(phi)

    # Z-direction boundaries
    # z_top: k=1 (Julia), k=0 (Python) - surface
    if haskey(bc_dict, "z_top_type")
        bc_type = bc_dict["z_top_type"]
        bc_value = get(bc_dict, "z_top_value", 0.0)

        if bc_type == "neumann"
            # d_phi/d_z = bc_value
            phi[1, :, :] .= phi[2, :, :] .- bc_value * h
        elseif bc_type == "dirichlet"
            phi[1, :, :] .= bc_value
        end
    end

    # z_bottom: k=nz (Julia), k=nz-1 (Python) - bottom
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
                  max_iterations, method, verbose)

Main Poisson solver using SOR method.

# Arguments
- `phi::Array{Float64,3}`: Initial potential distribution (modified in-place)
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
- `method::AbstractMethod`: Solver method (SOR() or RedBlack())
- `verbose::Bool`: Print progress

# Returns
- `phi::Array{Float64,3}`: Converged potential distribution
- `info::Dict`: Convergence information
"""
function solve_poisson(
    phi::Array{Float64,3},
    rho::Array{Float64,3},
    epsilon::Array{Float64,3},
    electrode_mask::Array{Bool,3},
    electrode_voltages::Array{Float64,3},
    boundary_conditions,  # Accept any type (PyDict or Dict)
    h::Float64,
    omega::Float64,
    epsilon_0::Float64,
    tolerance::Float64,
    max_iterations::Int,
    method::AbstractMethod,
    verbose::Bool
)
    println("Using CPU backend with Julia for Poisson solver.")
    # Prepare boundary conditions dict
    bc_julia = prepare_bc_dict(boundary_conditions)

    # Set electrode potentials
    phi[electrode_mask] .= electrode_voltages[electrode_mask]

    # Apply initial boundary conditions
    apply_boundary_conditions!(phi, bc_julia, h)

    # Convergence tracking
    convergence_history = Float64[]
    phi_prev = copy(phi)

    # Main SOR iteration loop
    for iteration in 1:max_iterations
        # Perform SOR iteration
        sor_iteration!(phi, rho, epsilon, electrode_mask,
            h, omega, epsilon_0, method)

        # Reapply boundary conditions
        apply_boundary_conditions!(phi, bc_julia, h)

        # Reapply electrode voltages
        phi[electrode_mask] .= electrode_voltages[electrode_mask]

        # Compute convergence metric
        phi_diff = maximum(abs.(phi .- phi_prev))
        push!(convergence_history, phi_diff)

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
            info = Dict(
                "converged" => true,
                "iterations" => iteration,
                "final_phi_change" => phi_diff,
                "convergence_history" => convergence_history
            )
            return phi, info
        end

        # Check for divergence
        if isnan(phi_diff) || isinf(phi_diff)
            error("Phi change became NaN or Inf, diverging solution.")
        end

        # Update phi_prev for next iteration
        phi_prev .= phi
    end

    # Maximum iterations reached
    info = Dict(
        "converged" => false,
        "iterations" => max_iterations,
        "final_phi_change" => convergence_history[end],
        "convergence_history" => convergence_history
    )

    return phi, info
end


"""
    get_num_threads()

Return the number of threads Julia is using for parallel execution.
"""
function get_num_threads()
    return Threads.nthreads()
end