#!/usr/bin/env julia
# ChenSignatures.jl adapter for signature benchmarks

using ChenSignatures
using StaticArrays
using JSON
using Statistics

# -------- Path Generators --------

"""Generate linear path of static vectors: [t, 2t, 2t, ...]"""
function make_path_linear_svector(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    [SVector{d,Float32}(ntuple(i -> Float32(i == 1 ? t : 2t), d)) for t in ts]
end

"""Generate linear path as dense matrix (N x d)"""
function make_path_linear_matrix(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    path = Array{Float32}(undef, N, d)
    path[:, 1] .= ts
    if d > 1
        @views path[:, 2:end] .= 2.0 .* ts
    end
    return path
end

"""Generate sinusoidal path of static vectors: [sin(2pi*1*t), sin(2pi*2*t), ...]"""
function make_path_sin_svector(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    omega = 2.0 * pi
    [SVector{d,Float32}(ntuple(i -> Float32(sin(omega * i * t)), d)) for t in ts]
end

"""Generate sinusoidal path as dense matrix: [sin(2pi*1*t), sin(2pi*2*t), ...]"""
function make_path_sin_matrix(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    omega = 2.0 * pi
    path = Array{Float32}(undef, N, d)
    for j in 1:d
        @views path[:, j] .= sin.(omega * j .* ts)
    end
    return path
end

"""Generate path of specified kind as Vector of SVectors"""
function make_path_svector(d::Int, N::Int, kind::String)
    kind_lower = lowercase(kind)
    if kind_lower == "linear"
        return make_path_linear_svector(d, N)
    elseif kind_lower == "sin"
        return make_path_sin_svector(d, N)
    else
        error("Unknown path_kind: $kind")
    end
end

"""Generate path of specified kind as dense matrix"""
function make_path_matrix(d::Int, N::Int, kind::String)
    kind_lower = lowercase(kind)
    if kind_lower == "linear"
        return make_path_linear_matrix(d, N)
    elseif kind_lower == "sin"
        return make_path_sin_matrix(d, N)
    else
        error("Unknown path_kind: $kind")
    end
end

# -------- Manual Timing Loop --------

"""
Execute manual timing loop with warmup and GC disabled.

This is the Julia equivalent of the Python BenchmarkAdapter.manual_timing_loop.

Returns a tuple of (avg_time_ms, avg_alloc_bytes):
    - avg_time_ms: Average time per iteration in milliseconds
    - avg_alloc_bytes: Average bytes allocated per iteration
"""
function manual_timing_loop(func::Function, repeats::Int; warmup_iterations::Int=3)
    # Warmup phase (untimed)
    for _ in 1:warmup_iterations
        func()
    end

    # Timed phase with GC disabled and allocation tracking
    GC.enable(false)
    local total_alloc_bytes
    samples_ms = Float64[]
    try
        total_alloc_bytes = 0
        for _ in 1:repeats
            t0 = time_ns()
            total_alloc_bytes += @allocated begin
                func()
            end
            push!(samples_ms, (time_ns() - t0) / 1e6)
        end
    finally
        GC.enable(true)
    end

    # Calculate average bytes allocated per iteration
    avg_alloc_bytes = div(total_alloc_bytes, repeats)

    return median(samples_ms), avg_alloc_bytes, samples_ms
end

# -------- Benchmark Operations --------

"""
Prepare signature computation kernel.

Returns a closure that performs only the kernel (no setup).
"""
function run_signature(path, m::Int)
    # Setup phase (untimed): path is already prepared
    tensor_type = ChenSignatures.Tensor{Float32}

    # Return kernel closure
    return () -> signature_path(tensor_type, path, m)
end

"""
Prepare logsignature computation kernel.

Returns a closure that performs only the kernel (no setup).
"""
function run_logsignature(path, m::Int)
    # Setup phase (untimed): path is already prepared
    tensor_type = ChenSignatures.Tensor{Float32}

    # Return kernel closure
    return () -> ChenSignatures.log(signature_path(tensor_type, path, m))
end

"""Select operation and prepare kernel closure"""
function prepare_kernel(path, operation::String, m::Int)
    if operation == "signature"
        return run_signature(path, m), "signature_path"
    elseif operation == "logsignature"
        return run_logsignature(path, m), "log"
    else
        # Operation not supported (sigdiff not available yet)
        error("Unsupported operation: $operation")
    end
end

# -------- Main Benchmark Runner --------

"""Format benchmark result for JSON output"""
function output_result(
    N,
    d,
    m,
    path_kind,
    operation,
    t_ms,
    alloc_bytes,
    samples_ms,
    library,
    method,
    path_type,
)
    return Dict(
        "N" => N,
        "d" => d,
        "m" => m,
        "path_kind" => path_kind,
        "operation" => operation,
        "language" => "julia",
        "library" => library,
        "method" => method,
        "path_type" => path_type,
        "t_ms" => t_ms,
        "t_ms_mean" => mean(samples_ms),
        "t_ms_std" => std(samples_ms; corrected=false),
        "samples_ms" => samples_ms,
        "alloc_bytes" => alloc_bytes
    )
end

"""Run the benchmark and output result as JSON"""
function run_benchmark(config::Dict)
    # Extract configuration
    N = config["N"]
    d = config["d"]
    m = config["m"]
    path_kind = config["path_kind"]
    operation = config["operation"]
    repeats = config["repeats"]

    path_variants = [
        ("Vector{SVector{Float32}}", make_path_svector(d, N, path_kind)),
        ("Matrix{Float32}", make_path_matrix(d, N, path_kind))
    ]

    for (path_label, path_data) in path_variants
        # Prepare kernel for this operation
        kernel, method = prepare_kernel(path_data, operation, m)

        # Run manual timing loop
        t_ms, alloc_bytes, samples_ms = manual_timing_loop(kernel, repeats)

        # Format result
        result = output_result(
            N, d, m, path_kind, operation, t_ms, alloc_bytes, samples_ms,
            "ChenSignatures.jl",
            method,
            path_label
        )

        # Output as JSON
        println(JSON.json(result))
    end
end

# -------- Entry Point --------

if isempty(ARGS)
    println(stderr, "Usage: run_chen.jl '<json_config>'")
    exit(1)
end

# Parse configuration from command line
config_json = JSON.parse(ARGS[1])
# Convert JSON.Object to Dict
config = Dict{String, Any}(config_json)

# Run benchmark
try
    run_benchmark(config)
catch e
    error_result = Dict(
        "error" => string(e),
        "N" => get(config, "N", nothing),
        "d" => get(config, "d", nothing),
        "m" => get(config, "m", nothing),
        "operation" => get(config, "operation", nothing)
    )
    println(stderr, JSON.json(error_result))
    exit(1)
end
