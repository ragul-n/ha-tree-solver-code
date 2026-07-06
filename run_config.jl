using Base.Threads
using CSV
using JLD2
using DataFrames
using TOML

println("Using ", Threads.nthreads(), " threads")

include("HuggettTree.jl")
using .HuggettTree

config_input = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "CONFIG_PATH", "config.toml")

if startswith(config_input, "s3://")
    config_path = joinpath(tempdir(), "config.toml")
    println("Downloading config from $config_input")
    run(`aws s3 cp $config_input $config_path`)
else
    config_path = config_input
end

println("Loading config from $config_path")
cfg = TOML.parsefile(config_path)

# --- Parse aggregate states ---
agg_cfgs = cfg["aggregate_states"]
aggregate_states = [
    AggregateState(
        ac["name"],
        Int32(i),
        Float64.(ac["z_states"]),
        hvcat(length(ac["Pz"]), [Float64.(row) for row in ac["Pz"]]...)'  |> Matrix{Float64}
    )
    for (i, ac) in enumerate(agg_cfgs)
]

Pagg_raw = cfg["aggregate_transition"]["Pagg"]
Pagg = hvcat(length(Pagg_raw), [Float64.(row) for row in Pagg_raw]...)' |> Matrix{Float64}

# --- Build economy ---
ec = cfg["economy"]
economy = build_economy(
    agg_states=aggregate_states, Pagg=Pagg,
    T=ec["T"], uncertain_T=ec["uncertain_T"],
    β=ec["beta"], γ=ec["gamma"],
    borrowing_limit=get(ec, "borrowing_limit", 0.0),
    relax=ec["relax"],
    B=ec["B"],
    a_min=ec["a_min"], a_max=ec["a_max"],
    n_a=ec["n_a"], curvature=ec["curvature"],
    rmin=get(ec, "rmin", -1.0), rmax=get(ec, "rmax", 10.0),
)

# --- Solver settings ---
sv = cfg["solver"]
maxit = get(sv, "maxit", 1000)
tol = get(sv, "tol", 1e-6)
root_node_index = get(sv, "root_node_index", 1)
stationary_node_index = get(sv, "stationary_node_index", -1)
checkpoint_every = get(sv, "checkpoint_every", nothing)
checkpoint_every_secs = get(sv, "checkpoint_every_secs", nothing)

# --- Checkpoint resume ---
checkpoint_path = joinpath(pwd(), "checkpoint.jld2")
start_iter = 1

ckpt = load_checkpoint(checkpoint_path)
if ckpt !== nothing
    tree, start_iter = ckpt
    start_iter += 1
    println("Resuming from checkpoint at iteration $start_iter")
else
    tree = build_tree(economy, root_node_index=root_node_index, stationary_node_index=stationary_node_index)
    println("No. of nodes: ", length(tree.nodes))
end

# --- Solve ---
solve_kwargs = Dict{Symbol,Any}(
    :maxit => maxit,
    :tol => tol,
    :start_iter => start_iter,
    :checkpoint_path => checkpoint_path,
)
if checkpoint_every !== nothing
    solve_kwargs[:checkpoint_every] = checkpoint_every
end
if checkpoint_every_secs !== nothing
    solve_kwargs[:checkpoint_every_secs] = checkpoint_every_secs
end

solve_tree(economy, tree; solve_kwargs...)

# --- Save results ---
outdir = mktempdir(pwd(), cleanup=false)
csv_path = joinpath(outdir, "results.csv")
jld2_path = joinpath(outdir, "tree.jld2")

df = DataFrame(
    node_id   = [n.node_id for n in tree.nodes],
    agg_state = [n.agg_state.name for n in tree.nodes],
    r         = [n.r for n in tree.nodes],
    A         = [n.A for n in tree.nodes],
    excess    = [n.excess for n in tree.nodes],
)
CSV.write(csv_path, df)
println("CSV written to $csv_path")

jldsave(jld2_path; tree, economy)
println("JLD2 written to $jld2_path")

if haskey(ENV, "OUTPUT_S3_PATH")
    s3path = ENV["OUTPUT_S3_PATH"]
    try
        run(`aws s3 cp $csv_path $(s3path)results.csv`)
        run(`aws s3 cp $jld2_path $(s3path)tree.jld2`)
        println("Results uploaded to $s3path")
    catch e
        @warn "S3 upload failed" exception=e
    end
else
    println("OUTPUT_S3_PATH not set — skipping S3 upload")
end
