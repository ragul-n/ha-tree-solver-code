using Base.Threads
using CSV
using JLD2
using DataFrames

println("Using ", Threads.nthreads(), " threads")

include("HuggettTree.jl")
using .HuggettTree

aggregate_states = [
    AggregateState(
        "Normal",
        1,
        [0.5, 0.05],
        [0.9 0.1; 0.5 0.5]
    ),
    AggregateState(
        "Recession",
        2,
        [0.3, 0.05],
        [0.6 0.4; 0.2 0.8]
    )
]

Pagg = [0.8 0.2;
        0.5 0.5]

economy = build_economy(
    agg_states=aggregate_states, Pagg=Pagg,
    T=40, uncertain_T=4,
    β=0.96, γ=0.9,
    borrowing_limit=0.0,
    relax=0.01,
    B=2.0,
    a_min=0.0, a_max=5.0, n_a=300, curvature=2.0,
    rmin=-0.05, rmax=0.045
)

checkpoint_path = joinpath(pwd(), "checkpoint.jld2")
start_iter = 1

ckpt = load_checkpoint(checkpoint_path)
if ckpt !== nothing
    tree, start_iter = ckpt
    start_iter += 1  # resume from the next iteration
    println("Resuming from checkpoint at iteration $start_iter")
else
    tree = build_tree(economy, root_node_index=1, stationary_node_index=-1)
    println("No. of nodes: ", length(tree.nodes))
end

solve_tree(economy, tree, maxit=1000,
           start_iter=start_iter,
           checkpoint_path=checkpoint_path,
           # checkpoint_every=100
        )

# --- Save results ---
outdir = mktempdir(pwd(), cleanup=false)
csv_path = joinpath(outdir, "results.csv")
jld2_path = joinpath(outdir, "tree.jld2")

# CSV summary
df = DataFrame(
    node_id   = [n.node_id for n in tree.nodes],
    agg_state = [n.agg_state.name for n in tree.nodes],
    r         = [n.r for n in tree.nodes],
    A         = [n.A for n in tree.nodes],
    excess    = [n.excess for n in tree.nodes],
)
CSV.write(csv_path, df)
println("CSV written to $csv_path")

# JLD2 checkpoint
jldsave(jld2_path; tree, economy)
println("JLD2 written to $jld2_path")

# Upload to S3 if OUTPUT_S3_PATH is set
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
