#!/usr/bin/env julia
#
# Example: Solve a Huggett model on a finite aggregate event tree.
#
# Run with:
#   julia run_tree_example.jl
#
# Assumes HuggettTree.jl is in the same directory.

include("HuggettTree.jl")
using .HuggettTree
using Printf

# -------------------------------------------------------------------
# 1. Idiosyncratic income process (employed / unemployed)
# -------------------------------------------------------------------
# These are the idiosyncratic states used inside each node.
# They are NOT aggregate states.
#
# Here:
#   z = 1  -> employed
#   z = 2  -> unemployed
#
# p.w below stores these income levels for the household problem.
# The tree-specific aggregate wages are passed separately.

n_z = 2
z_grid = [1.0, 2.0]   # informational labels only; not used directly
w_idio = [1.0, 0.2]   # idiosyncratic income levels

Π_normal = [
    0.95  0.05;
    0.25  0.75
]

Π_recession = [
    0.90  0.10;
    0.35  0.65
]

# -------------------------------------------------------------------
# 2. Build model parameters
# -------------------------------------------------------------------
# HuggettTree.jl expects a standard incompletes-markets household block.
# The wage vector here is the idiosyncratic income vector.
#
# Note: aggregate wages are supplied separately to the event tree.

p = build_params(
    β = 1.0 - 0.08/4,
    γ = 2.0,
    z_states = w_idio,
    # Π = Π_normal,   # used by the base steady-state / helper routines
    a_min = -1.0,
    a_max = 25.0,
    n_a = 120,
    B = 2.0,
    curvature = 2.0
)

# -------------------------------------------------------------------
# 3. Aggregate states and aggregate transition matrix
# -------------------------------------------------------------------
# 1 = normal, 2 = recession
agg_states = Int32[1, 2]

Πagg = [
    0.90  0.10;
    0.50  0.50
]

# Aggregate wages by state (normal, recession)
wages_by_state = [1.00, 0.80]

# Idiosyncratic transition matrix by aggregate state
Πz_by_state = [Π_normal, Π_recession]

# -------------------------------------------------------------------
# 4. Build the tree
# -------------------------------------------------------------------
# uncertain_T = number of aggregate shock dates.
# If uncertain_T = 2, you get a 3-level tree:
#   root -> 2 children -> 4 terminal nodes

tree = build_tree(
    p,
    Πagg,
    agg_states;
    uncertain_T = Int32(2),
    wages_by_state = wages_by_state,
    Πz_by_state = Πz_by_state,
    root_state = Int32(1),
    r0 = 0.01
)

println("Tree built.")
print_tree(tree)

# -------------------------------------------------------------------
# 5. Initial distribution
# -------------------------------------------------------------------
# Uniform starting distribution over (a,z).

initial_dist = fill(1.0 / (p.n_a * p.n_z), p.n_a, p.n_z)

# -------------------------------------------------------------------
# 6. Solve the tree equilibrium
# -------------------------------------------------------------------
# For a first test, keep the horizon small.
# Increase uncertain_T later once the code is behaving well.

println("\nSolving tree equilibrium...\n")
solve_tree_equilibrium(
    tree,
    initial_dist;
    maxiter = 30,
    tol = 1e-4,
    maxit_leaf = 500,
    tol_node = 1e-8,
    verbose = true
)

# -------------------------------------------------------------------
# 7. Print a summary
# -------------------------------------------------------------------
println("\nDone.\n")

for node in tree.nodes
    @printf("node=%2d  state=%d  prob=%6.3f  r=% .5f  excess=% .5f\n",
            node.id, node.agg_state, node.prob, node.r, node.excess)
end
