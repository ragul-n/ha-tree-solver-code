module HuggettTree

using LinearAlgebra
using SparseArrays
using Printf
using Statistics
using Base.Threads
using JLD2

export AggregateState, Node, SteadyState, Economy, EventTree
export crra_utility, make_asset_grid, build_economy, build_tree, print_tree, solve_tree
export save_checkpoint, load_checkpoint

struct AggregateState
    name::String
    index::Int32
    z_states::Vector{Float64}
    Pz::Matrix{Float64}
end

mutable struct Node
    node_id::Int32
    agg_state::AggregateState
    children::Vector{Int32}
    child_prob::Vector{Float64}

    r::Float64
    c_pol::Matrix{Float64}
    a_pol::Matrix{Float64}
    Phi::Matrix{Float64}
    A::Float64

    excess::Float64
end

struct SteadyState
    r::Float64
    c_pol::Matrix{Float64}
    a_pol::Matrix{Float64}
    Phi::Matrix{Float64}     # n_a x n_z stationary distribution
    A::Float64                # aggregate assets (equals p.B at the solution)
end


struct Economy{F1<:Function,F2<:Function,F3<:Function}
    agg_states::Vector{AggregateState}
    Pagg::Matrix{Float64}

    T::Int64
    uncertain_T::Int64

    borrowing_limit::Float64

    β::Float64
    γ::Float64

    relax::Float64
    B::Float64

    rmin::Float64
    rmax::Float64


    u::F1
    du::F2
    duinv::F3
    n_z::Int


    a_min::Float64
    a_max::Float64
    n_a::Int
    a_grid::Vector{Float64}
end

mutable struct EventTree
    nodes::Vector{Node}
    levels::Vector{Vector{Int32}}   # node ids by depth
    root::Int32
    economy::Economy
end


#########################################################################
# Helper functions
#########################################################################


"CRRA utility u(c) = c^(1-γ)/(1-γ), with the log case for γ==1."
function crra_utility(γ::Float64)
    if γ == 1.0
        u      = c -> log(c)
        du     = c -> 1.0 / c
        du_inv = x -> 1.0 / x
    else
        u      = c -> c^(1 - γ) / (1 - γ)
        du     = c -> c^(-γ)
        du_inv = x -> x^(-1 / γ)
    end

    return u, du, du_inv
end

"Power-spaced asset grid, concentrated near a_min when curvature > 1."
function make_asset_grid(a_min::Real, a_max::Real, n_a::Int; curvature::Float64=2.0)
    x = range(0.0, 1.0, length=n_a)
    return a_min .+ (a_max - a_min) .* (x .^ curvature)
end


function build_economy(; agg_states, Pagg,
                        T, uncertain_T,
                        β, γ,
                        relax,
                        B,
                        a_min, a_max, n_a, curvature,
                        borrowing_limit=-1.0,
                        rmin=-1.0, rmax=10.0,
                    )

n_z = length(agg_states[1].z_states)
u, du, du_inv = crra_utility( γ)
a_grid = collect(make_asset_grid(a_min, a_max, n_a, curvature=curvature))

return Economy(
    agg_states, Pagg,
    T, uncertain_T,
    borrowing_limit,
     β, γ,  relax,
     B,
     rmin, rmax,
     u, du ,  du_inv,
     n_z,
    a_min, a_max, n_a, a_grid
)

end



function build_tree( economy;
                        root_node_index,
                        stationary_node_index=-1,
                        r0=0.03
)

    certain_T = economy.T - economy.uncertain_T

    root_node = Node(
        Int32(1),
        economy.agg_states[root_node_index],
        Int32[],
        Float64[],
         r0,
        Matrix{Float64}(undef, economy.n_a, economy.n_z),
        Matrix{Float64}(undef, economy.n_a, economy.n_z),
        Matrix{Float64}(undef, economy.n_a, economy.n_z),
        0.0,
        0.0
    )

    nodes = Node[]
    levels = [Int32[] for _ in 1:(economy.T + 1)]

    push!(nodes, root_node)
    push!(levels[1], Int32(1))

    # forward build
    for d in 1:Int(economy.uncertain_T)
        for pid32 in levels[d]
            pid = Int(pid32)
            parent = nodes[pid]
            parent_state_index = parent.agg_state.index

            for child_state_index in 1:length(economy.agg_states)
                p_branch = economy.Pagg[parent_state_index, child_state_index]
                child_id = Int32(length(nodes) + 1)

                push!(nodes, Node(
                    child_id,
                    economy.agg_states[child_state_index],
                    Int32[],
                    Float64[],
                    r0,
                    zeros(Float64, economy.n_a, economy.n_z),
                    zeros(Float64, economy.n_a, economy.n_z),
                    zeros(Float64, economy.n_a, economy.n_z),
                    0.0,
                    0.0
                ))

                push!(nodes[pid].children, child_id)
                push!(nodes[pid].child_prob, p_branch)
                push!(levels[d + 1], child_id)
            end
        end
    end

    for d in Int(economy.uncertain_T+1):economy.T
        for pid32 in levels[d]
            if stationary_node_index==-1
                pid=Int(pid32)
                parent=nodes[pid]
                parent_state_index = parent.agg_state.index

                p_branch=1
                child_id=Int32(length(nodes)+1)
                push!(nodes, Node(
                    child_id,
                    economy.agg_states[parent_state_index],
                    Int32[],
                    Float64[],
                    r0,
                    zeros(Float64, economy.n_a, economy.n_z),
                    zeros(Float64, economy.n_a, economy.n_z),
                    zeros(Float64, economy.n_a, economy.n_z),
                    0.0,
                    0.0
                ))
                push!(nodes[pid].children, child_id)
                push!(nodes[pid].child_prob, p_branch)
                push!(levels[d + 1], child_id)
            else
                pid=Int(pid32)
                parent=nodes[pid]

                p_branch=1
                child_id=Int32(length(nodes)+1)
                push!(nodes, Node(
                    child_id,
                    economy.agg_states[stationary_node_index],
                    Int32[],
                    Float64[],
                    r0,
                    zeros(Float64, economy.n_a, economy.n_z),
                    zeros(Float64, economy.n_a, economy.n_z),
                    zeros(Float64, economy.n_a, economy.n_z),
                    0.0,
                    0.0
                ))
                push!(nodes[pid].children, child_id)
                push!(nodes[pid].child_prob, p_branch)
                push!(levels[d + 1], child_id)
            end
        end
    end

    steady_states = [ solve_steady_state(economy, aggregate_state) for aggregate_state in economy.agg_states]

    for node_id in levels[end]
        node=nodes[node_id]
        steady_state = steady_states[node.agg_state.index]
        node.r = steady_state.r
        node.c_pol = steady_state.c_pol
        node.a_pol = steady_state.a_pol
        node.Phi = steady_state.Phi
        node.A = steady_state.A
    end

    nodes[1].Phi= steady_states[1].Phi
    return EventTree(nodes, levels, Int32(1), economy)
end



function print_tree(tree::EventTree)
    econ = tree.economy
    println("=================================================")
    println("Event tree")
    println("Aggregate states: ", econ.agg_states)
    println("=================================================")

    for (d, level) in enumerate(tree.levels)
        println("Depth $(d - 1):")
        for nid32 in level
            node = tree.nodes[Int(nid32)]
            println("  node=$(nid32)  agg_state=$(node.agg_state.name)  prob=$(node.child_prob)")
            println("      r=$(round(node.r, digits=6))  excess=$(round(node.excess, digits=6))")
            println("      children=$(node.children)")
            println("      child_probs=$(node.child_prob)")
        end
    end
end




"Flat-extrapolating linear interpolation. `xgrid` must be sorted ascending."
function interp_linear(xgrid::AbstractVector, yvals::AbstractVector, xq::Real)
    n = length(xgrid)
    if xq <= xgrid[1]
        slope = (yvals[2] - yvals[1]) / (xgrid[2] - xgrid[1])
        return yvals[1] + slope * (xq - xgrid[1])
    elseif xq >= xgrid[n]
        slope = (yvals[n] - yvals[n-1]) / (xgrid[n] - xgrid[n-1])
        return yvals[n] + slope * (xq - xgrid[n])
    end
    i = clamp(searchsortedlast(xgrid, xq), 1, n - 1)
    x0, x1 = xgrid[i], xgrid[i+1]
    y0, y1 = yvals[i], yvals[i+1]
    return y0 + (y1 - y0) * (xq - x0) / (x1 - x0)
end


function egm_step(economy::Economy, aggregate_state::AggregateState, r_t::Float64, r_tp1::Float64,
                   w_t::AbstractVector, c_next::AbstractMatrix)
    n_a, n_z = economy.n_a, economy.n_z
    a_grid = economy.a_grid
    Π = aggregate_state.Pz

    # Expected marginal utility tomorrow, for each a' on the grid and each z today
    MU_next = economy.du.(c_next)                      # n_a x n_z
    EMU = Matrix{Float64}(undef, n_a, n_z)
    for z in 1:n_z
        @views EMU[:, z] = MU_next * Π[z, :]      # sum_z' Π(z,z') u'(c_next(a',z'))
    end

    # Euler equation -> consumption today consistent with each choice of a'
    c_endo = economy.duinv.(economy.β * (1 + r_tp1) .* EMU)   # n_a x n_z

    # Budget constraint, inverted -> endogenous grid of *current* assets
    a_endo = similar(c_endo)
    for z in 1:n_z
        @views a_endo[:, z] = (c_endo[:, z] .+ a_grid .- w_t[z]) ./ (1 + r_t)
    end

    # Map back onto the exogenous a_grid, imposing the borrowing constraint
    c_pol = Matrix{Float64}(undef, n_a, n_z)
    a_pol = Matrix{Float64}(undef, n_a, n_z)
    for z in 1:n_z
        amin_endo = a_endo[1, z]
        @views ae_z = a_endo[:, z]
        for (i, a) in enumerate(a_grid)
            if a <= amin_endo
                a_pol[i, z] = economy.a_min
                c_pol[i, z] = (1 + r_t) * a + w_t[z] - economy.a_min
            else
                aprime = clamp(interp_linear(ae_z, a_grid, a), economy.a_min, economy.a_max)
                a_pol[i, z] = aprime
                c_pol[i, z] = (1 + r_t) * a + w_t[z] - aprime
            end
        end
    end
    return c_pol, a_pol
end




function solve_policy_steady(economy::Economy,
                            aggregate_state::AggregateState,
                                r::Float64;
                              tol::Float64=1e-8, maxit::Int=2000,
                              c_init::Union{Nothing,Matrix{Float64}}=nothing)
    c_pol = c_init === nothing ?
        [max(1e-6, 0.05 * ((1 + r) * a + aggregate_state.z_states[z])) for a in economy.a_grid, z in 1:economy.n_z] :
        c_init
    a_pol = similar(c_pol)
    it = 0
    for outer it in 1:maxit
        c_new, a_new = egm_step(economy, aggregate_state, r, r, aggregate_state.z_states, c_pol)
        d = maximum(abs.(c_new .- c_pol))
        c_pol, a_pol = c_new, a_new
        d < tol && break
    end
    return c_pol, a_pol, it
end




"""
    transition_operator(p, a_pol)

Build the sparse Markov transition matrix Γ over the joint discretized
state (a,z) implied by a savings policy `a_pol` (n_a x n_z, generally
off-grid) and the exogenous income chain `p.Π`. Off-grid a' values are
split between their two nearest grid neighbours with weights that match
the first moment exactly (Young, 2010).

States are indexed as `idx(i,z) = (z-1)*n_a + i`, consistent with Julia's
column-major `reshape(vec, n_a, n_z)`.
"""
function transition_operator(economy::Economy,aggregate_state::AggregateState, a_pol::AbstractMatrix)
    n_a, n_z = economy.n_a, economy.n_z
    a_grid = economy.a_grid
    N = n_a * n_z
    idx(i, z) = (z - 1) * n_a + i

    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, n_a * n_z * n_z)
    sizehint!(J, n_a * n_z * n_z)
    sizehint!(V, n_a * n_z * n_z)

    for z in 1:n_z, i in 1:n_a
        aprime = a_pol[i, z]
        if aprime <= a_grid[1]
            j_lo, j_hi, ω = 1, 1, 1.0
        elseif aprime >= a_grid[n_a]
            j_lo, j_hi, ω = n_a, n_a, 1.0
        else
            j_lo = searchsortedlast(a_grid, aprime)
            j_hi = j_lo + 1
            ω = (a_grid[j_hi] - aprime) / (a_grid[j_hi] - a_grid[j_lo])
        end
        s = idx(i, z)
        for zp in 1:n_z
            pzp = aggregate_state.Pz[z, zp]
            pzp == 0.0 && continue
            push!(I, s); push!(J, idx(j_lo, zp)); push!(V, ω * pzp)
            if j_hi != j_lo
                push!(I, s); push!(J, idx(j_hi, zp)); push!(V, (1 - ω) * pzp)
            end
        end
    end
    return sparse(I, J, V, N, N)
end

"""
    stationary_distribution(Γ; tol=1e-12, maxit=100_000)

Power-iterate `φ <- φ' Γ` from a uniform start to find the stationary
distribution (left eigenvector of Γ for eigenvalue 1). Returns a vector of
length `size(Γ,1)`.
"""
function stationary_distribution(Γ::AbstractMatrix; tol::Float64=1e-12, maxit::Int=100_000)
    N = size(Γ, 1)
    φ = fill(1.0 / N, N)
    for _ in 1:maxit
        φ_new = vec(φ' * Γ)
        d = maximum(abs.(φ_new .- φ))
        φ = φ_new
        d < tol && break
    end
    return φ
end

"Aggregate next-period assets ∫ a'(a,z) dΦ(a,z), given policy and distribution."
function aggregate_assets(economy::Economy, a_pol::AbstractMatrix, Phi::AbstractMatrix)
    return sum(a_pol .* Phi)
end

"Aggregate next-period assets ∫ a'(a,z) dΦ(a,z), given policy and distribution."
function aggregate_assets(economy::Economy, a_pol::AbstractMatrix, φ::AbstractVector)
    Phi = reshape(φ, economy.n_a, economy.n_z)
    return sum(a_pol .* Phi)
end




"""
    solve_steady_state(p; r_lo=-0.05, r_hi=1/p.β-1-1e-4, tol=1e-6, maxit=100, verbose=true)

Bisect on the interest rate `r` until the household sector's aggregate
asset demand equals the bond supply `p.B`.
"""
function solve_steady_state(economy::Economy,
                            aggregate_state::AggregateState;
                             r_lo::Float64=-0.05,
                             r_hi::Float64=1 / economy.β - 1 - 1e-4,
                             tol::Float64=1e-6, maxit::Int=100, verbose::Bool=true)
    function excess_assets(r::Float64)
        c_pol, a_pol, _ = solve_policy_steady(economy, aggregate_state, r)
        Γ = transition_operator(economy, aggregate_state, a_pol)
        φ = stationary_distribution(Γ)
        A = aggregate_assets(economy, a_pol, φ)
        return A - economy.B, c_pol, a_pol, φ, A
    end

    flo, = excess_assets(r_lo)
    fhi, = excess_assets(r_hi)
    @assert flo < 0 && fhi > 0 "bracket [r_lo, r_hi] = [$r_lo, $r_hi] does not " *
        "bracket a root (excess assets = $flo, $fhi); widen the bracket."

    r = 0.5 * (r_lo + r_hi)
    local c_pol, a_pol, φ, A
    for it in 1:maxit
        r = 0.5 * (r_lo + r_hi)
        f, c_pol, a_pol, φ, A = excess_assets(r)
        verbose && @printf("steady state | iter %3d | r = %+.6f | excess assets = %+.6e\n", it, r, f)
        abs(f) < tol && break
        if f > 0
            r_hi = r
        else
            r_lo = r
        end
    end
    Phi = reshape(φ, economy.n_a, economy.n_z)
    return SteadyState(r, c_pol, a_pol, Phi, A)
end




function egm_step_on_tree(r::Float64, economy::Economy, node::Node, children::Vector{Node}, child_prob)
    n_a, n_z = economy.n_a, economy.n_z
    a_grid = economy.a_grid

    EMU = zeros(n_a, n_z)

    for (child, prob) in zip(children, child_prob)
        Π = child.agg_state.Pz
        MU_next = economy.du.(child.c_pol)
        for z in 1:n_z
            @views EMU[:, z] += (prob*(1+child.r))*MU_next * Π[z, :]
        end
    end

    # Euler equation -> consumption today consistent with each choice of a'
    c_endo = economy.duinv.(economy.β .* EMU)   # n_a x n_z

    # Budget constraint, inverted -> endogenous grid of *current* assets
    a_endo = similar(c_endo)
    for z in 1:n_z
        @views a_endo[:, z] = (c_endo[:, z] .+ a_grid .- node.agg_state.z_states[z]) ./ (1 + r)
    end

    # Map back onto the exogenous a_grid, imposing the borrowing constraint
    c_pol = Matrix{Float64}(undef, n_a, n_z)
    a_pol = Matrix{Float64}(undef, n_a, n_z)
    for z in 1:n_z
        x = copy(a_endo[:, z])
        y = copy(a_grid)

        perm = sortperm(x)
        x = x[perm]
        y = y[perm]

        for (i, a) in enumerate(a_grid)
            if a <= x[1]
                a_pol[i, z] = economy.a_min
                c_pol[i, z] = (1 + r) * a + node.agg_state.z_states[z] - economy.a_min
            else
                aprime = clamp(interp_linear(x, y, a), economy.a_min, economy.a_max)
                a_pol[i, z] = aprime
                c_pol[i, z] = (1 + r) * a + node.agg_state.z_states[z] - aprime
            end
        end
    end
    return c_pol, a_pol

end



function transition_given_prices(economy::Economy,tree::EventTree)
    n_a, n_z = economy.n_a, economy.n_z


     for t in (length(tree.levels)-1):-1:1
         Threads.@threads for node_id in tree.levels[t]
            node = tree.nodes[node_id]
            c_pol, a_pol = egm_step_on_tree(node.r, economy, tree.nodes[node_id], [tree.nodes[child_id] for child_id in node.children], node.child_prob )
            tree.nodes[node_id].c_pol, tree.nodes[node_id].a_pol = c_pol, a_pol
        end
    end

    # --- 2. forward simulation of the distribution, initial condition = old steady state
    maxED=0


     for t in 1:1:(length(tree.levels)-2)
        Threads.@threads for node_id in tree.levels[t]
            node=tree.nodes[node_id]
            for child_id in node.children
                child = tree.nodes[child_id]
                Γ = transition_operator(economy, child.agg_state, node.a_pol)
                φ_next = vec( vec(node.Phi)' * Γ )
                child.Phi = reshape(φ_next, n_a, n_z)
                child.A = aggregate_assets(economy, child.a_pol, child.Phi)
                child.excess = economy.B - child.A
                if abs(child.excess)>abs(maxED)
                    maxED=child.excess
                end
            end
        end
    end
    return maxED
end

function find_r_star(r, ED, economy::Economy, node::Node, children::Vector{Node}, child_prob)
    rmin, rmax = copy(economy.rmin), copy(economy.rmax)
    r_new=deepcopy(r)
    ED=node.excess
    for i in 1:30
        if ED>0
            rmax = deepcopy(r_new)
        else
            rmin= deepcopy(r_new)
        end
        r_new = (rmin+rmax)/2
        c_pol, a_pol=egm_step_on_tree(r_new, economy, node, children, child_prob)
        ED = aggregate_assets(economy, a_pol, node.Phi) - economy.B
        if abs(ED) < 1e-2
            return r_new
        end
    end
    return r_new
end



function price_update_on_tree(tree::EventTree, economy::Economy)
    Threads.@threads for node in tree.nodes
        node.r = node.r + economy.relax * node.excess
    end
end


function save_checkpoint(tree::EventTree, iter::Int, path::String)
    jldsave(path; tree, iter)
end

function load_checkpoint(path::String)
    if isfile(path)
        data = load(path)
        return (data["tree"], data["iter"])
    end
    return nothing
end

function solve_tree(economy::Economy, tree::EventTree;
                          tol::Float64=1e-6,
                          maxit::Int=1000,
                          r_bounds::Tuple{Float64,Float64}=(-0.1, 1 / economy.β - 1 - 1e-6),
                          verbose::Bool=true,
                          start_iter::Int=1,
                          checkpoint_path::Union{Nothing,String}=nothing,
                          checkpoint_every::Int=nothing,
                          checkpoint_every_secs::Union{Nothing,Real}=nothing)

    last_ckpt_time = time()

    for iter in start_iter:maxit
        maxED=transition_given_prices(economy, tree)
        verbose && @printf(" iter %4d | max|excess assets| = %.3e\n", iter, maxED)
        if abs(maxED) < tol
            print("reached tol!")
            return
        end
        price_update_on_tree(tree, economy)

        if checkpoint_path !== nothing
            by_iter = mod(iter, checkpoint_every) == 0
            by_time = checkpoint_every_secs !== nothing && (time() - last_ckpt_time) >= checkpoint_every_secs
            if by_iter || by_time
                save_checkpoint(tree, iter, checkpoint_path)
                last_ckpt_time = time()
                verbose && println("  [checkpoint saved at iteration $iter]")
            end
        end
    end
    @warn "solve_mit_shock: transition did not converge within maxit=$maxit iterations"
end

end # module
