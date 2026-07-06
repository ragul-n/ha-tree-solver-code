using LinearAlgebra
using Base.Threads

# ============================================================
# Core structs
# ============================================================

mutable struct Node
    agg_state::Int32
    children::Vector{Int32}
    Child_prob::Vector{Float64}

    prob::Float64          # unconditional probability from root to node
    r::Float64             # node interest rate
    V::Float64             # diagnostic average utility
    excess::Float64        # asset-market excess demand

    dist::Matrix{Float64}   # unconditional distribution at node
    cp::Matrix{Float64}     # consumption policy
    ap::Matrix{Float64}     # asset policy

    a_grid::Vector{Float64}
end

struct Economy
    agg_states::Vector{Int64}
    Pagg::Matrix{Float64}

    z_states::Vector{Float64}          # employed / unemployed levels
    Pz::Vector{Matrix{Float64}}        # idio transition matrix by aggregate state
    w::Vector{Float64}                 # wage by aggregate state

    uncertain_T::Int32
    T::Int32

    a_grid::Vector{Float64}
    borrowing_limit::Float64

    beta::Float64
    gamma::Float64

    relax::Float64
    B::Float64

    rmin::Float64
    rmax::Float64
end

mutable struct EventTree
    nodes::Vector{Node}
    levels::Vector{Vector{Int32}}   # node ids by depth
    root::Int32
    economy::Economy
end

# ============================================================
# Helpers
# ============================================================

utility(c, γ) = γ == 1.0 ? log.(max.(c, 1e-12)) :
    ((max.(c, 1e-12) .^ (1 - γ) .- 1) ./ (1 - γ))

marg_u(c, γ) = max.(c, 1e-12) .^ (-γ)
inv_marg_u(u, γ) = max.(u, 1e-14) .^ (-1 / γ)

function ensure_strictly_increasing!(x::Vector{Float64})
    for i in 2:length(x)
        if x[i] <= x[i - 1]
            x[i] = x[i - 1] + 1e-12
        end
    end
    return x
end

function linear_interp_sorted(xgrid::Vector{Float64},
                              ygrid::Vector{Float64},
                              xquery::Vector{Float64})
    n = length(xgrid)
    yquery = similar(xquery, Float64)

    j = 1
    for i in eachindex(xquery)
        x = xquery[i]

        if x <= xgrid[1]
            yquery[i] = ygrid[1]
        elseif x >= xgrid[end]
            yquery[i] = ygrid[end]
        else
            while j < n - 1 && xgrid[j + 1] < x
                j += 1
            end
            xL = xgrid[j]
            xH = xgrid[j + 1]
            yL = ygrid[j]
            yH = ygrid[j + 1]
            t = (x - xL) / (xH - xL)
            yquery[i] = (1 - t) * yL + t * yH
        end
    end
    return yquery
end

function coordGridIntp!(xgrid::Vector{Float64},
                        xtarget::Vector{Float64},
                        ibelow::Vector{Int32},
                        iweight::Vector{Float64};
                        robust::Bool = false)
    xg = length(xgrid)
    xi = 1
    xlow = xgrid[1]
    xhigh = xgrid[2]

    @inbounds for it in eachindex(xtarget)
        xt = xtarget[it]
        while xi < xg - 1
            if xhigh >= xt
                break
            end
            xi += 1
            xlow = xhigh
            xhigh = xgrid[xi + 1]
        end
        iweight[it] = (xhigh - xt) / (xhigh - xlow)
        ibelow[it] = Int32(xi)
    end

    if robust
        iweight .= clamp.(iweight, 0.0, 1.0)
        ibelow  .= clamp.(ibelow, Int32(1), Int32(xg - 1))
    end

    return iweight, ibelow
end

# ============================================================
# Economy constructor
# ============================================================

function make_economy(;
    agg_states::Vector{Int64},
    Pagg::Matrix{Float64},
    z_states::Vector{Float64},
    Pz::Vector{Matrix{Float64}},
    w::Vector{Float64},
    uncertain_T::Int32,
    T::Int32,
    a_grid::Vector{Float64},
    borrowing_limit::Float64,
    beta::Float64,
    gamma::Float64,
    relax::Float64 = 0.05,
    B::Float64 = 0.0,
    rmin::Float64 = -0.05,
    rmax::Float64 = 0.10
)
    return Economy(
        agg_states, Pagg, z_states, Pz, w,
        uncertain_T, T,
        a_grid, borrowing_limit,
        beta, gamma,
        relax, B,
        rmin, rmax
    )
end

# ============================================================
# Tree builder
# ============================================================

function build_tree(econ::Economy; root_state::Int32 = 1, r0::Float64 = 0.01)
    nagg = length(econ.agg_states)
    nA = length(econ.a_grid)
    nZ = length(econ.z_states)

    @assert size(econ.Pagg, 1) == nagg && size(econ.Pagg, 2) == nagg
    @assert length(econ.w) == nagg
    @assert length(econ.Pz) == nagg

    for s in 1:nagg
        @assert size(econ.Pz[s], 1) == nZ && size(econ.Pz[s], 2) == nZ
    end

    nodes = Node[]
    levels = [Int32[] for _ in 1:(econ.uncertain_T + 1)]

    # root
    push!(nodes, Node(
        root_state,
        Int32[],
        Float64[],
        1.0,
        r0,
        0.0,
        0.0,
        zeros(nA, nZ),
        zeros(nA, nZ),
        zeros(nA, nZ),
        econ.a_grid
    ))
    push!(levels[1], Int32(1))

    # forward build
    for d in 1:Int(econ.uncertain_T)
        for pid32 in levels[d]
            pid = Int(pid32)
            parent = nodes[pid]
            parent_state = Int(parent.agg_state)

            for child_state in 1:nagg
                p_branch = econ.Pagg[parent_state, child_state]
                child_id = Int32(length(nodes) + 1)

                push!(nodes, Node(
                    Int32(child_state),
                    Int32[],
                    Float64[],
                    parent.prob * p_branch,
                    r0,
                    0.0,
                    0.0,
                    zeros(nA, nZ),
                    zeros(nA, nZ),
                    zeros(nA, nZ),
                    econ.a_grid
                ))

                push!(nodes[pid].children, child_id)
                push!(nodes[pid].Child_prob, p_branch)
                push!(levels[d + 1], child_id)
            end
        end
    end

    return EventTree(nodes, levels, Int32(1), econ)
end

function print_tree(tree::EventTree)
    econ = tree.economy
    println("=================================================")
    println("Event tree")
    println("Aggregate states: ", econ.agg_states)
    println("Idiosyncratic states: ", econ.z_states)
    println("=================================================")

    for (d, level) in enumerate(tree.levels)
        println("Depth $(d - 1):")
        for nid32 in level
            node = tree.nodes[Int(nid32)]
            println("  node=$(nid32)  agg_state=$(node.agg_state)  prob=$(round(node.prob, digits=6))")
            println("      r=$(round(node.r, digits=6))  excess=$(round(node.excess, digits=6))  V=$(round(node.V, digits=6))")
            println("      children=$(node.children)")
            println("      child_probs=$(node.Child_prob)")
        end
    end
end

# ============================================================
# EGM step
# ============================================================

function egm_step_given_emu(econ::Economy, r::Float64, w::Float64, emu::Matrix{Float64})
    β = econ.beta
    γ = econ.gamma
    a_grid = econ.a_grid
    nA = length(a_grid)
    nZ = length(econ.z_states)

    a_mat = repeat(a_grid, 1, nZ)
    z_mat = repeat(reshape(econ.z_states, 1, :), nA, 1)

    c_endog = inv_marg_u(β * (1 + r) .* emu, γ)
    a_endog = (c_endog .+ a_mat .- w .* z_mat) ./ (1 + r)

    cp = zeros(nA, nZ)
    ap = zeros(nA, nZ)

    for z in 1:nZ
        xs = copy(a_endog[:, z])
        ys = copy(a_grid)
        perm = sortperm(xs)
        xs = xs[perm]
        ys = ys[perm]

        ensure_strictly_increasing!(xs)

        ap[:, z] = linear_interp_sorted(xs, ys, a_grid)

        # borrowing constraint
        for a in 1:nA
            if ap[a, z] < econ.borrowing_limit
                ap[a, z] = econ.borrowing_limit
            else
                break
            end
        end
    end

    cp .= (1 + r) .* a_mat .+ w .* z_mat .- ap
    cp = max.(cp, 1e-12)

    return cp, ap
end

# ============================================================
# Policy solve at one node
# ============================================================

function solve_node_policy!(tree::EventTree, node_id::Int32;
                            maxit::Int = 3000,
                            tol::Float64 = 1e-10)
    econ = tree.economy
    node = tree.nodes[Int(node_id)]
    s = Int(node.agg_state)
    w = econ.w[s]
    Pz = econ.Pz[s]

    nA = length(econ.a_grid)
    nZ = length(econ.z_states)

    # initial guess
    if maximum(abs.(node.cp)) == 0.0
        for a in 1:nA, z in 1:nZ
            coh = (1 + node.r) * econ.a_grid[a] + w * econ.z_states[z]
            node.cp[a, z] = 0.5 * max(coh - econ.borrowing_limit, 1e-12)
        end
    end

    cp = copy(node.cp)
    ap = copy(node.ap)

    for _ in 1:maxit
        emu = zeros(nA, nZ)

        if isempty(node.children)
            # stationary leaf
            emu .= marg_u(cp, econ.gamma) * Pz'
        else
            # continuation through child nodes
            for (child_id32, p_branch) in zip(node.children, node.Child_prob)
                child = tree.nodes[Int(child_id32)]
                s_child = Int(child.agg_state)
                emu .+= p_branch .* (marg_u(child.cp, econ.gamma) * econ.Pz[s_child]')
            end
        end

        cp_new, ap_new = egm_step_given_emu(econ, node.r, w, emu)

        d = maximum(abs.(cp_new .- cp))
        cp .= cp_new
        ap .= ap_new

        if d < tol
            break
        end
    end

    node.cp .= cp
    node.ap .= ap
    node.V = sum(utility(node.cp, econ.gamma)) / (nA * nZ)  # diagnostic only

    return nothing
end

function backward_pass!(tree::EventTree)
    for d in length(tree.levels):-1:1
        @threads for node_id32 in tree.levels[d]
            solve_node_policy!(tree, node_id32)
        end
    end
    return nothing
end

# ============================================================
# Forward distribution propagation
# ============================================================

function asset_transition!(econ::Economy, dist_cond::Matrix{Float64}, ap::Matrix{Float64})
    nA, nZ = size(dist_cond)
    nextmass = zeros(nA, nZ)

    ibelow  = fill(Int32(1), nA, nZ)
    iweight = zeros(nA, nZ)

    for z in 1:nZ
        iweight[:, z], ibelow[:, z] = coordGridIntp!(econ.a_grid, ap[:, z], ibelow[:, z], iweight[:, z], robust = true)
    end

    @inbounds for a in 1:nA, z in 1:nZ
        m = dist_cond[a, z]
        if m > 0.0
            b = Int(ibelow[a, z])
            w = iweight[a, z]
            nextmass[b, z]     += w * m
            nextmass[b + 1, z] += (1 - w) * m
        end
    end

    return nextmass
end

function forward_pass!(tree::EventTree, initial_dist::Matrix{Float64})
    econ = tree.economy

    for node in tree.nodes
        fill!(node.dist, 0.0)
    end

    # root mass
    root = Int(tree.root)
    tree.nodes[root].dist .= initial_dist ./ sum(initial_dist)

    # propagate down the tree
    for d in 1:(length(tree.levels) - 1)
        @threads for node_id32 in tree.levels[d]
            node = tree.nodes[Int(node_id32)]
            if node.prob <= 0.0
                continue
            end

            # conditional distribution at this node
            dist_cond = node.dist ./ node.prob

            # assets transition using this node policy
            mass_after_assets = asset_transition!(econ, dist_cond, node.ap)

            # IMPORTANT: idiosyncratic transition uses CHILD aggregate state
            for (child_id32, p_branch) in zip(node.children, node.Child_prob)
                child = tree.nodes[Int(child_id32)]
                child_state = Int(child.agg_state)

                child.dist .= node.prob * p_branch .* (mass_after_assets * econ.Pz[child_state]')
            end
        end
    end

    return nothing
end

# ============================================================
# Excess demand and price update
# ============================================================

function compute_excess!(tree::EventTree)
    econ = tree.economy
    a_mat = repeat(econ.a_grid, 1, length(econ.z_states))

    for node in tree.nodes
        if node.prob > 0.0
            dist_cond = node.dist ./ node.prob
            Ea = sum(dist_cond .* a_mat)
            node.excess = Ea - econ.B
        else
            node.excess = 0.0
        end
    end

    return nothing
end

function update_prices!(tree::EventTree)
    econ = tree.economy
    scale = max(abs(econ.B), 1.0)

    for node in tree.nodes
        node.r = clamp(node.r - econ.relax * node.excess / scale,
                       econ.rmin, econ.rmax)
    end

    return nothing
end

# ============================================================
# Main solver
# ============================================================

function solve_tree!(tree::EventTree, initial_dist::Matrix{Float64};
                     maxiter::Int = 100,
                     tol::Float64 = 1e-5)

    initial_dist = initial_dist ./ sum(initial_dist)

    for iter in 1:maxiter
        r_old = [node.r for node in tree.nodes]

        backward_pass!(tree)
        forward_pass!(tree, initial_dist)
        compute_excess!(tree)
        update_prices!(tree)

        r_new = [node.r for node in tree.nodes]
        dr = maximum(abs.(r_new .- r_old))
        ed = maximum(abs.([node.excess for node in tree.nodes]))

        println("iter = $iter   max|Δr| = $(dr)   max|excess| = $(ed)")

        if dr < tol && ed < tol
            println("Converged.")
            break
        end
    end

    return tree
end

# ============================================================
# Example setup (small, should run quickly)
# ============================================================

agg_states = Int64[1, 2]   # 1 = normal, 2 = recession

Pagg = [
    0.90  0.10;
    0.50  0.50
]

# idiosyncratic states: employed, unemployed
z_states = [1.0, 0.0]

Pz_normal = [
    0.95  0.05;
    0.20  0.80
]

Pz_recession = [
    0.90  0.10;
    0.35  0.65
]

Pz = [Pz_normal, Pz_recession]

w = [1.0, 0.8]

a_grid = collect(range(-1.0, 25.0, length = 100))

econ = make_economy(
    agg_states = agg_states,
    Pagg = Pagg,
    z_states = z_states,
    Pz = Pz,
    w = w,
    uncertain_T = Int32(2),   # start small; increase later
    T = Int32(30),
    a_grid = a_grid,
    borrowing_limit = -1.0,
    beta = 1.0 - 0.08/4,
    gamma = 0.6,
    relax = 0.05,
    B = 3.0,
    rmin = -0.05,
    rmax = 0.10
)

tree = build_tree(econ; root_state = Int32(1), r0 = 0.01)

initial_dist = fill(1.0 / (length(a_grid) * length(z_states)),
                    length(a_grid), length(z_states))

solve_tree!(tree, initial_dist; maxiter = 100, tol = 1e-4)

print_tree(tree)