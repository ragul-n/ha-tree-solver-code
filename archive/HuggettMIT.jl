module HuggettMIT

using LinearAlgebra
using SparseArrays
using Printf

export ModelParams, SteadyState, TransitionPath
export crra_utility, make_asset_grid, build_params
export egm_step, solve_policy_steady, transition_operator, stationary_distribution
export aggregate_assets, solve_steady_state
export transition_given_prices, solve_mit_shock

# =========================================================================
#  1. MODEL PRIMITIVES
# =========================================================================
#
# Idiosyncratic income is an arbitrary discrete Markov chain: n_z states,
# a wage/income level in each state, and an n_z x n_z transition matrix.
# Everything else (grids, preferences) is passed in as data/functions so
# the code does not hard-code any particular calibration.

"""
    ModelParams

Immutable container for every primitive of the Huggett economy.

Fields
- β        : discount factor
- γ        : CRRA coefficient (kept only as a convenience label; the
             actual utility functions used by the solver are `u`,`du`,`duinv`)
- u        : utility function,        u(c)
- du       : marginal utility,        u'(c)
- duinv    : inverse marginal utility, (u')^{-1}(x)
- n_z      : number of idiosyncratic states
- z_grid   : labels for the states (informational only)
- w        : steady-state income/wage in each state, length n_z
- Π        : n_z x n_z Markov transition matrix, Π[z,z'] = P(z'|z), rows sum to 1
- a_min    : borrowing constraint (a' ≥ a_min always)
- a_max    : upper bound of the asset grid
- n_a      : number of asset grid points
- a_grid   : the asset grid itself (ascending, length n_a)
- B        : net supply of the traded asset (bond market clears to this)
"""
struct ModelParams{F1<:Function,F2<:Function,F3<:Function}
    β::Float64
    γ::Float64
    u::F1
    du::F2
    duinv::F3
    n_z::Int
    z_grid::Vector{Float64}
    w::Vector{Float64}
    Π::Matrix{Float64}
    a_min::Float64
    a_max::Float64
    n_a::Int
    a_grid::Vector{Float64}
    B::Float64
end

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

"""
    build_params(; β, γ=2.0, w, Π, a_min, a_max, n_a=200, B=0.0,
                   z_grid=1:n_z, curvature=2.0, u=nothing, du=nothing, duinv=nothing)

Convenience constructor for `ModelParams`. Supply `w` (income by state) and
`Π` (transition matrix) to fully describe the idiosyncratic income process;
everything else has sensible defaults. Pass your own `u`,`du`,`duinv` to use
a non-CRRA utility function.
"""
function build_params(; β::Float64, γ::Float64=2.0,
                        w::Vector{Float64}, Π::Matrix{Float64},
                        a_min::Float64, a_max::Float64, n_a::Int=200,
                        B::Float64=0.0,
                        z_grid=collect(1:length(w)),
                        curvature::Float64=2.0,
                        u=nothing, du=nothing, duinv=nothing)
    n_z = length(w)
    @assert size(Π) == (n_z, n_z) "Π must be n_z x n_z"
    @assert all(isapprox.(sum(Π, dims=2), 1.0, atol=1e-8)) "rows of Π must sum to 1"
    @assert length(z_grid) == n_z

    if u === nothing
        u, du, duinv = crra_utility(γ)
    end
    a_grid = collect(make_asset_grid(a_min, a_max, n_a; curvature=curvature))
    return ModelParams(β, γ, u, du, duinv, n_z, Float64.(collect(z_grid)), w, Π,
                        a_min, a_max, n_a, a_grid, B)
end

# =========================================================================
#  2. HOUSEHOLD PROBLEM: ENDOGENOUS GRID METHOD (EGM)
# =========================================================================

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

"""
    egm_step(p, r_t, r_tp1, w_t, c_next)

One backward step of the endogenous grid method.

- `r_t`    : interest rate that applies to this period's budget constraint
             c_t + a' = (1+r_t) a + w_t(z)
- `r_tp1`  : interest rate that applies to the *return on savings a'*
             (i.e. the rate between t and t+1); used in the Euler equation
- `w_t`    : income vector (length n_z) in the current period
- `c_next` : n_a x n_z matrix, next period's consumption policy evaluated
             ON THE EXOGENOUS GRID `p.a_grid`, i.e. c_next[i,z'] = c_{t+1}(a_grid[i], z')

Returns `(c_pol, a_pol)`, each n_a x n_z, this period's policy on `p.a_grid`.

This single function is reused both to iterate the stationary household
problem to convergence (r_t == r_tp1 == r, w_t == p.w every period) and to
do the finite-horizon backward induction along a perfect-foresight
transition path (r_t, r_tp1, w_t all time-varying).
"""
function egm_step(p::ModelParams, r_t::Float64, r_tp1::Float64,
                   w_t::AbstractVector, c_next::AbstractMatrix)
    n_a, n_z = p.n_a, p.n_z
    a_grid = p.a_grid
    Π = p.Π

    # Expected marginal utility tomorrow, for each a' on the grid and each z today
    MU_next = p.du.(c_next)                      # n_a x n_z
    EMU = Matrix{Float64}(undef, n_a, n_z)
    for z in 1:n_z
        @views EMU[:, z] = MU_next * Π[z, :]      # sum_z' Π(z,z') u'(c_next(a',z'))
    end

    # Euler equation -> consumption today consistent with each choice of a'
    c_endo = p.duinv.(p.β * (1 + r_tp1) .* EMU)   # n_a x n_z

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
                a_pol[i, z] = p.a_min
                    c_pol[i, z] = (1 + r_t) * a + w_t[z] - p.a_min
            else
                aprime = clamp(interp_linear(ae_z, a_grid, a), p.a_min, p.a_max)
                a_pol[i, z] = aprime
                c_pol[i, z] = (1 + r_t) * a + w_t[z] - aprime
            end
        end
    end
    return c_pol, a_pol
end

"""
    solve_policy_steady(p, r; tol=1e-8, maxit=2000, c_init=nothing)

Iterate `egm_step` to convergence at a *constant* interest rate `r` and
constant income `p.w`, returning the stationary household policy.
"""
function solve_policy_steady(p::ModelParams, r::Float64;
                              tol::Float64=1e-8, maxit::Int=2000,
                              c_init::Union{Nothing,Matrix{Float64}}=nothing)
    c_pol = c_init === nothing ?
        [max(1e-6, 0.05 * ((1 + r) * a + p.w[z])) for a in p.a_grid, z in 1:p.n_z] :
        c_init
    a_pol = similar(c_pol)
    it = 0
    for outer it in 1:maxit
        c_new, a_new = egm_step(p, r, r, p.w, c_pol)
        d = maximum(abs.(c_new .- c_pol))
        c_pol, a_pol = c_new, a_new
        d < tol && break
    end
    return c_pol, a_pol, it
end

# =========================================================================
#  3. DISTRIBUTION OVER (a, z): NON-STOCHASTIC "LOTTERY" METHOD
# =========================================================================

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
function transition_operator(p::ModelParams, a_pol::AbstractMatrix)
    n_a, n_z = p.n_a, p.n_z
    a_grid = p.a_grid
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
            pzp = p.Π[z, zp]
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
function aggregate_assets(p::ModelParams, a_pol::AbstractMatrix, φ::AbstractVector)
    Phi = reshape(φ, p.n_a, p.n_z)
    return sum(a_pol .* Phi)
end

# =========================================================================
#  4. STEADY STATE
# =========================================================================

"Bundle describing a stationary equilibrium."
struct SteadyState
    r::Float64
    c_pol::Matrix{Float64}
    a_pol::Matrix{Float64}
    Phi::Matrix{Float64}     # n_a x n_z stationary distribution
    A::Float64                # aggregate assets (equals p.B at the solution)
end

"""
    solve_steady_state(p; r_lo=-0.05, r_hi=1/p.β-1-1e-4, tol=1e-6, maxit=100, verbose=true)

Bisect on the interest rate `r` until the household sector's aggregate
asset demand equals the bond supply `p.B`.
"""
function solve_steady_state(p::ModelParams;
                             r_lo::Float64=-0.05,
                             r_hi::Float64=1 / p.β - 1 - 1e-4,
                             tol::Float64=1e-6, maxit::Int=100, verbose::Bool=true)
    function excess_assets(r::Float64)
        c_pol, a_pol, _ = solve_policy_steady(p, r)
        Γ = transition_operator(p, a_pol)
        φ = stationary_distribution(Γ)
        A = aggregate_assets(p, a_pol, φ)
        return A - p.B, c_pol, a_pol, φ, A
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
    Phi = reshape(φ, p.n_a, p.n_z)
    return SteadyState(r, c_pol, a_pol, Phi, A)
end

# =========================================================================
#  5. MIT SHOCK / PERFECT-FORESIGHT TRANSITION PATH
# =========================================================================

"""
Bundle describing a perfect-foresight transition path of length T+1
(t = 1 is the impact period, t = T+1 is the (assumed converged) new
steady state).
"""
struct TransitionPath
    T::Int
    r_path::Vector{Float64}
    w_path::Matrix{Float64}
    c_path::Array{Float64,3}
    a_path::Array{Float64,3}
    Phi_path::Array{Float64,3}
    A_path::Vector{Float64}
end

"""
    transition_given_prices(p, ss0, ssT, w_path, r)

The core mapping of the MIT-shock algorithm: given a *candidate* interest
rate path `r` (length T+1, with `r[end]` fixed at the terminal steady
state rate) and an income path `w_path` (n_z x (T+1)), do

  1. backward induction of household policies from the terminal steady
     state `ssT` down to the impact period, then
  2. forward simulation of the distribution starting from the initial
     (pre-shock) steady state `ss0.Phi`.

Returns a NamedTuple with the resulting `c_path`, `a_path`, `Phi_path`,
`A_path`. This function has no fixed-point logic in it, so it can be
plugged into any external root-finder (e.g. NLsolve) as well as the
built-in dampened iteration in `solve_mit_shock`.
"""
function transition_given_prices(p::ModelParams, ss0::SteadyState, ssT::SteadyState,
                                  w_path::AbstractMatrix, r::AbstractVector)
    n_a, n_z = p.n_a, p.n_z
    Tp1 = length(r)
    T = Tp1 - 1
    @assert size(w_path) == (n_z, Tp1)

    c_path = Array{Float64,3}(undef, n_a, n_z, Tp1)
    a_path = Array{Float64,3}(undef, n_a, n_z, Tp1)
    Phi_path = Array{Float64,3}(undef, n_a, n_z, Tp1)
    A_path = Vector{Float64}(undef, Tp1)

    # --- 1. backward induction of policies, terminal condition = new steady state
    c_path[:, :, Tp1] = ssT.c_pol
    a_path[:, :, Tp1] = ssT.a_pol
    for t in T:-1:1
        c_t, a_t = egm_step(p, r[t], r[t+1], view(w_path, :, t), view(c_path, :, :, t + 1))
        c_path[:, :, t] = c_t
        a_path[:, :, t] = a_t
    end

    # --- 2. forward simulation of the distribution, initial condition = old steady state
    Phi_path[:, :, 1] = ss0.Phi
    for t in 1:T
        Γ = transition_operator(p, view(a_path, :, :, t))
        φ_next = vec(vec(view(Phi_path, :, :, t))' * Γ)
        Phi_path[:, :, t+1] = reshape(φ_next, n_a, n_z)
        A_path[t] = aggregate_assets(p, view(a_path, :, :, t), vec(view(Phi_path, :, :, t)))
    end
    A_path[Tp1] = ssT.A

    return (c_path=c_path, a_path=a_path, Phi_path=Phi_path, A_path=A_path)
end

"Default price-update rule: a dampened Gauss-Seidel step on excess asset demand."
default_price_update(r::AbstractVector, ED::AbstractVector, damp::Float64) = r .- damp .* ED

"""
    solve_mit_shock(p, ss0, ssT, w_path; kwargs...) -> TransitionPath

Solve for the perfect-foresight path of interest rates `{r_t}_{t=0}^{T}`
that clears the bond market (asset demand = `B_path[t]`) in every period
of the transition following an unanticipated ("MIT") shock, given:

- `ss0` : initial steady state (the economy is assumed to sit here at the
          moment the shock hits — its distribution `ss0.Phi` is the fixed
          initial condition of the transition)
- `ssT` : terminal steady state (computed under the *long-run* parameters,
          i.e. `w_path[:, end]`; equals `ss0` if the shock is purely
          transitory)
- `w_path` : n_z x (T+1) matrix, the (possibly time-varying) income
             process during the transition; column 1 is the impact period,
             column T+1 should equal the income vector used to compute `ssT`

Keyword arguments
- `B_path`       : bond-supply path (defaults to constant `p.B`)
- `r_guess`      : initial guess for the rate path (defaults to a straight
                   line between `ss0.r` and `ssT.r`)
- `damp`         : damping parameter for the default price update
- `price_update` : `(r, excess_demand, damp) -> r_new`, swappable for e.g.
                   a Newton step from an external solver
- `tol`, `maxit` : convergence tolerance / iteration cap on max|excess demand|
- `r_bounds`     : numerical safety bounds imposed on every candidate rate
"""
function solve_mit_shock(p::ModelParams, ss0::SteadyState, ssT::SteadyState,
                          w_path::AbstractMatrix;
                          B_path::AbstractVector=fill(p.B, size(w_path, 2)),
                          r_guess::Union{Nothing,AbstractVector}=nothing,
                          damp::Float64=0.3,
                          tol::Float64=1e-6,
                          maxit::Int=1000,
                          r_bounds::Tuple{Float64,Float64}=(-0.1, 1 / p.β - 1 - 1e-6),
                          price_update::Function=default_price_update,
                          verbose::Bool=true)
    Tp1 = size(w_path, 2)
    T = Tp1 - 1
    @assert size(w_path, 1) == p.n_z
    @assert length(B_path) == Tp1

    r = r_guess === nothing ? collect(range(ss0.r, ssT.r, length=Tp1)) : collect(r_guess)
    r[Tp1] = ssT.r   # terminal condition is always imposed, never solved for

    res = nothing
    for iter in 1:maxit
        res = transition_given_prices(p, ss0, ssT, w_path, r)
        ED = res.A_path[1:T] .- B_path[1:T]
        maxED = length(ED) == 0 ? 0.0 : maximum(abs.(ED))
        verbose && @printf("MIT shock   | iter %4d | max|excess assets| = %.3e\n", iter, maxED)
        if maxED < tol
            return TransitionPath(T, r, Matrix(w_path), res.c_path, res.a_path, res.Phi_path, res.A_path)
        end
        r_new = price_update(r[1:T], ED, damp)
        r[1:T] = clamp.(r_new, r_bounds[1], r_bounds[2])
    end
    @warn "solve_mit_shock: transition did not converge within maxit=$maxit iterations"
    return TransitionPath(T, r, Matrix(w_path), res.c_path, res.a_path, res.Phi_path, res.A_path)
end

end # module HuggettMIT
