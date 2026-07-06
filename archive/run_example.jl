#!/usr/bin/env julia
#
# Example: MIT-shock transition path for a discrete-income Huggett economy.
#
# Run with:   julia run_example.jl
#
# This script only uses the Julia standard library (LinearAlgebra,
# SparseArrays, Printf), so it should run with any recent Julia (1.6+)
# without installing any packages.

include("HuggettMIT.jl")
using .HuggettMIT
using Printf
using Plots
# -------------------------------------------------------------------
# 1. Idiosyncratic income process — SUPPLIED AS PARAMETERS
# -------------------------------------------------------------------
# Any discrete Markov chain works: just hand in the number of states,
# the wage/income level in each state, and the transition matrix.
# Below is a persistent 3-state productivity process (e.g. low/mid/high)
# calibrated so its stationary distribution is roughly [0.25, 0.5, 0.25].

n_z  = 3
z    = [0.5, 1.0, 1.5]                 # relative productivity per state
w_ss = z                               # wage = productivity (linear labor income)
Π    = [0.90 0.08 0.02;
        0.05 0.90 0.05;
        0.02 0.08 0.90]

# -------------------------------------------------------------------
# 2. Build model parameters and solve the initial steady state
# -------------------------------------------------------------------

p = build_params(
    β       = 0.96,
    γ       = 0.8,
    w       = w_ss,
    Π       = Π,
    a_min   = -1.0,     # natural-ish borrowing limit
    a_max   = 30.0,
    n_a     = 150,
    B       = 2.0,      # pure risk-sharing Huggett economy: zero net bond supply
    curvature = 2.0,
)

println("Solving initial steady state...")
ss0 = solve_steady_state(p; r_lo=-0.05, r_hi=1/p.β - 1 - 1e-4, verbose=true)
@printf("\nInitial steady state: r* = %.4f%%,  aggregate assets = %.4f\n\n", 100*ss0.r, ss0.A)

# -------------------------------------------------------------------
# 3. Define the MIT shock as a TIME PATH of the wage vector
# -------------------------------------------------------------------
# Example: an unanticipated, temporary aggregate income shock that hits
# at t=0 (a "recession"), scaling every state's wage down by 10% on
# impact and fading away geometrically at rate ρ. Because the shock is
# purely transitory, the terminal steady state equals the initial one.

T        = 125                          # transition length (periods after impact)
shock0   = -0.10                        # -10% wages on impact
ρ        = 0.85                         # persistence of the shock

w_path = Matrix{Float64}(undef, n_z, T+1)
for t in 1:T+1
    mult = 1 + shock0 * ρ^(t-1)
    w_path[:, t] = w_ss .* mult
end

ssT = ss0   # transitory shock ⇒ same long-run steady state
            # (for a PERMANENT shock instead: build a second ModelParams
            #  with the new long-run w/Π and call solve_steady_state on it,
            #  then pass that as ssT below, and make w_path[:, end] match it)

# -------------------------------------------------------------------
# 4. Solve for the perfect-foresight transition path
# -------------------------------------------------------------------

println("Solving MIT-shock transition path...")
path = solve_mit_shock(p, ss0, ssT, w_path;
                        damp = 0.0005,
                        tol  = 1e-6,
                        maxit = 10000,
                        verbose = true)

# -------------------------------------------------------------------
# 5. Inspect results
# -------------------------------------------------------------------

println("\nPeriod-by-period interest rate and aggregate assets (first 15 periods):")
println(" t      r_t (%)     A_t")
for t in 1:path.T+1
    @printf("%3d    %8.4f   %8.4f\n", t-1, 100*path.r_path[t], path.A_path[t])
end

@printf("\nTerminal check: r_T = %.4f%% (should equal new steady-state r = %.4f%%)\n",
        100*path.r_path[end], 100*ssT.r)

# Aggregate consumption path (nice sanity-check quantity): C_t = sum_{a,z} c_t(a,z) Phi_t(a,z)
C_path = [sum(path.c_path[:,:,t] .* path.Phi_path[:,:,t]) for t in 1:path.T+1]
println("\nAggregate consumption, first 10 periods:")
println(round.(C_path[1:10], digits=4))

# To plot (requires the Plots.jl package, not a dependency of this code):
#
#   using Plots
  display(plot(0:path.T, 100 .* path.r_path, xlabel="t", ylabel="r_t (%)",
       title="MIT shock: interest rate path", legend=false))
  sleep(10)
  display(plot(0:path.T, path.A_path, xlabel="t", ylabel="aggregate assets",
       title="MIT shock: asset market clearing", legend=false))
  sleep(10)
