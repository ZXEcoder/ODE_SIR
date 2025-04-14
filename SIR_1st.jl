using Lux, Optimization, OptimizationOptimisers, Zygote, OrdinaryDiffEq, Plots, LuxCUDA,
      SciMLSensitivity, Random, ComponentArrays, Printf, LuxCore
import DiffEqFlux: NeuralODE

# Setup GPU / CPU devices and disable scalar indexing on GPU.
CUDA.allowscalar(false)
const gdev = gpu_device()
const cdev = cpu_device()

# SIR data generation (using Float64 now)
rng = Xoshiro(0)
N = 1000.0
β, γ = 0.3, 0.1
u0 = [999.0, 1.0, 0.0]
tspan = (0.0, 160.0)
tsteps = range(tspan[1], tspan[2], length=161)

function sir_ode!(du, u, p, t)
    S, I, R = u
    β, γ = p
    inv_N = 1.0 / N
    du[1] = -β * S * I * inv_N
    du[2] = β * S * I * inv_N - γ * I
    du[3] = γ * I
end

prob = ODEProblem(sir_ode!, u0, tspan, [β, γ])
sol = solve(prob, Tsit5(); saveat=tsteps)
ode_data = (Array(sol) ./ N) |> gdev

# Define a custom transformation suitable for SIR dynamics.
# It takes the state vector u = [S, I, R] and returns a vector with an extra feature S*I.
#sir_transform(x) = vcat(x, x[1]*x[2])

# NeuralODE model: use the custom sir_transform for a tailored feature expansion.
# The input size increases from 3 to 4, so the first Dense layer is set to accept 4 inputs.
# NeuralODE model
dudt = Chain(
    Dense(3, 64, tanh; init_weight=Lux.glorot_uniform),
    Dense(64, 64, tanh; init_weight=Lux.glorot_uniform),
    Dense(64, 64, tanh; init_weight=Lux.glorot_uniform),
    Dense(64, 3)
)


# Setup model parameters
p, st = Lux.setup(rng, dudt)
p = p |> ComponentArray |> gdev .|> Float64
st = st |> gdev

# Create NeuralODE instance
u0_nn = (u0 ./ N) |> gdev
prob_neuralode = NeuralODE(dudt, tspan, Tsit5(); saveat=tsteps)

# Prediction and loss functions
predict_neuralode(p) = reduce(hcat, first(prob_neuralode(u0_nn, p, st)).u)

# L2 regularization strength (tunable)
#λ = 1e-4

function loss_neuralode(p)
    pred = predict_neuralode(p)
    data_loss = sum(abs2, ode_data .- pred)
    #l2_penalty = sum(x -> sum(abs2, x), values(p))
    return data_loss #+ λ * l2_penalty
end

# Training visualization callback
function verbose_callback(state, l; plot_every=50)
    p = state.u
    global iter
    pred = predict_neuralode(p)
    iter += 1
    if iter == 1
        @printf "%-8s %-12s\n" "Iter" "Loss"
    end
    @printf "%-8d %-12.6f\n" iter l
    if iter % plot_every == 0
        plt = plot(layout=(3,1), size=(800,600), dpi=150)
        for (i, label) in enumerate(["Susceptible", "Infected", "Recovered"])
            plot!(plt[i], tsteps, Array(ode_data[i, :]), label="True $label", linewidth=2, color=:black)
            plot!(plt[i], tsteps, Array(pred[i, :]), label="Predicted $label", linestyle=:dash, linewidth=2, color=:red)
            ylabel!(plt[i], "Count")
            if i == 1
                title!(plt[i], "Training Progress (Iter $iter)")
            end
        end
        xlabel!(plt[3], "Time (days)")
        display(plt)
    end
    return false
end

# Setup optimization
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

# === Two-Phase Training ===
# Phase 1: Initial training with Adam(0.01)
println("\n--- Initial Training: Adam(0.01) ---\n")
iter = 0
optprob = Optimization.OptimizationProblem(optf, p)

result1 = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.01),
    callback = verbose_callback,
    maxiters = 5000,
)

# Phase 2: Fine-tuning with Adam(0.001)
using OptimizationOptimJL
println("\n--- Fine-Tuning: BFGS ---\n")
iter = 0
optprob2 = Optimization.OptimizationProblem(optf, result1.u)

result2 = Optimization.solve(
    optprob2,
    Optim.BFGS(; initial_stepnorm = 0.05),
    callback = verbose_callback,
    maxiters = 200,
)

# Final prediction and plotting
p_trained = result2.u
pred_trained = predict_neuralode(p_trained)
pred_cpu = pred_trained |> cdev

# Plot true vs predicted SIR dynamics
plt_true = plot(sol, label=["S" "I" "R"], title="True SIR Model", linewidth=2)
plt_neural = plot(tsteps, pred_cpu[1, :], label="S (Neural)", linestyle=:dash)
plot!(plt_neural, tsteps, pred_cpu[2, :], label="I (Neural)", linestyle=:dash)
plot!(plt_neural, tsteps, pred_cpu[3, :], label="R (Neural)", linestyle=:dash)
title!("Neural ODE Predictions (After Fine-Tuning)")
plot(plt_true, plt_neural, layout=(2,1), size=(800,600))
