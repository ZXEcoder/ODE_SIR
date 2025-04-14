using Lux, Optimization, OptimizationOptimisers, Zygote, OrdinaryDiffEq, Plots, LuxCUDA,
    SciMLSensitivity, Random, ComponentArrays, Printf, LuxCore
import DiffEqFlux: NeuralODE

# ------------------------------------------------------------
# Setup GPU / CPU Devices
# ------------------------------------------------------------
CUDA.allowscalar(false)
const gdev = gpu_device()
const cdev = cpu_device()

# ------------------------------------------------------------
# SIR Model Data Generation (using Float64)
# ------------------------------------------------------------
rng = Xoshiro(0)
N = 1000.0
β, γ = 0.3, 0.1
u0 = Float64[999.0, 1.0, 0.0]          # [Susceptible, Infected, Recovered]
tspan = (0.0, 160.0)
tsteps = range(tspan[1], tspan[2], length=161)

function sir_ode!(du, u, p, t)
    S, I, R = u
    β, γ = p
    inv_N = 1.0 / N
    du[1] = -β * S * I * inv_N
    du[2] =  β * S * I * inv_N - γ * I
    du[3] = γ * I
end

prob = ODEProblem(sir_ode!, u0, tspan, [β, γ])
sol = solve(prob, Tsit5(); saveat=tsteps)
# Ensure solution arrays are Float64 and push to GPU
ode_data = (Array(sol) ./ N) |> x -> Float64.(x) |> gdev

# ------------------------------------------------------------
# Define the RealNPU Layer as a Custom Lux Layer
# ------------------------------------------------------------
struct RealNPU <: Lux.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    eps::Float64
end
RealNPU(in_dim::Int, out_dim::Int; eps=1e-7) = RealNPU(in_dim, out_dim, eps)
Lux.initialparameters(rng::AbstractRNG, layer::RealNPU) = (
    W = Float64.(Lux.glorot_uniform(layer.out_dim, layer.in_dim)),
    g = fill(0.5, layer.out_dim, layer.in_dim) # Use Float64 consistently
)
Lux.initialstates(rng::AbstractRNG, layer::RealNPU) = NamedTuple()
Lux.parameterlength(layer::RealNPU) = layer.out_dim * layer.in_dim * 2
Lux.statelength(layer::RealNPU) = 0
function (layer::RealNPU)(x::AbstractArray{T}, p, st) where T <: Real
    x64 = eltype(x) <: Float64 ? x : Float64.(x)
    m, n = layer.out_dim, layer.in_dim
    xresh = reshape(x64, 1, :)
    r = p.g .* (abs.(xresh) .+ layer.eps) .+ (1.0 .- p.g) # Use Float64 consistently
    r_clipped = max.(r, layer.eps) # Ensure r is at least a small positive value
    s = vec(sum(p.W .* log.(r_clipped), dims=2))
    kvec = ifelse.(x64 .< 0, 1.0, 0.0)
    d = p.W * kvec
    return exp.(s) .* cos.(π .* d)
end
function (layer::RealNPU)(x::AbstractVector)
    x_vec = x isa Number ? [Float64(x)] : (eltype(x) <: Float64 ? x : Float64.(x))
    return layer(x_vec, Lux.initialparameters(Random.GLOBAL_RNG, layer), NamedTuple())
end

# ------------------------------------------------------------
# Define the NeuralODE Model Using RealNPU Layers
# ------------------------------------------------------------
dudt = Chain(
    RealNPU(3, 128), # Increased number of neurons
    x -> tanh.(x),
    RealNPU(128, 128), # Increased number of neurons
    x -> tanh.(x),
    RealNPU(128, 3) # Increased number of neurons
)

# ------------------------------------------------------------
# Setup Model Parameters and State via Lux
# ------------------------------------------------------------
p, st = Lux.setup(rng, dudt)
p = p |> ComponentArray |> gdev # Remove explicit Float32 conversion
st = st |> gdev

# Create NeuralODE instance.
u0_nn = (u0 ./ N) |> gdev
prob_neuralode = NeuralODE(dudt, tspan, Tsit5(); saveat=tsteps)

# ------------------------------------------------------------
# Define Prediction and Loss Functions
# ------------------------------------------------------------
function predict_neuralode(p)
    # Transfer parameters and initial state to CPU for ODE solving (can be done on GPU if Tsit5 supports)
    p_cpu = p |> cdev
    u0_cpu = u0_nn |> cdev |> x -> convert(Vector{Float64}, x)
    st_cpu = st |> cdev

    function dudt_cpu(u, _p, t)
        layer1_out = RealNPU(3, 128)(u, _p.layer_1, st_cpu.layer_1)
        layer2_out = tanh.(layer1_out)
        layer3_out = RealNPU(128, 128)(layer2_out, _p.layer_3, st_cpu.layer_3)
        layer4_out = tanh.(layer3_out)
        layer5_out = RealNPU(128, 3)(layer4_out, _p.layer_5, st_cpu.layer_5)
        return layer5_out
    end

    prob_neuralode_cpu = ODEProblem(dudt_cpu, u0_cpu, tspan, p_cpu)
    sol = solve(prob_neuralode_cpu, Tsit5(); saveat=tsteps)
    pred_cpu = reduce(hcat, sol.u)
    # Transfer prediction back to GPU
    return pred_cpu |> gdev
end

λ = 1e-4 # Increased L2 regularization

function loss_neuralode(p)
    pred = predict_neuralode(p)
    data_loss = sum(abs2, ode_data .- pred)
    l2_penalty = 0.0
    p_cpu = p |> cdev
    for layer_params in p_cpu
        if layer_params isa NamedTuple && haskey(layer_params, :W)
            l2_penalty += sum(abs2, layer_params.W)
        end
    end
    return data_loss + λ * l2_penalty
end

# ------------------------------------------------------------
# Training Visualization Callback
# ------------------------------------------------------------
function verbose_callback(state, l; plot_every=50)
    global iter
    pred = predict_neuralode(state.u)
    iter += 1
    if iter == 1
        @printf("%-8s %-12s\n", "Iter", "Loss")
    end
    @printf("%-8d %-12.6f\n", iter, l)
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

# ------------------------------------------------------------
# Training Setup Using the Optimization Package
# ------------------------------------------------------------
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

println("\n--- Initial Training: Adam(0.005) ---\n") # Slightly lower learning rate
iter = 0
optprob = Optimization.OptimizationProblem(optf, p)
result1 = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.005),
    callback = verbose_callback,
    maxiters = 100, # Increased max iterations
)
using OptimizationOptimJL
println("\n--- Fine-Tuning: BFGS(0.01) ---\n") # Smaller initial step norm
iter = 0
optprob2 = Optimization.OptimizationProblem(optf, result1.u)
result2 = Optimization.solve(
    optprob2,
    Optim.BFGS(; initial_stepnorm = 0.01),
    callback = verbose_callback,
    maxiters = 50, # Increased max iterations for fine-tuning
)

# ------------------------------------------------------------
# Final Prediction and Plotting
# ------------------------------------------------------------
p_trained = result2.u
pred_trained = predict_neuralode(p_trained)
pred_cpu = pred_trained |> cdev

plt_true = plot(sol, label=["S" "I" "R"], title="True SIR Model", linewidth=2)
plt_neural = plot(tsteps, pred_cpu[1, :], label="S (Neural)", linestyle=:dash)
plot!(plt_neural, tsteps, pred_cpu[2, :], label="I (Neural)", linestyle=:dash)
plot!(plt_neural, tsteps, pred_cpu[3, :], label="R (Neural)", linestyle=:dash)
title!("Neural ODE Predictions (Using RealNPU Layers)")
plot(plt_true, plt_neural, layout=(2,1), size=(800,600))
