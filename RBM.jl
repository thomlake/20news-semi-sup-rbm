module RBM

sigmoid(x::Float64) = 1.0 / (1.0 + exp(-x))
bernoulli(x::Float64) = rand() < x ? 1.0 : 0.0

function sigmoid!(x::Array{Float64,1})
    n = length(x)
    for i = 1:n
        x[i] = sigmoid(x[i])
    end
end

function bernoulli(x::Array{Float64,1})
    b = zero(x)
    for i = 1:length(x)
        b[i] = bernoulli(x[i])
    end
    b
end

type RBMParams
    W::Array{Float64,2}
    b::Array{Float64,1}
    c::Array{Float64,1}
    ΔW::Array{Float64,2}
    Δb::Array{Float64,1}
    Δc::Array{Float64,1}
    ρ::Array{Float64,1}
end

function RBMParams(W::Array{Float64,2}, b::Array{Float64,1}, c::Array{Float64,1})
    RBMParams(W, b, c, zero(W), zero(b), zero(c), 0.5 * ones(length(c)))
end

RBMParams(vdim::Int, hdim::Int) = RBMParams(0.2 * rand(hdim, vdim) - 0.1, zeros(vdim), zeros(hdim))

function free_energy(θ::RBMParams, v::Array{Float64,1})
    t1 = dot(θ.b, v)
    t2 = sum(log(1 + exp(θ.W * v + θ.c)))
    -t1 - t2
end

function prob_h_given_v(θ::RBMParams, v::Array{Float64,1})
    h = θ.W * v + θ.c
    sigmoid!(h)
    h
end

function prob_v_given_h(θ::RBMParams, h::Array{Float64,1})
    v = θ.W' * h + θ.b
    sigmoid!(v)
    v
end

function reconstruction_loglikelihood(θ::RBMParams, x::Array{Float64,1})
    y = prob_v_given_h(θ, bernoulli(prob_h_given_v(θ, x)))
    ll = 0
    for i = 1:length(y)
        ll += (x[i] == 1.0) ? log(y[i]) : log(1 - y[i])
    end
    ll / length(y)
end

function cdk!(θ::RBMParams, 
              x::Array{Float64,1}, 
              η::Float64, 
              μ::Float64, 
              σ::Float64, 
              τ::Float64,
              δ::Float64=0.9,
              k::Int=1)
    # [Description]
    # Contrastive Divergence update of θ with sparsity.
    #
    # [Arguments]
    # θ :the parameters of the model
    # x :visible vector
    # η :learning rate
    # μ :momentum
    # σ :sparse penalty coefficient
    # τ :sparsity target
    # δ :decay rate for exponential moving average
    #    estimate of hidden variable activity
    # k :number of gibbs steps for negative samples (defaults to 1)
    #
    # [Returns]
    # nothing, θ is updated in place

    # positive and negative phase
    ph = prob_h_given_v(θ, x)
    nv = prob_v_given_h(θ, bernoulli(ph))
    nh = prob_h_given_v(θ, bernoulli(nv))
    for i = 2:k
        nv = prob_v_given_h(θ, bernoulli(nh))
        nh = prob_h_given_v(θ, bernoulli(nv))
    end 
    
    # update the exponential moving average
    θ.ρ = δ * θ.ρ + (1 - δ) * ph
    # compute the sparsity penalty
    s = σ * (θ.ρ - τ)

    # unrolling the paramter update is approximately 
    # 6 times faster than an equivalent vectorized update.
    hdim, vdim = size(θ.W)
    for j = 1:vdim
        for i = 1:hdim
            θ.ΔW[i,j] = μ * θ.ΔW[i,j] + η * ((ph[i] - s[i]) * x[j] - nh[i] * nv[j])
            θ.W[i,j] += θ.ΔW[i,j]
        end
    end

    for j = 1:vdim
        θ.Δb[j] = μ * θ.Δb[j] + η * (x[j] - nv[j])
        θ.b[j] += θ.Δb[j]
    end

    for i = 1:hdim
        θ.Δc[i] = μ * θ.Δc[i] + η * (ph[i] - nh[i] - s[i])
        θ.c[i] += θ.Δc[i]
    end
end

function fit!(θ::RBMParams, 
              train::Array{Array{Float64,1},1}, 
              valid::Array{Array{Float64,1},1},
              η::Float64, 
              μ::Float64, 
              σ::Float64, 
              τ::Float64,
              δ::Float64,
              k::Int=1;
              maxit::Int=20,
              nest::Int=50,
              freq::Int=5,
              tol::Float64=1e-5)
    # [Description]
    # Fit θ using stochastic gradient descent.
    # Uses Contrastive Divergence with sparsity and
    # reconstruction error as termination criteria.
    #
    # [Arguments]
    # θ     :the parameters of the model
    # train :training data
    # valid :validation data
    # η     :learning rate
    # μ     :momentum
    # σ     :sparse penalty coefficient
    # τ     :sparsity target
    # δ     :decay rate for exponential moving average
    #        estimate of hidden variable activity
    # k     :number of gibbs steps for negative samples (defaults to 1)
    # maxit :maximum number of iterations
    # freq  :compute reconstruction likelihood every freq epochs
    # tol   :tolerance, stop if ll - ll_{best} < tol
    #
    # [Returns]
    # The best parameters as measured by reconstruction loglikelihood.

    # Copy of θ to θcurr for working and copy back when a
    # better set of parameters is found. A little awkward,
    # but doing it this way ensures that if caller catches an error,
    # their θ will be the best θcurr so far.
    epoch = 1
    θcurr = deepcopy(θ)
    llbest = typemin(Float64)
    while epoch <= maxit
        shuffle!(train)
        for x in train
            cdk!(θcurr, x, η, μ, σ, τ, δ, k)
        end
        if rem(epoch, freq) == 0
            ll = 0.0
            for x in valid
                ll += reconstruction_loglikelihood(θcurr, x)
            end
            ll /= length(valid)
            println("epoch: $epoch, ll: $ll")
            if ll - llbest < tol
                # stop, θcurr isn't tol better than θ
                epoch = maxit + 1
            elseif ll > llbest
                # save the best model 
                llbest = ll
                θ.W = deepcopy(θcurr.W)
                θ.b = deepcopy(θcurr.b)
                θ.c = deepcopy(θcurr.c)
                θ.ΔW = deepcopy(θcurr.ΔW)
                θ.Δb = deepcopy(θcurr.Δb)
                θ.Δc = deepcopy(θcurr.Δc)
                θ.ρ = deepcopy(θcurr.ρ)
            end
        end
        epoch += 1
    end
    θ
end

end
