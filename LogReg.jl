module LogReg

function softmax(x)
    etox = exp(x - maximum(x))
    etox / sum(etox)
end

function softmax!(x)
    xmax = maximum(x)
    for i = 1:length(x)
        x[i] = exp(x[i] - xmax)
    end
    Z = sum(x)
    for i = 1:length(x)
        x[i] = x[i] / Z
    end
    x
end

type Params
    W::Array{Float64,2}
    b::Array{Float64,1}
end

Params(idim::Int, odim::Int) = Params(zeros(odim, idim), zeros(odim))

function prob(θ::Params, x::Array{Float64,1})
    y = θ.W * x + θ.b
    softmax!(y)
    y
end
predict(θ::Params, x::Array{Float64,1}) = indmax(θ.W * x + θ.b)

function fit!(θ::Params, 
              train::Array{(Array{Float64,1},Int),1}, 
              valid::Array{(Array{Float64,1},Int),1}, 
              η::Float64; 
              maxit::Int=20,
              tol::Float64=1e-5)
    # [Description]
    # Fit θ using stochastic gradient descent. 
    # Uses early stopping as termination criteria.
    #
    # [Arguments]
    # θ         :the parameters of the model
    # train     :training data
    # valid     :validation data (used for early stopping)
    # η         :learning rate
    # maxit     :maximum number of iterations
    # tol       :tolerance, stop if ll - ll_{best} < tol
    #
    # [Returns]
    # The fit parameters

    epoch = 1
    llbest = typemin(Float64)
    θcurr = deepcopy(θ)
    
    while epoch <= maxit
        shuffle!(train)
        for (x,y) in train
            ŷ = prob(θcurr, x)
            g = -ŷ
            g[y] += 1
            θcurr.W += η * (g * x.')
            θcurr.b += η * g
        end
        
        ll = 0.0
        for (x,y) in valid
            ŷ = prob(θcurr, x)
            ll += log(ŷ[y])
        end
        ll /= length(valid)
        println("epoch: $epoch, ll: $ll")
        if ll - llbest < tol
            # stop, θcurr is more than tol worse than θ
            epoch = maxit + 1
        elseif ll > llbest
            # save the best model 
            llbest = ll
            θ.W = deepcopy(θcurr.W)
            θ.b = deepcopy(θcurr.b)
        end
        epoch += 1
    end
    θ
end

end
