using RBM, Utils, LogReg

const NUM_LABELS = 20

function fit_and_save_rbm(fname, hdim::Int=3000, maxit::Int=20)
    train, valid, test, lbldict, vdim = readsplits()
    θ = RBM.RBMParams(vdim, hdim)
    println("vdim: $vdim, hdim: $hdim")
    trainX = [x::Array{Float64,1} for (x,y) in train]
    validX = [x::Array{Float64,1} for (x,y) in valid[1:100]]
    println("num train: $(length(trainX))")
    println("num valid: $(length(validX))")
    println("training")
    try
        RBM.fit!(θ, trainX, validX, 0.001, 0.7, 0.02, 0.1, 0.9, maxit=maxit, freq=1)
    catch err
        println(err)
    end
    serialize(open(fname, "w"), θ)
end

function make_latent(rbmfname)
    train, valid, test, lbldict, vdim = readsplits()
    θ = deserialize(open(rbmfname, "r"))
    hdim, vdim = length(θ.c), length(θ.b)
    println("making latent representations")
    println("vdim: $vdim, hdim: $hdim")
    latent_train = [(RBM.prob_h_given_v(θ, x), y::Int) for (x, y) in train]
    latent_valid = [(RBM.prob_h_given_v(θ, x), y::Int) for (x, y) in valid]
    latent_test = [(RBM.prob_h_given_v(θ, x), y::Int) for (x, y) in test]
    latent_train, latent_valid, latent_test, hdim, vdim
end

function logreg_with_latent_experiment()
    results = (Int=>Float64)[]
    rbmfname = "./sparserbm.4000hid.jls"
    stepsize = 50
    train, valid, test, idim, _ = make_latent(rbmfname)

    numtrain = stepsize
    while numtrain <= length(train)
        n = min(length(train), numtrain)
        println("using $n")
        θ = LogReg.Params(idim, NUM_LABELS)
        θ = LogReg.fit!(θ, train[1:n], valid, 0.01, maxit=100)
        errors = 0
        for (x,y) in test
            ŷ = LogReg.predict(θ, x)
            if y != ŷ
                errors += 1
            end
        end
        trainerror = errors / length(test)
        println("error (train) = $trainerror")
        results[n] = trainerror
        numtrain += stepsize
    end
    experiment = {"rbmfname"=>rbmfname,
                  "results"=>results}
    serialize(open("rbmexp.jls", "w"), experiment)
end

function logreg_with_raw_experiment()
    results = (Int=>Float64)[]
    stepsize = 50
    println("reading data")
    train, valid, test, lbldict, idim = readsplits()

    numtrain = stepsize
    while numtrain <= length(train)
        n = min(length(train), numtrain)
        println("using $n")
        θ = LogReg.Params(idim, NUM_LABELS)
        θ = LogReg.fit!(θ, train[1:n], valid, 0.01, maxit=100)
        errors = 0
        for (x,y) in test
            ŷ = LogReg.predict(θ, x)
            if y != ŷ
                errors += 1
            end
        end
        trainerror = errors / length(test)
        println("error (train) = $trainerror")
        results[n] = trainerror
        numtrain += stepsize
    end
    experiment = {"results"=>results}
    serialize(open("lrexp.jls", "w"), experiment)
end


