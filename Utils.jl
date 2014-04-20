module Utils

export hashvec, stidict, getdata, approxsplit, dumpsplits, readsplits

function hashvec(words, dim)
    # hashing trick
    x = zeros(dim)
    for word in words
        i = int(rem(hash(word), dim)) + 1
        x[i] = 1
    end
    x
end

function stidict(strings)
    # return a dictionary mapping each string 
    # in strings to a unique integer
    dict = (String=>Int)[]
    nextint = 1

    for string in strings
        if !haskey(dict, string)
            dict[string] = nextint
            nextint += 1
        end
    end

    dict
end

function getdata()
    # return an array of (x,y) pairs where 
    # x is a document and
    # y is the topic
    text = open(readlines, "./data/20news.txt")
    
    lbls = [strip(text[i]) for i = 1:2:length(text)]
    docs = [split(strip(text[i+1])) for i = 1:2:length(text)]
    
    @assert length(lbls) == length(docs)

    collect(zip(docs, lbls))
end

function approxsplit(things, p)
    # split things into length(p) chunks such that 
    # length(chunks[i]) ≈ length(things) * p[i]
    chunks = (typeof(things))[]
    l = length(things)
    s1 = 0
    for i = 1:length(p) - 1
        s2 = ifloor(s1 + l * p[i])
        push!(chunks, things[s1 + 1:s2])
        s1 = s2
    end
    push!(chunks, things[s1:end])

    chunks
end

function dumpsplits(dim::Int=5000, p::Array{Float64,1}=[0.7,0.1,0.2])
    D = getdata()
    lbldict = stidict(map((x)->x[2], D))
    train, valid, test = approxsplit(D, p)
    
    ntrain, nvalid, ntest = length(train), length(valid), length(test)
    println("ntrain: $ntrain, nvalid: $nvalid, ntest: $ntest")
    
    htrain = [(hashvec(x, dim), lbldict[y]) for (x,y) in train] 
    hvalid = [(hashvec(x, dim), lbldict[y]) for (x,y) in valid]
    htest = [(hashvec(x, dim), lbldict[y]) for (x,y) in test]

    stuff = (htrain, hvalid, htest, lbldict, dim)
    serialize(open("data.jls", "w"), stuff)
end

function readsplits()
    train, valid, test, lbldict, dim = deserialize(open("data.jls", "r"))
    ntrain, nvalid, ntest = length(train), length(valid), length(test)
    println("ntrain: $ntrain, nvalid: $nvalid, ntest: $ntest")
    train, valid, test, lbldict, dim
end

end
