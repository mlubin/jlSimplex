load("sparse.jl")
load("linalg_sparse.jl")
load("suitesparse.jl")

type PackedEtaVector
    elts::Vector{Float64}
    idx::Vector{Int64}
    nnz::Int64
    pivotalIndex::Int64
end

function PackedEtaVector(v::Vector{Float64},pivotalIndex)
    
    n = length(v)
    nnz = 0
    for i in 1:n
        if (abs(v[i]) > 1e-10)
            nnz += 1
        end
    end
    
    idx = Array(Int64,nnz)
    elts = Array(Float64,nnz)
    nnz = 0
    pivot = 1/v[pivotalIndex]
    v[pivotalIndex] = -1. 
    for i in 1:n
        if (abs(v[i]) > 1e-10)
            nnz += 1
            idx[nnz] = i
            elts[nnz] = -pivot*v[i]
        end
    end
    PackedEtaVector(elts,idx,nnz,pivotalIndex)
end

function dot(eta::PackedEtaVector,v::Vector{Float64})
    out = 0.
    for j in 1:eta.nnz
        out += v[eta.idx[j]]*eta.elts[j]
    end
    return out
end

function applyRowEta(eta::PackedEtaVector,v::Vector{Float64})
    v[eta.pivotalIndex] = dot(eta,v)
end

function applyColumnEta(eta::PackedEtaVector,v::Vector{Float64})
    pivot = v[eta.pivotalIndex]
    v[eta.pivotalIndex] = 0.
    for j in 1:eta.nnz
        v[eta.idx[j]] += pivot*eta.elts[j]
    end
end

type PFIManager
    origFactor
    npfi::Int
    etas::Vector{PackedEtaVector}
end

function PFIManager(mat)
    return PFIManager(UmfpackLU!(mat),0,Array(PackedEtaVector,1000))
end

function replaceColumn(pfi::PFIManager,tableauColumn::Vector{Float64},pivotalIndex)
    @assert pfi.npfi < 1000
    pfi.npfi += 1
    pfi.etas[pfi.npfi] = PackedEtaVector(tableauColumn,pivotalIndex)
end

# these destroy the input, doesn't really matter
function (\)(pfi::PFIManager,v::Vector{Float64})
    x = pfi.origFactor \ v
    for i in 1:pfi.npfi
        applyColumnEta(pfi.etas[i],x)
    end
    return x
end

function Ac_ldiv_B(pfi::PFIManager,v::Vector{Float64})
    # what's the syntax for a range with negative increments?
    i = pfi.npfi
    while i > 0
        applyRowEta(pfi.etas[i],v)
        i -= 1
    end

    return pfi.origFactor'\v
end
   
