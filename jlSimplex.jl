load("sparse.jl")
load("linalg_sparse.jl")
load("linalg_suitesparse.jl")
load("glpk.jl") # for reading MPS

typealias ConstraintType Int # why no enum...
typealias VariableState Int
typealias SolverStatus Int

const LB = 1
const UB = 2
const Rnge = 3
const Fixed = 4
const Free = 5

const Basic = 1
const AtLower = 2
const AtUpper = 3

const Uninitialized = 1
const Initialized = 2
const PrimalFeasible = 3
const DualFeasible = 4
const Optimal = 5
const Unbounded = 6

const Below = 1
const Above = 2


type LPData
    c::Vector{Float64} # objective vector 
    l::Vector{Float64}
    u::Vector{Float64}
    boundClass::Vector{ConstraintType}
    A::SparseMatrixCSC{Float64,Int64} # constraint matrix
end

function copy(l::LPData)
    LPData(copy(l.c),copy(l.l),copy(l.u),copy(l.boundClass),copy(l.A))
end

function LPData(c,xlb,xub,l,u,A)
    nrow,ncol = size(A)
    vt = Array(ConstraintType,ncol+nrow)
    function checkType(l::Float64,u::Float64)
        if (l > typemin(Float64))
            if (u < typemax(Float64))
                if (abs(l-u)<1e-7)
                    Fixed
                else
                    Rnge
                end
            else
                LB
            end
        else
            if (u < typemax(Float64))
                UB
            else
                Free
            end
        end
    end
    for i in 1:ncol
        vt[i] = checkType(xlb[i],xub[i])
    end
    for i in 1:nrow
        vt[i+ncol] = checkType(l[i],u[i])
    end

    return LPData([c,zeros(nrow)],[xlb,l],[xub,u],vt,A)
end


type dualSimplexData

    data::LPData
    c::Vector{Float64} # may be perturbed

    nIter::Int # current iteration number
    basicIdx::Vector{Int} # indices of columns in basis
    variableState::Vector{VariableState}
    status::SolverStatus

    x::Vector{Float64}
    d::Vector{Float64}
    dse::Vector{Float64}

    objval::Float64
    phase1::Bool
    didperturb::Bool

    dualTol::Float64
    primalTol::Float64
    zeroTol::Float64

    factor
        

end

function copy(d::dualSimplexData)
    dualSimplexData(copy(d.data),copy(d.c),d.nIter,copy(d.basicIdx),copy(d.variableState),d.status,copy(d.x),copy(d.d),copy(d.dse),d.objval,d.phase1,d.didperturb,d.dualTol,d.primalTol,d.zeroTol,d.factor)
end

function dualSimplexData(d::LPData)
    nrow,ncol = size(d.A)
    state = Array(VariableState,ncol+nrow) # slack basis
    state[ncol+1:nrow+ncol] = Basic
    state[1:ncol] = AtLower
    state[d.boundClass[1:ncol] .== UB] = AtUpper

    return dualSimplexData(d,copy(d.c),
        0, Array(Int,nrow),state,Uninitialized,
        Array(Float64,nrow+ncol),
        Array(Float64,nrow+ncol),
        ones(nrow+ncol), # DSE
        0.,false,false,1e-6,1e-6,1e-12,0)
end


function initialize(d,reinvert::Bool) 
    nrow,ncol = size(d.data.A)
    if (reinvert) 
        d.basicIdx = find(d.variableState .== Basic) # can we do a comprehension with a condition?
        nbasic = length(d.basicIdx)
        @assert length(d.basicIdx) == nrow
        # form basis matrix w/ slacks
        structural = d.data.A[1:nrow,find(d.variableState[1:ncol] .== Basic)] # no ref implementation for bool index matrix?
        slacks = d.basicIdx[d.basicIdx.>ncol]
        colptr = [ structural.colptr[1:(size(structural)[2])], (nnz(structural)+1):(nnz(structural)+length(slacks)+1) ]
        @assert length(colptr) == nrow+1
        rowval = [ structural.rowval[1:nnz(structural)], slacks-ncol ]
        nzval = [ structural.nzval[1:nnz(structural)], -ones(length(slacks)) ]
        d.factor = UmfpackLU!(SparseMatrixCSC(nrow,nrow,colptr,rowval,nzval))
    end

    for i in 1:(nrow+ncol)
        if (d.variableState[i] == Basic) || (d.data.boundClass[i] == Free)
            d.x[i] = 0.
        elseif d.variableState[i] == AtLower
            d.x[i] = d.data.l[i]
        else
            d.x[i] = d.data.u[i]
        end
    end
    #calculate Xb = B^{-1}(b-A_n*x_n)
    xb = zeros(nrow)
    # take linear combination of columns for matrix-vector product
    for i in 1:ncol
        if d.x[i] == 0.
            continue
        end
        for k in d.data.A.colptr[i]:(d.data.A.colptr[i+1]-1)
            xb[d.data.A.rowval[k]] -= d.x[i]*d.data.A.nzval[k]
            #printf("xb[%d] += %f*%f (col %d)\n",d.data.A.rowval[k],d.x[i],d.data.A.nzval[k],i)
        end
    end
    for i in 1:nrow
        #if x[i+ncol] == 0.
        #    continue
        #end
        #if (d.x[i+ncol] != 0.)
        #    printf("xb[%d] -= %f col: %d state: %d class: %d\n",i,d.x[i+ncol],i+ncol,d.variableState[i+ncol],d.data.boundClass[i+ncol])
        #end
        xb[i] += d.x[i+ncol] 
    end

    #println(xb)
    xb = d.factor\xb
    for i in 1:nrow
        d.x[d.basicIdx[i]] = xb[i]
    #    println("$(d.basicIdx[i]): $(xb[i])")
    end
    # calculate y = B^{-T}c_B
    y = d.factor'\d.c[d.basicIdx]

    # calculate dn = cn - An^Ty
    # take dot products with the columns for matrix-vector product
    for i in 1:ncol
        if (d.variableState[i] == Basic)
            d.d[i] = 0.
            continue
        end
        val = 0.
        for k in d.data.A.colptr[i]:(d.data.A.colptr[i+1]-1)
            val += y[d.data.A.rowval[k]]*d.data.A.nzval[k]
        end
        d.d[i] = d.c[i]-val
    end
    for i in 1:nrow
        k = i + ncol
        if (d.variableState[k] == Basic)
            d.d[k] = 0.
            continue
        end
        d.d[k] = d.c[k]+y[i]
    end

    d.objval = dot(d.x,d.c)

    # check dual feasibilites
    dualinfeas = 0.
    ndualinfeas = 0
    for i in 1:(ncol+nrow)
        if (d.variableState[i] == Basic) 
            continue 
        end
        infeas = false
        if (d.data.boundClass[i] == Free && (d.d[i] <-d.dualTol || d.d[i] >d.dualTol))
            #println("$i infeas1")
            infeas=true
        elseif (d.variableState[i] == AtLower && d.d[i] < -d.dualTol && d.data.boundClass[i] != Fixed)
            infeas=true
            #println("$i infeas2")
        elseif (d.variableState[i] == AtUpper && d.d[i] > d.dualTol && d.data.boundClass[i] != Fixed)
            infeas=true
            #println("$i inteas3")
        end
        if (infeas)
            dualinfeas += abs(d.d[i])
            ndualinfeas += 1
        end
    end

    primalinfeas = 0.
    nprimalinfeas = 0
    for i in 1:nrow
        bidx = d.basicIdx[i]
        if (d.x[bidx] < d.data.l[bidx] - d.primalTol)
            primalinfeas += d.data.l[bidx]-d.x[bidx]
            nprimalinfeas += 1
        elseif (d.x[bidx] > d.data.u[bidx] + d.primalTol)
            primalinfeas += d.x[bidx] - d.data.u[bidx]
            nprimalinfeas += 1
        end
    end

    if (dualinfeas > 0)
        if (primalinfeas > 0)
            d.status = Initialized
            print("jlSimplex $(d.nIter) Obj: $(d.objval) Primal inf $primalinfeas ($nprimalinfeas) Dual inf $dualinfeas ($ndualinfeas)")
        else
            d.status = PrimalFeasible
            print("jlSimplex $(d.nIter) Obj: $(d.objval) Dual inf $dualinfeas ($ndualinfeas)")
        end
    else
        if (primalinfeas > 0)
            d.status = DualFeasible
            print("jlSimplex $(d.nIter) Obj: $(d.objval) Primal inf $primalinfeas ($nprimalinfeas)")
        else
            d.status = Optimal
            print("jlSimplex $(d.nIter) Obj: $(d.objval)")
        end
    end
    if (d.phase1)
        print(" (Phase I)\n")
    else
        print("\n")
    end

end


function dualEdgeSelection(d::dualSimplexData)
    nrow,ncol = size(d.data.A)

    rmax::Float64 = 0.
    maxidx = -1
    for k in 1:nrow
        i = d.basicIdx[k]
        r = d.data.l[i] - d.x[i]
        if (r > d.primalTol && r*r > rmax*d.dse[i])
            rmax = r*r/d.dse[i]
            maxidx = k
        else
            r = d.x[i] - d.data.u[i]
            if (r > d.primalTol && r*r > rmax*d.dse[i])
                rmax = r*r/d.dse[i]
                maxidx = k
            end
        end
    end

    return maxidx
end

# two-pass "Harris" ratio test
function dualRatioTest(d::dualSimplexData,alpha2)
    nrow,ncol = size(d.data.A)

    candidates = zeros(Int,ncol)
    ncandidates = 0
    thetaMax = 1e25
    pivotTol = 1e-7

    for i in 1:(ncol+nrow)
        if d.variableState[i] == Basic || d.data.boundClass[i] == Fixed
            continue
        end
        #print("d: $(d.d[i]) alpha: $(alpha2[i])\n")
        if ((d.variableState[i] == AtLower && alpha2[i] > pivotTol) || (d.variableState[i] == AtUpper && alpha2[i] < -pivotTol) || (d.variableState[i] == Free && (alpha2[i] > pivotTol || alpha2[i] < -pivotTol)))
            ratio = 0.
            if (alpha2[i] < 0.)
                ratio = (d.d[i] - d.dualTol)/alpha2[i]
            else
                ratio = (d.d[i] + d.dualTol)/alpha2[i]
            end
            #print("d: $(d.d[i]) alpha: $(alpha2[i]) ratio: $ratio \n")
            if (ratio < thetaMax)
                thetaMax = ratio
                candidates[ncandidates += 1] = i
            end
        end
    end
    #print("$ncandidates candidates, thetaMax = $thetaMax\n")

    # pass 2
    enter = -1
    maxAlpha = 0.
    for k in 1:ncandidates
        i = candidates[k]
        ratio = d.d[i]/alpha2[i]
        if (ratio <= thetaMax)
            absalpha = abs(alpha2[i])
            if (absalpha > maxAlpha)
                maxAlpha = absalpha
                enter = i
            end
        end
    end
    return enter # -1 means unbounded
end

function iterate(d::dualSimplexData)
    nrow,ncol = size(d.data.A)
    @assert (d.status == DualFeasible || d.status == Optimal)

    leave = dualEdgeSelection(d)
    if (leave == -1)
        d.status = Optimal
        return
    end
    leaveIdx = d.basicIdx[leave]
    leaveType = 0
    if (d.x[leaveIdx] > d.data.u[leaveIdx])
        leaveType = Above
    elseif (d.x[leaveIdx] < d.data.l[leaveIdx])
        leaveType = Below
    else
        @assert false
    end

    rho = zeros(nrow)
    rho[leave] = 1.
    rho = d.factor'\rho
    #print("rho: ")
    #for bob in rho
    #    print(bob," ")
    #end
    #println()

    alpha = zeros(ncol+nrow)
    # todo: put in PRICE function
    for i in 1:ncol
        if (d.variableState[i] == Basic)
            continue
        end
        val = 0.
        for k in d.data.A.colptr[i]:(d.data.A.colptr[i+1]-1)
            val += rho[d.data.A.rowval[k]]*d.data.A.nzval[k]
        end
        #println("val: ",i,": ",val)
        alpha[i] = val
    end
    for i in 1:nrow
        k = i+ncol
        if (d.variableState[k] == Basic)
            continue
        end
        alpha[k] = -rho[i]
        #println("val: ",k,": ",alpha[k])
    end

    if leaveType == Below
        alpha = -alpha
    end

    delta = (leaveType == Below) ? (d.x[leaveIdx] - d.data.l[leaveIdx]) : (d.x[leaveIdx] - d.data.u[leaveIdx])
    absdelta = abs(delta)

    enterIdx = dualRatioTest(d,alpha)
    if enterIdx == -1
        print("unbounded?")
        d.status = Unbounded
        @assert false
        return
    end

    #print("enter: $enterIdx leave: $leaveIdx delta: $delta\n")

    # TODO: do updates

    d.variableState[enterIdx] = Basic
    if leaveType == Below
        d.x[leaveIdx] = d.data.l[leaveIdx]
        d.variableState[leaveIdx] = AtLower
    else
        d.x[leaveIdx] = d.data.u[leaveIdx]
        d.variableState[leaveIdx] = AtUpper
    end


end

function go(d::dualSimplexData)
    if (d.status == Uninitialized)
        initialize(d,true)
    end


    if (d.status == Initialized) 
        makeFeasible(d)
    end
    
    for d.nIter in 1:100000
        iterate(d)
        initialize(d,true)
        if (d.status == Optimal)
            break
        end
        if (d.status != DualFeasible)
            perturbForFeasibility(d)
        end
    end

    d.c = copy(d.data.c)
    initialize(d,true)
    if (d.status != Optimal)
        println("Oops, lost optimality after removing perturbations")
        # TODO: switch to primal simplex
    end

end

function perturbForFeasibility(d::dualSimplexData)
    nrow,ncol = size(d.data.A)
    didflip = false
    for i in 1:(ncol+nrow)
        if (d.variableState[i] == Basic) 
            continue 
        end
        if (d.variableState[i] == AtLower || d.data.boundClass[i] == Free)
            if (d.d[i] < -d.dualTol)
                delta = -d.d[i]-d.dualTol
                d.c[i] += delta
                d.d[i] = -d.dualTol
            end
        end
        if (d.variableState[i] == AtUpper || d.data.boundClass[i] == Free)
            if (d.d[i] > d.dualTol)
                delta = -d.d[i]+d.dualTol
                d.c[i] += delta
                d.d[i] = d.dualTol
            end
        end
    end

    if (d.status == Initialized)
        d.status = DualFeasible
    else
        @assert d.status == PrimalFeasible
        d.status = Optimal
    end

end

function makeFeasible(d::dualSimplexData)
    nrow,ncol = size(d.data.A)
    
    if (d.status == DualFeasible)
        return
    end

    initialize(d,false)
    flipBounds(d)

    if (d.status == DualFeasible)
        return
    end

    @assert !d.phase1

    d2 = copy(d)
    d2.phase1 = true
    for i in 1:(nrow+ncol)
        t = d.data.boundClass[i]
        d2.data.boundClass[i] = Rnge
        if (t == Rnge || t == Fixed) 
            d2.data.l[i] = 0.
            d2.data.u[i] = 0.
            d2.data.boundClass[i] = Fixed
        elseif (t == LB)
            assert(d.data.u[i] == typemax(Float64))
            d2.data.l[i] = 0.
            d2.data.u[i] = 1.
        elseif (t == UB)
            assert(d.data.l[i] == typemin(Float64))
            d2.data.l[i] = -1.
            d2.data.u[i] = 0.
        elseif (t == Free)
            d2.data.l[i] = -1000.
            d2.data.u[i] = 1000.
        end
    end
    go(d2)
    @assert d2.status == Optimal
    ## TODO: fix for FINNIS
    d.variableState = d2.variableState
    for i in 1:(nrow+ncol)
        t = d.data.boundClass[i]
        if (d.variableState[i] == AtLower && t == UB)
            println("oops need to flip")
        elseif (d.variableState[i] == AtUpper && t == LB)
            println("oops need to flip2")
        end
    end
    d.c = d2.c
    initialize(d,true)
    flipBounds(d)


end

function flipBounds(d::dualSimplexData)
    nrow,ncol = size(d.data.A)
    didflip = false
    for i in 1:(ncol+nrow)
        if (d.variableState[i] == Basic) 
            continue 
        end
        infeas = false
        if (d.data.boundClass[i] == Free && (d.d[i] <-d.dualTol || d.d[i] >d.dualTol))
            infeas=true
        elseif (d.variableState[i] == AtLower && d.d[i] < -d.dualTol && d.data.boundClass[i] != Fixed)
            infeas=true
        elseif (d.variableState[i] == AtUpper && d.d[i] > d.dualTol && d.data.boundClass[i] != Fixed)
            infeas=true
        end
        if (infeas && (d.data.boundClass[i] == Rnge || d.data.boundClass[i] == Fixed))
            didflip = true
            if (d.variableState[i] == AtLower)
                d.variableState[i] = AtUpper
            elseif (d.variableState[i] == AtUpper)
                d.variableState[i] = AtLower
            end
        end
    end

    if (didflip)
        initialize(d,false)
    end
end



function LPDataFromMPS(mpsfile::String) 

    lp = GLPProb()
    ret = glp_read_mps(lp,GLP_MPS_FILE,C_NULL,mpsfile)
    @assert ret == 0
    nrow::Int = glp_get_num_rows(lp)
    nrow = nrow - 1 # gkpk wtf
    ncol::Int = glp_get_num_cols(lp)
    
    index1 = Array(Int32,nrow)
    coef1 = Array(Float64,nrow)
    
    
    starts = Array(Int64,ncol+1)
    idx = Array(Int64,0)
    elt = Array(Float64,0)
    nnz = 0

    c = Array(Float64,ncol)
    xlb = Array(Float64,ncol)
    xub = Array(Float64,ncol)
    l = Array(Float64,nrow)  
    u = Array(Float64,nrow)
    for i in 1:ncol
        c[i] = glp_get_obj_coef(lp,i)
        t = glp_get_col_type(lp,i)
        if t == GLP_FR
            xlb[i] = typemin(Float64)
            xub[i] = typemax(Float64)
        elseif t == GLP_UP
            xlb[i] = typemin(Float64)
            xub[i] = glp_get_col_ub(lp,i)
        elseif t == GLP_LO
            xlb[i] = glp_get_col_lb(lp,i)
            xub[i] = typemax(Float64)
        elseif t == GLP_DB || t == GLP_FX
            xlb[i] = glp_get_col_lb(lp,i)
            xub[i] = glp_get_col_ub(lp,i)
        end
    end

    objname = glp_get_obj_name(lp)
    glp_create_index(lp)
    objrow = glp_find_row(lp,objname)
    println("objective is in row $objrow")

    for i in 1:nrow
        reali = i
        if (i >= objrow)
            reali += 1
        end
        t = glp_get_row_type(lp,reali)
        if t == GLP_UP
            l[i] = typemin(Float64)
            u[i] = glp_get_row_ub(lp,reali)
        elseif t == GLP_LO
            l[i] = glp_get_row_lb(lp,reali)
            u[i] = typemax(Float64)
        elseif t == GLP_DB || t == GLP_FX
            l[i] = glp_get_row_lb(lp,reali)
            u[i] = glp_get_row_ub(lp,reali)
        end
    end

    
    sel = Array(Bool,nrow)
    for i in 1:ncol
        starts[i] = nnz+1
        nnz1 = glp_get_mat_col(lp,i,index1,coef1)
        sel[:] = false
        for k in 1:nnz1
            if (index1[k] != objrow)
                sel[k] = true
            end
            if (index1[k] > objrow)
                index1[k] -= 1
            end
        end
        nnz1 = sum(sel)
        idx = [idx,index1[sel]]
        elt = [elt,coef1[sel]]
        nnz += nnz1
    end
    starts[ncol+1] = nnz+1

    A = SparseMatrixCSC(nrow,ncol,starts,idx,elt)

    return LPData(c,xlb,xub,l,u,A)


end
