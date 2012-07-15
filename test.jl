load("jlSimplex.jl")

mpsfile = "GREENBEA.SIF"
d = dualSimplexData(LPDataFromMPS(mpsfile));
@time go(d)

println("Now with glpk:")
SolveMPSWithGLPK(mpsfile)

