load("jlSimplex.jl")

d = dualSimplexData(LPDataFromMPS("AFIRO.SIF"))

go(d)

