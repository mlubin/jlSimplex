require("jlSimplex")

function doTests()

    mpsfile = "GREENBEA.SIF"
    d = DualSimplexData(LPDataFromMPS(mpsfile));
    @time go(d)

    println("Now with glpk:")
    SolveMPSWithGLPK(mpsfile)
end

doTests()
