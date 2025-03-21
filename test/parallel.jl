using MPI
MPI.Init()
comm = MPI.COMM_WORLD

using ThreadPinning
pinthreads(:affinitymask)

include("mmsomp.jl")

# each process will take care of one initial condition
const seed = [1234, 1234] # 2 processes
const rank = MPI.Comm_rank(comm)

# TODO: implement sweep on wimpact
volfrac = 0.4
rρ = 4.0
rθ = 4.0
wimpact = 0.9
mbb_minimpact_mmsomp(volfrac, rρ, rθ, wimpact; maxiter=1)
@time mbb_minimpact_mmsomp(volfrac, rρ, rθ, wimpact; seed=seed[rank+1], filename="w09_$(rank+1).jld2")
