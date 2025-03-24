using MPI
MPI.Init()
comm = MPI.COMM_WORLD

using ThreadPinning
pinthreads(:affinitymask)

include("mmsomp3d.jl")

# each process will take care of one initial condition
const seed = [1, 12] # 2 processes
const rank = MPI.Comm_rank(comm)

# TODO: implement sweep on wimpact
volfrac = 0.4
rρ = 4.0
rθ = 4.0
wimpact = 0.5
mbb_minimpact_mmsomp3d(volfrac, rρ, rθ, wimpact; maxiter=1, echo=false) # precompile
mbb_minimpact_mmsomp3d(volfrac, rρ, rθ, wimpact; seed=seed[rank+1], filename="results/w05_$(rank+1)")
