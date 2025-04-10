using Logging
logger = SimpleLogger(Logging.Warn)
global_logger(logger)

using MPI
MPI.Init()
comm = MPI.COMM_WORLD

using ThreadPinning
pinthreads(:affinitymask)

using Printf
include("mmsomp.jl")

# each process will work with one weight
const rank = MPI.Comm_rank(comm)
const w_list = 0.1:0.1:0.9
const wimpact = w_list[rank+1]

# path to save results
const SCRATCH = ENV["SCRATCH"]
mkpath(SCRATCH * "results/")

const angles = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
const volfrac = 0.4
const rρ = 4.0
const rθ = 2.0

compliances = Float64[]
impacts = Float64[]
for angle in angles
    filename = @sprintf "w%02d_%d" 10 * wimpact angle
    filename = SCRATCH * "results/" * replace(filename, "-" => "n")

    history = mbb_minimpact_mmsomp(volfrac, rρ, rθ, wimpact; angle, filename)
    push!(compliances, history[:final_compliance])
    push!(impacts, history[:impact][end])
end

MPI.Barrier(comm)
all_compliances = MPI.Gather(compliances, comm)
all_impacts = MPI.Gather(impacts, comm)
all_w = MPI.Gather(fill(wimpact, length(angles)), comm)
all_angles = MPI.Gather(angles, comm)
if rank == 0
    @show all_compliances
    @show all_impacts
    @show all_w
    @show all_angles
end
